# NOTE (kbartol): This is only the prediction generation part in AGORA format.
# NOTE (kbartol): Need to take from `demo_bodymocap.py` in the frankmocap repository to run the FrankMocap predictions.
# NOTE (kbartol): Finally, need to also include the predictions made by our method.

from typing import (
    List, 
    Union, 
    Tuple
)
import os
import pickle as pkl
import numpy as np
import torch
import cv2
import sys
import argparse
import shutil

sys.path.append('/garmentor/')

from configs import paths
from configs.const import (
    IMG_WH,
    HEATMAP_GAUSSIAN_STD
)
from configs.poseMF_shapeGaussian_net_config import get_cfg_defaults
from models.smpl_official import SMPL
from models.poseMF_shapeGaussian_net import PoseMFShapeGaussianNet
from models.canny_edge_detector import CannyEdgeDetector
from models.pose2D_hrnet import get_pretrained_detector, PoseHighResolutionNet
from predict.predict_hrnet import predict_hrnet
from utils.label_conversions import convert_2Djoints_to_gaussian_heatmaps

from agora_for_garmentor.agora_evaluation.evaluate_agora import run_evaluation

from frankmocap.bodymocap.body_bbox_detector import BodyPoseEstimator
from frankmocap.bodymocap.body_mocap_api import BodyMocap


PRED_TEMPLATE = '{img_name}_personId_{subject_idx}.pkl'
ZIP_TEMPLATE = '/garmentor/frankmocap/output/agora/predictions/{dirname}.zip'

MOCAP_CHECKPOINT_PATH = '/data/frankmocap/pretrained/frankmocap/2020_05_31-00_50_43-best-51.749683916568756.pt'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_garmentor', action='store_true', default=True,
                        help='whether to use our predictions for shape (and style)')
    parser.add_argument('--pose_shape_weights', '-W3D', type=str, 
                        default='./model_files/poseMF_shapeGaussian_net_weights.tar')
    parser.add_argument('--use_smplx', action='store_true', default=False,
                        help='Whether to regenerate the predictions')
    parser.add_argument('--regenerate', action='store_true', default=False,
                        help='Whether to regenerate the predictions')
    parser.add_argument('--single_person', action='store_true', default=False,
                        help='Used in case of single-person crops which will be used for CAT-3D eval.')
    parser.add_argument('--pred_path', type=str, default=None,
                        help='Path containing the predicitons')
    parser.add_argument('--debug_path', type=str, default=None,
                        help='Path where the debug files will be stored')
    parser.add_argument('--modelFolder', type=str,
                        default='demo/model/smplx')
    parser.add_argument('--numBetas', type=int, default=10)

    parser.add_argument('--result_savePath', type=str, default=None,
                        help='Path where all the results will be saved')
    parser.add_argument(
        '--indices_path',
        type=str,
        default='',
        help='Path to hand,face and body vertex indices for SMPL-X')
    parser.add_argument('--imgHeight', type=int, default=2160,
                        help='Height of the image')
    parser.add_argument('--imgWidth', type=int, default=3840,
                        help='Width of the image')
    parser.add_argument('--numBodyJoints', type=int, default=24,
                        help='Num of body joints used for evaluation')
    parser.add_argument(
        '--imgFolder',
        type=str,
        default='',
        help='Path to the folder containing test/validation images')
    parser.add_argument('--loadPrecomputed', type=str, default='',
                        help='Path to the ground truth SMPL/SMPLX dataframe')
    parser.add_argument(
        '--loadMatched',
        type=str,
        default='',
        help='Path to the dataframe after the matching is done')
    parser.add_argument(
        '--meanPoseBaseline',
        default=False,
        action='store_true')
    parser.add_argument(
        '--onlyComputeErrorLoadPath',
        type=str,
        default='',
        help='Path to the dataframe with all the errors already calculated and stored')
    parser.add_argument(
        '--baseline',
        type=str,
        default='SPIN',
        help='Name of the baseline or the model being evaluated')
    parser.add_argument(
        '--modeltype',
        type=str,
        default='SMPLX',
        help='SMPL or SMPLX')
    parser.add_argument('--kid_template_path', type=str, default='template')
    parser.add_argument('--gt_model_path', type=str, default='')
    parser.add_argument('--onlybfh', action='store_true')
    parser.add_argument('--debug', action='store_true')

    return parser.parse_args()


def run_garmentor(
        garmentor_model: PoseMFShapeGaussianNet,
        kpt_detector: PoseHighResolutionNet,
        kpt_detector_cfg,
        edge_detector: CannyEdgeDetector,
        cropped_imgs: np.ndarray,
        device: torch.device
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    imgs_tensor = torch.from_numpy(cropped_imgs.swapaxes(1, 3)).float().to(device)
    shape_params_list = []
    style_params_list = []
    for img_tensor in imgs_tensor:
        hrnet_output = predict_hrnet(
            hrnet_model=kpt_detector,
            hrnet_config=kpt_detector_cfg,
            image=img_tensor
        )
        joints_2d = hrnet_output['joints2D'].detach().cpu().numpy()[:, ::-1]
        heatmaps = convert_2Djoints_to_gaussian_heatmaps(
            joints2D=joints_2d.round().astype(np.int16),
            img_wh=IMG_WH,
            std=HEATMAP_GAUSSIAN_STD
        )
        #heatmaps_list.append(heatmaps)
        heatmaps = np.expand_dims(np.transpose(heatmaps, [2, 0, 1]), axis=0)
        heatmaps = torch.from_numpy(heatmaps).to(device)
        edge_detector_output = edge_detector(torch.unsqueeze(img_tensor, dim=0))
        edge_map = edge_detector_output['thresholded_thin_edges']  # EDGE_NMS=True
        proxy_rep_input = torch.cat([edge_map, heatmaps], dim=1)

        _, _, _, _, _, pred_shape_dist, pred_style_dist, _, _ = garmentor_model(proxy_rep_input)
        shape_params_list.append(pred_shape_dist.loc[0].detach().cpu().numpy())
        style_params_list.append(pred_style_dist.loc[0].detach().cpu().numpy())

    return shape_params_list, style_params_list


def sort_by_bbox_size(
        body_bbox_list: List[np.ndarray],
        single_person: bool
    ) -> Union[np.ndarray, List[np.ndarray]]:
    #Sort the bbox using bbox size 
    # (to make the order as consistent as possible without tracking)
    bbox_size =  [ (x[2] * x[3]) for x in body_bbox_list]
    idx_big2small = np.argsort(bbox_size)[::-1]
    body_bbox_list = [ body_bbox_list[i] for i in idx_big2small ]
    
    # NOTE (kbartol): In case of AGORA, multi-person, in case of our Agora, single-person.
    if single_person:
        return [body_bbox_list[0], ]
    else:
        return body_bbox_list
    

def check_regenerate(
        regenerate: bool, 
        pred_dir: str
    ) -> bool:
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)
        return True
    if not regenerate:
        if not os.path.exists(ZIP_TEMPLATE.format(dirname=os.path.basename(os.path.normpath(pred_dir)))):
            return True
    return regenerate
    

def save_predictions(
        img_path: str,
        pred_dir: str,
        pred_output_list: List[np.ndarray],
        pred_shape_list: List[np.ndarray],
        pred_style_list: List[np.ndarray]
    ) -> None:
    img_name = os.path.basename(img_path).split('.')[0]
    for subject_idx, pred_output in enumerate(pred_output_list):
        img_cropped = pred_output['img_cropped']
        joints = pred_output['pred_joints_img']
        pose = pred_output['pred_body_pose']
        glob_orient = pred_output['pred_rotmat']
        betas = pred_output['pred_betas']

        pred_dict = {
            'pose2rot': True,
            'joints': joints,
            'params': {
                'transl': np.array([[0., 0., 0.]]),
                'betas': np.expand_dims(pred_shape_list[subject_idx], axis=0),
                'global_orient': pose[None, :, :3],
                'body_pose': np.reshape(pose[:, 3:], (1, 23, 3))
            }
        }
        pred_fpath = os.path.join(
            pred_dir, 
            PRED_TEMPLATE.format(
                img_name=img_name, 
                subject_idx=subject_idx)
        )
        print(f'Saving predictions to {pred_fpath}...')
        with open(pred_fpath, 'wb') as pkl_f:
            pkl.dump(pred_dict, pkl_f, protocol=pkl.HIGHEST_PROTOCOL)


def predict(
        img_dir: str,
        pred_dir: str,
        edge_detector: CannyEdgeDetector,
        kpt_detector: PoseHighResolutionNet,
        kpt_detector_cfg,
        garmentor_model: PoseMFShapeGaussianNet,
        body_bbox_detector: BodyPoseEstimator,
        body_mocap: BodyMocap,
        device: torch.device,
        regenerate: bool = True,
        single_person: bool = False
    ):
    if check_regenerate(
            regenerate=regenerate,
            pred_dir=pred_dir
        ):
        print('(Re-)generating predictions...')
        for img_name in os.listdir(img_dir):
            img_path = os.path.join(img_dir, img_name)
            img = cv2.imread(img_path)
            _, body_bbox_list = body_bbox_detector.detect_body_pose(
                img=img
            )
            body_bbox_list = sort_by_bbox_size(
                body_bbox_list=body_bbox_list,
                single_person=single_person
            )
            pred_output_list = body_mocap.regress(
                img_original=img, 
                body_bbox_list=body_bbox_list
            )
            assert len(body_bbox_list) == len(pred_output_list)
            if len(body_bbox_list) > 0:
                pred_shape, pred_style = run_garmentor(
                    garmentor_model=garmentor_model,
                    kpt_detector=kpt_detector,
                    kpt_detector_cfg=kpt_detector_cfg,
                    edge_detector=edge_detector,
                    cropped_imgs=np.stack(
                        [cv2.resize(x['img_cropped'], (IMG_WH, IMG_WH)) \
                            for x in pred_output_list],
                        axis=0
                    ),
                    device=device
                )
                save_predictions(
                    img_path=img_path,
                    pred_dir=pred_dir,
                    pred_output_list=pred_output_list,
                    pred_shape_list=pred_shape,
                    pred_style_list=pred_style
                )
        shutil.make_archive(
            ZIP_TEMPLATE.format(dirname=os.path.basename(os.path.normpath(pred_dir))), 
            'zip', 
            pred_dir
        )


if __name__ == '__main__':
    args = parse_args()
    pose_shape_cfg = get_cfg_defaults()
    device = torch.device('cuda')

    # Our method initialization.
    edge_detector = CannyEdgeDetector(
        non_max_suppression=pose_shape_cfg.DATA.EDGE_NMS,
        gaussian_filter_std=pose_shape_cfg.DATA.EDGE_GAUSSIAN_STD,
        gaussian_filter_size=pose_shape_cfg.DATA.EDGE_GAUSSIAN_SIZE,
        threshold=pose_shape_cfg.DATA.EDGE_THRESHOLD
    ).to(device)
    smpl_model = SMPL(
        paths.SMPL,
        batch_size=1,
        num_betas=pose_shape_cfg.MODEL.NUM_SMPL_BETAS
    ).to(device)
    smpl_immediate_parents = smpl_model.parents.tolist()
    pose_shape_dist_model = PoseMFShapeGaussianNet(
        smpl_parents=smpl_immediate_parents,
        config=pose_shape_cfg
    ).to(device).eval()
    checkpoint = torch.load(args.pose_shape_weights, map_location=device)
    pose_shape_dist_model.load_state_dict(checkpoint['best_model_state_dict'])
    print('\nLoaded Distribution Predictor weights from', args.pose_shape_weights)

    kpt_detector, kpt_detector_cfg = get_pretrained_detector()

    # FrankMocap initialization.
    body_bbox_detector = BodyPoseEstimator()
    body_mocap = BodyMocap(
        regressor_checkpoint=MOCAP_CHECKPOINT_PATH, 
        smpl_dir=paths.SMPL, 
        device=device, 
        use_smplx=args.use_smplx
    )

    # Predict using the provided models and data.
    predict(
        img_dir=args.imgFolder,
        pred_dir=args.pred_path,
        edge_detector=edge_detector,
        kpt_detector=kpt_detector,
        kpt_detector_cfg=kpt_detector_cfg,
        garmentor_model=pose_shape_dist_model,
        body_bbox_detector=body_bbox_detector,
        body_mocap=body_mocap,
        regenerate=args.regenerate,
        single_person=args.single_person,
        device=device
    )

    # Evaluate AGORA predictions.
    run_evaluation(args)
