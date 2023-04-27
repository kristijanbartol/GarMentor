# NOTE (kbartol): This is only the prediction generation part in AGORA format.
# NOTE (kbartol): Need to take from `demo_bodymocap.py` in the frankmocap repository to run the FrankMocap predictions.
# NOTE (kbartol): Finally, need to also include the predictions made by our method.

from typing import List, Union
import os
import pickle as pkl
import numpy as np
import torch
import cv2
import sys
import argparse
import shutil

sys.path.append('/garmentor/')

from configs.paths import SMPL

from agora_for_garmentor.agora_evaluation.evaluate_agora import run_evaluation

from frankmocap.bodymocap.body_bbox_detector import BodyPoseEstimator
from frankmocap.bodymocap.body_mocap_api import BodyMocap


PRED_TEMPLATE = '{img_name}_personId_{subject_idx}.pkl'
ZIP_TEMPLATE = '/garmentor/frankmocap/output/agora/predictions/{dirname}.zip'

MOCAP_CHECKPOINT_PATH = '/data/frankmocap/pretrained/frankmocap/2020_05_31-00_50_43-best-51.749683916568756.pt'


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
    

def save_predictions(
        img_path: str,
        pred_dir: str,
        pred_output_list: List[np.ndarray]
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
                'betas': betas,
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


def predict(
        img_dir: str,
        pred_dir: str,
        body_bbox_detector: BodyPoseEstimator,
        body_mocap: BodyMocap,
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
            save_predictions(
                img_path=img_path,
                pred_dir=pred_dir,
                pred_output_list=pred_output_list
            )
        shutil.make_archive(
            ZIP_TEMPLATE.format(dirname=pred_dir), 
            'zip', 
            pred_dir
        )


def parse_args():
    parser = argparse.ArgumentParser()
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


if __name__ == '__main__':
    args = parse_args()
    device = torch.device('cuda')
    body_bbox_detector = BodyPoseEstimator()
    body_mocap = BodyMocap(
        regressor_checkpoint=MOCAP_CHECKPOINT_PATH, 
        smpl_dir=SMPL, 
        device=device, 
        use_smplx=args.use_smplx
    )
    predict(
        img_dir=args.imgFolder,
        pred_dir=args.pred_path,
        body_bbox_detector=body_bbox_detector,
        body_mocap=body_mocap,
        regenerate=args.regenerate,
        single_person=args.single_person
    )
    run_evaluation(args)
