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

sys.path.append('/garmentor/')

from configs.paths import SMPL

from frankmocap.bodymocap.body_bbox_detector import BodyPoseEstimator
from frankmocap.bodymocap.body_mocap_api import BodyMocap


#RESULTS_DIR = '/garmentor/frankmocap/results/mocap/'
PREDICTIONS_DIR = '/garmentor/frankmocap/results/agora/predictions/'
PRED_TEMPLATE = '{img_name}_personId_{subject_idx}.pkl'

AGORA_IMG_DIR = '/data/agora/validation/'
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
            PREDICTIONS_DIR, 
            PRED_TEMPLATE.format(
                img_name=img_name, 
                subject_idx=subject_idx)
        )
        print(f'Saving predictions to {pred_fpath}...')
        with open(pred_fpath, 'wb') as pkl_f:
            pkl.dump(pred_dict, pkl_f, protocol=pkl.HIGHEST_PROTOCOL)


def check_regenerate(regenerate: bool):
    if not os.path.exists(PREDICTIONS_DIR):
        os.makedirs(PREDICTIONS_DIR)
        return True
    if not regenerate:
        if len(os.listdir(PREDICTIONS_DIR)) == 0:
            return True
    return regenerate


def evaluate_agora(
        img_dir: str,
        body_bbox_detector: BodyPoseEstimator,
        body_mocap: BodyMocap,
        regenerate: bool = True,
        single_person: bool = False
    ):
    if check_regenerate(regenerate):
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
                pred_output_list=pred_output_list
            )
    


if __name__ == '__main__':
    device = torch.device('cuda')
    body_bbox_detector = BodyPoseEstimator()
    body_mocap = BodyMocap(
        regressor_checkpoint=MOCAP_CHECKPOINT_PATH, 
        smpl_dir=SMPL, 
        device=device, 
        use_smplx=False
    )
    evaluate_agora(
        img_dir=AGORA_IMG_DIR,
        body_bbox_detector=body_bbox_detector,
        body_mocap=body_mocap,
        regenerate=False,
        single_person=False
    )
