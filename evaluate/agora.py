# NOTE (kbartol): This is only the prediction generation part in AGORA format.
# NOTE (kbartol): Need to take from `demo_bodymocap.py` in the frankmocap repository to run the FrankMocap predictions.
# NOTE (kbartol): Finally, need to also include the predictions made by our method.

import os
import pickle as pkl
import numpy as np


RESULTS_DIR = '/frankmocap/results/mocap/'
PREPARED_DIR = '/frankmocap/results/predictions/'
PRED_TEMPLATE = '{img_name}_personId_{subject_idx}.pkl'

if __name__ == '__main__':
    for fname in os.listdir(RESULTS_DIR):
        fpath = os.path.join(RESULTS_DIR, fname)
        with open(fpath, 'rb') as pkl_f:
            data = pkl.load(pkl_f)

        img_name = os.path.basename(data['image_path']).split('.')[0]

        for subject_idx, pred_output in enumerate(data['pred_output_list']):
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
                PREPARED_DIR, 
                PRED_TEMPLATE.format(
                    img_name=img_name, 
                    subject_idx=subject_idx)
            )
            with open(pred_fpath, 'wb') as pkl_f:
                pkl.dump(pred_dict, pkl_f, protocol=pkl.HIGHEST_PROTOCOL)
