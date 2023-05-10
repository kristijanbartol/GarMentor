from typing import Union, List
import torch
import numpy as np
import os
import cv2
import sys

sys.path.append('/garmentor/')

from frankmocap.bodymocap.body_bbox_detector import BodyPoseEstimator
from frankmocap.bodymocap.body_mocap_api import BodyMocap
import frankmocap.mocap_utils.demo_utils as demo_utils

from configs import paths
from vis.visualizers.clothed import ClothedVisualizer
from utils.garment_classes import GarmentClasses


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


if __name__ == '__main__':
    device = torch.device('cuda')

    # FrankMocap initialization.
    body_bbox_detector = BodyPoseEstimator()
    body_mocap = BodyMocap(
        regressor_checkpoint=MOCAP_CHECKPOINT_PATH, 
        smpl_dir=paths.SMPL, 
        device=device, 
        use_smplx=False
    )

    UPPER = 't-shirt'
    LOWER = 'pant'

    garment_classes = GarmentClasses(
        UPPER, 
        LOWER
    )
    clothed_visualizer = ClothedVisualizer(
            device='cuda',
            gender='male',
            garment_classes=garment_classes
        )

    for img_name in os.listdir('demo/'):
        img_path = os.path.join('demo/', img_name)
        img = cv2.imread(img_path)
        _, body_bbox_list = body_bbox_detector.detect_body_pose(
            img=img
        )
        body_bbox_list = sort_by_bbox_size(
            body_bbox_list=body_bbox_list,
            single_person=True
        )
        pred_output_list = body_mocap.regress(
            img_original=img, 
            body_bbox_list=body_bbox_list
        )
        #pose = pred_output_list[0]['pred_body_pose'][:, 3:]
        #global_orient = pred_output_list[0]['pred_body_pose'][:, :3]
        pose = pred_output_list[0]['pred_body_pose'][0]
        shape = pred_output_list[0]['pred_betas'][0]
        style_vector = np.zeros((4, 4))

        pose[0] = -pose[0]

        rgb_img, seg_maps, joints_3d = clothed_visualizer.vis_from_params(
            pose=pose,
            shape=shape,
            style_vector=style_vector,
            cam_t=None,     # TODO: Might remove cam_t as a parameter here.
        )
        rgb_img = rgb_img.cpu().numpy()

        #rgb_img = clothed_visualizer.add_background(
        #    rgb_img=np.swapaxes(rgb_img, 0, 2), 
        #    mask=seg_maps[0],
        #    back_img=np.ones((3, 256, 256), dtype=np.float32) / 2
        #)
        if not os.path.exists(f'/garmentor/output/demo-{UPPER}-{LOWER}/'):
            os.makedirs(f'/garmentor/output/demo-{UPPER}-{LOWER}/')

        clothed_visualizer.save_vis(
            rgb_img=rgb_img, 
            save_path=os.path.join(f'/garmentor/output/demo-{UPPER}-{LOWER}/{img_name}')
        )

        upper_rgb, lower_rgb = clothed_visualizer.vis_garments_only(
            pose=pose,
            shape=shape,
            style_vector=style_vector,
            
        )
        upper_rgb = upper_rgb.cpu().numpy()
        lower_rgb = lower_rgb.cpu().numpy()

        clothed_visualizer.save_vis(
            rgb_img=upper_rgb, 
            save_path=os.path.join(f'/garmentor/output/demo-{UPPER}-{LOWER}/upper_{img_name}')
        )
        clothed_visualizer.save_vis(
            rgb_img=lower_rgb, 
            save_path=os.path.join(f'/garmentor/output/demo-{UPPER}-{LOWER}/lower_{img_name}')
        )
