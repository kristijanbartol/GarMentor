from typing import Optional, Tuple
import torch
import numpy as np
import cv2

from configs.const import (
    DEFAULT_FOCAL_LENGTH,
    IMG_WH
)
from models.smpl_official import easy_create_smpl_model
from utils.cam_utils import perspective_project
from utils.label_conversions import (
    COCO_START_IDXS,
    COCO_END_IDXS,
    COCO_LR,
    ALL_JOINTS_TO_COCO_MAP
)
from vis.colors import (
    KPT_COLORS,
    LCOLOR,
    RCOLOR
)
from vis.visualizers.common import Visualizer2D


class KeypointsVisualizer(Visualizer2D):

    def __init__(
            self, 
            device: str = 'cpu', 
            img_wh: int = 256,
            gender: Optional[str] = None,
            backgrounds_dir_path: Optional[str] = None,
        ) -> None:
        super().__init__(backgrounds_dir_path)

        self.device = device
        self.img_wh = img_wh

        self.smpl_model = None
        if gender is not None:
            self.smpl_model = easy_create_smpl_model(
                gender=gender,
                device=device
            )

    def batched_vis_heatmaps(
            self,
            heatmaps: torch.Tensor, 
            num_heatmaps: int
        ) -> torch.Tensor:
        '''Visualize a number of colored heatmaps, given the batch of keypoints.'''
        colored_heatmaps = torch.zeros(num_heatmaps, 3, self.img_wh, self.img_wh).to(self.device)
        for color_idx, color_key in enumerate(KPT_COLORS):
            heatmaps = torch.stack((heatmaps[:num_heatmaps, color_idx],) * 3, dim=1)
            color_tensor = torch.tensor(KPT_COLORS[color_key])
            heatmaps[:, 0] *= color_tensor[0]
            heatmaps[:, 1] *= color_tensor[1]
            heatmaps[:, 2] *= color_tensor[2]
            colored_heatmaps += heatmaps
        return colored_heatmaps

    def vis_heatmap_torch(
            self,
            heatmap: torch.Tensor
    ) -> torch.Tensor:
        '''Visualize a colored heatmap based on given heatmap in Torch.'''
        colored_heatmap = torch.zeros(3, self.img_wh, self.img_wh).to(self.device)
        for color_idx, color_key in enumerate(KPT_COLORS):
            heatmap = torch.stack((heatmap[color_idx],) * 3, dim=1)
            color_tensor = torch.tensor(KPT_COLORS[color_key])
            heatmap[0] *= color_tensor[0]
            heatmap[1] *= color_tensor[1]
            heatmap[2] *= color_tensor[2]
            colored_heatmaps += heatmap
        return colored_heatmap

    def vis_heatmap_numpy(
            self,
            heatmap: np.ndarray
    ) -> np.ndarray:
        '''Visualize a colored heatmap based on given heatmap in Numpy.'''
        colored_heatmap = np.zeros((3, self.img_wh, self.img_wh))
        for color_idx, color_key in enumerate(KPT_COLORS):
            heatmap = np.stack((heatmap[color_idx],) * 3, dim=1)
            heatmap[0] *= KPT_COLORS[color_key][0]
            heatmap[1] *= KPT_COLORS[color_key][1]
            heatmap[2] *= KPT_COLORS[color_key][2]
            colored_heatmaps += heatmap
        return colored_heatmap
    
    def vis_keypoints(
            self,
            kpts: np.ndarray,
            back_img: Optional[np.ndarray] = None
    ) -> np.ndarray:
        '''Visualize a colored image of keypoints, given coordinates.'''
        if back_img is None:
            back_img = np.zeros((3, self.img_wh, self.img_wh))
        back_img = back_img.transpose((1, 2, 0)).astype(np.uint8).copy()

        for idx, color_key in enumerate(KPT_COLORS):
            kpt: Tuple[int, int] = [int(x) for x in kpts[idx]]
            color: Tuple[int, int, int] = KPT_COLORS[color_key]
            cv2.circle(
                img=back_img, 
                center=kpt, 
                radius=3, 
                color=color, 
                thickness=3
            )
        return back_img

    def _add_skeleton(
            self, 
            kpts: np.ndarray,
            pose_img: np.ndarray
    ) -> np.ndarray:
        '''Add line connections between the joints (COCO-specific).'''
        for line_idx, start_idx in enumerate(COCO_START_IDXS):
            start_kpt: Tuple[int, int] = [int(x) for x in kpts[start_idx]]
            end_kpt: Tuple[int, int] = [int(x) for x in kpts[COCO_END_IDXS[line_idx]]]
            color = LCOLOR if COCO_LR[line_idx] else RCOLOR
            cv2.line(
                img=pose_img, 
                pt1=start_kpt, 
                pt2=end_kpt, 
                color=color, 
                thickness=2
            ) 
        return pose_img

    def vis(self,
            kpts: np.ndarray,
            back_img: Optional[np.ndarray] = None,
            skeleton: Optional[bool] = True
    ) -> np.ndarray:
        '''Visualize a colored image of the pose, given coordinates.'''
        pose_img = self.vis_keypoints(kpts, back_img)
        if skeleton:
            self._add_skeleton(kpts, pose_img)
        return pose_img
    
    def vis_from_3d_keypoints(
            self,
            kpts_3d: np.ndarray,
            cam_t: Optional[np.ndarray] = None,
            back_img: Optional[np.ndarray] = None,
            skeleton: bool = True
    ) -> np.ndarray:
        '''Provide 3D keypoints, project, and visualize.'''
        if cam_t is None:
            cam_t = self.default_cam_t

        kpts = perspective_project(
            points=kpts_3d,
            rotation=None,
            translation=cam_t,
            cam_K=None,
            focal_length=DEFAULT_FOCAL_LENGTH,
            img_wh=IMG_WH
        )
        return self.vis(
            kpts=kpts,
            back_img=back_img,
            skeleton=skeleton
        )

    def vis_from_params(
            self,
            pose: np.ndarray,
            shape: np.ndarray,
            cam_t: Optional[np.ndarray] = None,
            gender: Optional[str] = None,
            back_img: Optional[np.ndarray] = None,
            skeleton: bool = True
    ) -> np.ndarray:
        '''Conveniently visualize pose from the parameters of the body.'''
        if self.smpl_model is not None:
            smpl_model = self.smpl_model
        else:
            if gender is None:
                print('WARNING: Gender unspecified, setting male.')
                gender = 'male'
            smpl_model = easy_create_smpl_model(
                gender=gender,
                device=self.device
            )

        joints_3d = self.create_body(
            pose=pose,
            shape=shape,
            smpl_model=smpl_model
        ).joints.detach().cpu().numpy()[0, ALL_JOINTS_TO_COCO_MAP]

        return self.vis_from_3d_keypoints(
            kpts_3d=joints_3d,
            cam_t=cam_t,
            back_img=back_img,
            skeleton=skeleton
        )

    def overlay_pose(
            self,
            kpts: np.ndarray,
            back_img: np.ndarray,
            skeleton: bool = True
    ) -> np.ndarray:
        '''Overlay pose on top of the background image.'''
        return self.vis(
            kpts=kpts, 
            back_img=back_img,
            skeleton=skeleton
        )
