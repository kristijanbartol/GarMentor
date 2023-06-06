from typing import Optional, Tuple
import torch
import numpy as np
import cv2
from PIL import Image

from models.smpl_official import (
    easy_create_smpl_model,
    SMPL
)
from utils.cam_utils import project_points
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

    default_projection_type = 'orthographic'

    def __init__(
            self, 
            device: str = 'cpu', 
            img_wh: int = 256,
            gender: Optional[str] = None,
            smpl_model: Optional[SMPL] = None,
            backgrounds_dir_path: Optional[str] = None,
            projection_type: Optional[str] = None
        ) -> None:
        """
        Initialize KeypointsVisualizer class.

        If smpl_model is provided, then gender argument is ignored.
        If smpl_model is not provided, then gender argument, if
        provided, is used to create an SMPL model. Finally, if both
        smpl_model and gender are not provided, the SMPL model is None
        and is expected that will be created in the visualization method.
        """
        super().__init__(backgrounds_dir_path)

        self.device = device
        self.img_wh = img_wh
        self.projection_type = self.default_projection_type
        if projection_type is not None:
            if projection_type in ['orthographic', 'perspective']:
                self.projection_type = projection_type
            else:
                print('WARNING: Projection type invalid. Keeping '\
                      f'{self.default_projection_type}.')
        print(f'[KeypointVisualizer] Projection: {self.projection_type}')

        self.smpl_model = None
        model_description = 'None'
        if smpl_model is not None:
            self.smpl_model = smpl_model
            model_description = f'predefined_{smpl_model.gender}'
        else:
            if gender is not None:
                self.smpl_model = easy_create_smpl_model(
                    gender=gender,
                    device=device
                )
                model_description = f'new_{gender}'
        print(f'[KeypointVisualizer] Using {model_description} SMPL.')

    def batched_vis_heatmaps(
            self,
            heatmaps: torch.Tensor, 
            num_heatmaps: int
        ) -> torch.Tensor:
        """
        Visualize a number of colored heatmaps, given the batch of keypoints.
        """
        colored_heatmaps = torch.zeros(
            num_heatmaps, 3, self.img_wh, self.img_wh).to(self.device)
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
            heatmap: torch.Tensor,
            height: Optional[int] = None,
            width: Optional[int] = None
    ) -> torch.Tensor:
        """
        Visualize a colored heatmap based on given heatmap in Torch.
        """
        if height is not None and width is not None:
            colored_heatmap = torch.zeros(3, height, width).to(self.device)
        else:
            colored_heatmap = torch.zeros(3, self.img_wh, self.img_wh).to(self.device)
        for color_idx, color_key in enumerate(KPT_COLORS):
            _heatmap = torch.stack((heatmap[color_idx],) * 3, dim=0)
            color_tensor = torch.tensor(KPT_COLORS[color_key])
            _heatmap[0] *= color_tensor[0]
            _heatmap[1] *= color_tensor[1]
            _heatmap[2] *= color_tensor[2]
            colored_heatmap += _heatmap
        return colored_heatmap

    def vis_heatmap_numpy(
            self,
            heatmap: np.ndarray
    ) -> np.ndarray:
        """
        Visualize a colored heatmap based on given heatmap in Numpy.
        """
        colored_heatmap = np.zeros((3, self.img_wh, self.img_wh))
        for color_idx, color_key in enumerate(KPT_COLORS):
            _heatmap = np.stack((heatmap[color_idx],) * 3, dim=0)
            _heatmap[0] *= KPT_COLORS[color_key][0]
            _heatmap[1] *= KPT_COLORS[color_key][1]
            _heatmap[2] *= KPT_COLORS[color_key][2]
            colored_heatmap += _heatmap
        return colored_heatmap
    
    def vis_keypoints(
            self,
            kpts: np.ndarray,
            back_img: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Visualize a colored image of keypoints, given coordinates.
        """
        if back_img is None:
            back_img = np.zeros((self.img_wh, self.img_wh, 3))
        if back_img.max() < 1:
            back_img = (back_img * 255).astype(np.uint8)

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
        """
        Add line connections between the joints (COCO-specific).
        """
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
        """
        Visualize a colored image of the pose, given coordinates.
        """
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
        """
        Provide 3D keypoints, project, and visualize.
        """
        kpts = project_points(
            points=kpts_3d,
            projection_type=self.projection_type,
            cam_t=cam_t
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
        """
        Conveniently visualize pose from the parameters of the body.
        """
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
        """
        Overlay pose on top of the background image.
        """
        return self.vis(
            kpts=kpts, 
            back_img=back_img,
            skeleton=skeleton
        )

    def save_vis(
            self,
            img: np.ndarray,
            save_path: str
    ) -> None:
        img = Image.fromarray(img.astype(np.uint8))
        img.save(save_path)
        print(f'Saved keypoints image: {save_path}...')
