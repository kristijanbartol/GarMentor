from typing import Union
import torch
import numpy as np

from colors import KPT_COLORS
from configs import paths
from data.const import MEAN_CAM_T
from data.off_the_fly_train_datasets import SurrealTrainDataset
from models.parametric_model import ParametricModel
from renderers.surreal_renderer import SurrealRenderer
from utils.garment_classes import GarmentClasses
from utils.image_utils import batch_add_rgb_background
from utils.convert_arrays import (
    to_tensors,
    to_arrays
)


class KeypointsVisualizer:

    def __init__(
            self, 
            device: str, 
            img_wh: int = 256
        ) -> None:
        self.device = device
        self.img_wh = img_wh

    def batched_vis_heatmaps(
            self,
            heatmaps: torch.Tensor, 
            num_heatmaps: int
        ) -> torch.Tensor:
        colored_heatmaps = torch.zeros(num_heatmaps, 3, 256, 256).to(self.device)
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
    ) -> Union[np.ndarray, torch.Tensor]:
        colored_heatmap = torch.zeros(3, 256, 256).to(self.device)
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
        colored_heatmap = np.zeros(3, 256, 256)
        for color_idx, color_key in enumerate(KPT_COLORS):
            heatmap = np.stack((heatmap[color_idx],) * 3, dim=1)
            color_tensor = np.array(KPT_COLORS[color_key])
            heatmap[0] *= color_tensor[0]
            heatmap[1] *= color_tensor[1]
            heatmap[2] *= color_tensor[2]
            colored_heatmaps += heatmap
        return colored_heatmap

    def batched_vis_pose(
            self,
            kpts: torch.Tensor,
            num_maps: int
    ) -> torch.Tensor:
        colored_pose_imgs = torch.zeros(num_maps, 3, 256, 256).to(self.device)
        # TODO: Need to define start and end joints to create stick figures.

    def overlay_keypoints(
            self,
            kpts: Union[np.ndarray, torch.Tensor],
            img: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        # TODO: Overlay keypoints on top of the image.
        pass


class BodyVisualizer:

    # TODO: Implement this.

    def __init__(
            self, 
            device: str,
            background_dir_path: str = None
        ) -> None:
        pass


class ClothedVisualizer:

    def __init__(
            self, 
            gender: str,
            upper_class: str,
            lower_class: str,
            device: str,
            backgrounds_dir_path: str = None,
            img_wh=256
        ) -> None:
        _garment_classes = GarmentClasses(
            upper_class, 
            lower_class
        )
        self.parametric_model = ParametricModel(
            gender=gender, 
            garment_classes=_garment_classes
        )
        self.device = device
        self.renderer = SurrealRenderer(
            device=self.device,
            batch_size=1
        )
        self.background_paths = None
        if backgrounds_dir_path is not None:
            self.backgrounds_paths = self._get_background_paths(
                backgrounds_dir_path=backgrounds_dir_path,
                num_backgrounds=1000
            )
        self.img_wh = img_wh

    def vis_body(
            self,
            pose: Union[np.ndarray, torch.Tensor], 
            shape: Union[np.ndarray, torch.Tensor], 
            style_vector: Union[np.ndarray, torch.Tensor]
        ) -> Union[np.ndarray, torch.Tensor]:
        np_arrays = False
        if type(pose) == np.ndarray:
            assert (type(pose) == type(shape) == type(style_vector))
            np_arrays = True
            pose, shape, style_vector = to_tensors(
                arrays=[pose, shape, style_vector]
            )

        smpl_output_dict = self.parametric_model.run(
            pose=pose,
            shape=shape,
            style_vector=style_vector
        )
        rgb_img, seg_maps = self.renderer(
            smpl_output_dict,
            garment_classes=self.parametric_model.garment_classes,
            cam_t=MEAN_CAM_T
        )
        
        if np_arrays:
            rgb_img, seg_maps = to_arrays([rgb_img, seg_maps])
        return rgb_img, seg_maps
    
    def add_background(
            self, 
            rgb_img: Union[np.ndarray, torch.Tensor],
            whole_seg_map: Union[np.ndarray, torch.Tensor]
        ) -> Union[np.ndarray, torch.Tensor]:
        if self.background_paths is not None:
            background = SurrealTrainDataset.load_background(
                backgrounds_paths=self.backgrounds_paths,
                img_wh=self.img_wh
            ).to(self.device)
            
            np_arrays = False
            if type(rgb_img) == np.ndarray:
                assert(type(rgb_img) == type(whole_seg_map))
                rgb_img, whole_seg_map = to_tensors(
                    [rgb_img, whole_seg_map]
                )

            rgb_img = batch_add_rgb_background(
                backgrounds=background,
                rgb=rgb_img,
                seg=whole_seg_map
            )

            if np_arrays:
                rgb_img = to_arrays([rgb_img])
        else:
            print('WARNING: The background paths not provided.'\
                ' Returning the original image.')
        return rgb_img
