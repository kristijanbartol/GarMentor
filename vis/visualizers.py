from typing import Union, Tuple
import torch
import numpy as np

from colors import KPT_COLORS
from data.const import MEAN_CAM_T
from data.off_the_fly_train_datasets import SurrealTrainDataset
from models.parametric_model import ParametricModel
from models.smpl_official import SMPL
from render.clothed_renderer import ClothedRenderer
from render.body_renderer import BodyRenderer
from utils.garment_classes import GarmentClasses
from utils.image_utils import batch_add_rgb_background
from utils.convert_arrays import (
    to_arrays,
    verify_arrays
)


class Visualizer(object):

    default_glob_orient = torch.Tensor([0., 0., 0.])

    def __init__(
            self,
            backgrounds_dir_path: str = None
    ) -> None:
        self.backgrounds_dir_path = backgrounds_dir_path

        self.background_paths = None
        if backgrounds_dir_path is not None:
            self.backgrounds_paths = self._get_background_paths(
                backgrounds_dir_path=backgrounds_dir_path,
                num_backgrounds=1000
            )

    def add_background(
            self, 
            rgb_img: Union[np.ndarray, torch.Tensor],
            mask: Union[np.ndarray, torch.Tensor]
        ) -> Union[np.ndarray, torch.Tensor]:
        if self.background_paths is not None:
            background = SurrealTrainDataset.load_background(
                backgrounds_paths=self.backgrounds_paths,
                img_wh=self.img_wh
            ).to(self.device)

            are_numpy, (rgb_img, mask) = verify_arrays(
                arrays=[rgb_img, mask])

            rgb_img = batch_add_rgb_background(
                backgrounds=background,
                rgb=rgb_img,
                seg=mask
            )

            if are_numpy:
                rgb_img = to_arrays([rgb_img])
        else:
            print('WARNING: The background paths not provided.'\
                ' Returning the original image.')
        return rgb_img


class KeypointsVisualizer(Visualizer):

    def __init__(
            self, 
            device: str, 
            img_wh: int = 256,
            backgrounds_dir_path: str = None
        ) -> None:
        super().__init__(backgrounds_dir_path)

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

    def vis_pose_torch(
            self,
            kpts: torch.Tensor,
            num_maps: int
    ) -> torch.Tensor:
        colored_pose_imgs = torch.zeros(num_maps, 3, 256, 256).to(self.device)
        # TODO: Need to define start and end joints to create stick figures.

    def vis_pose_numpy(
            self,
            kpts: torch.Tensor,
            num_maps: int
    ) -> torch.Tensor:
        colored_pose_imgs = torch.zeros(num_maps, 3, 256, 256).to(self.device)
        # TODO: Need to define start and end joints to create stick figures.

    def vis_pose_from_params_torch(
            self,
            kpts: torch.Tensor,
            num_maps: int
    ) -> torch.Tensor:
        colored_pose_imgs = torch.zeros(num_maps, 3, 256, 256).to(self.device)
        # TODO: Need to define start and end joints to create stick figures.

    def vis_pose_from_params_numpy(
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


class BodyVisualizer(Visualizer):

    def __init__(
            self, 
            device: str,
            backgrounds_dir_path: str = None,
            smpl_model: SMPL = None
        ) -> None:
        super().__init__(backgrounds_dir_path)

        self.device = device
        self.renderer = BodyRenderer(
            device=self.device,
            batch_size=1
        )
        self.smpl_model = smpl_model

    def vis_body_from_params(
            self,
            pose: Union[np.ndarray, torch.Tensor],
            shape: Union[np.ndarray, torch.Tensor],
            glob_orient: Union[np.ndarray, torch.Tensor] = None
    ) -> Tuple[Union[np.ndarray, torch.Tensor],
               Union[np.ndarray, torch.Tensor]]:
        if glob_orient is None:
            glob_orient = self.default_glob_orient

        if self.smpl_body is not None:
            are_numpy, (pose, shape) = verify_arrays(
                arrays=[pose, shape, glob_orient]
            )
            body_vertices = self.smpl_model(
                body_pose=pose,
                global_orient=glob_orient,
                betas=shape,
                pose2rot=False
            ).vertices

            body_rgb, body_mask = self.renderer(body_vertices)
            if are_numpy:
                # TODO: Update this part (that's why I'm updating Renderers anyways.)
                pass
            return body_rgb, body_mask
        else:
            print('WARNING: No SMPL model provided.'\
                ' Returning None.')
            return None, None

    def vis_body_from_verts(
        self,
        verts: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        are_numpy, (verts) = verify_arrays(
            arrays=[verts]
        )


class ClothedVisualizer(Visualizer):

    def __init__(
            self, 
            gender: str,
            upper_class: str,
            lower_class: str,
            device: str,
            backgrounds_dir_path: str = None,
            img_wh=256
        ) -> None:
        super().__init__(backgrounds_dir_path)

        _garment_classes = GarmentClasses(
            upper_class, 
            lower_class
        )
        self.parametric_model = ParametricModel(
            gender=gender, 
            garment_classes=_garment_classes
        )
        self.device = device
        self.renderer = ClothedRenderer(
            device=self.device,
            batch_size=1
        )
        self.img_wh = img_wh

    def vis_clothed(
            self,
            pose: Union[np.ndarray, torch.Tensor], 
            shape: Union[np.ndarray, torch.Tensor], 
            style_vector: Union[np.ndarray, torch.Tensor]
        ) -> Tuple[Union[np.ndarray, torch.Tensor],
                   Union[np.ndarray, torch.Tensor]]:
        are_numpy, (pose, shape, style_vector) = verify_arrays(
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
        
        if are_numpy:
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

            are_numpy, (rgb_img, whole_seg_map) = verify_arrays(
                arrays=[rgb_img, whole_seg_map])

            rgb_img = batch_add_rgb_background(
                backgrounds=background,
                rgb=rgb_img,
                seg=whole_seg_map
            )

            if are_numpy:
                rgb_img = to_arrays([rgb_img])
        else:
            print('WARNING: The background paths not provided.'\
                ' Returning the original image.')
        return rgb_img
