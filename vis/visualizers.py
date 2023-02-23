from typing import Union, Tuple
import torch
import numpy as np

from colors import KPT_COLORS
from data.off_the_fly_train_datasets import SurrealTrainDataset
from models.parametric_model import ParametricModel
from models.smpl_official import SMPL
from render.clothed_renderer import ClothedRenderer
from render.body_renderer import BodyRenderer
from utils.garment_classes import GarmentClasses
from utils.image_utils import add_rgb_background
from utils.convert_arrays import to_tensors


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
        '''Add random 2D background "behind" rendered person based on mask.'''
        if self.background_paths is not None:
            background = SurrealTrainDataset.load_background(
                backgrounds_paths=self.backgrounds_paths,
                img_wh=self.img_wh
            ).to(self.device)

            rgb_img = add_rgb_background(
                backgrounds=background,
                rgb=rgb_img,
                seg=mask
            )
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
        '''First run the SMPL body model to get verts and then render.'''
        if glob_orient is None:
            glob_orient = self.default_glob_orient

        if self.smpl_body is not None:
            pose, shape = to_tensors(
                arrays=[pose, shape, glob_orient]
            )
            body_vertices: np.ndarray = self.smpl_model(
                body_pose=pose,
                global_orient=glob_orient,
                betas=shape,
                pose2rot=False
            ).vertices

            body_rgb, body_mask = self.renderer(body_vertices)
            return body_rgb, body_mask
        else:
            print('WARNING: No SMPL model provided.'\
                ' Returning (None, None).')
            return None, None

    def vis_body_from_verts(
        self,
        verts: Union[np.ndarray, torch.Tensor]
    ) -> Union[Tuple[np.ndarray, np.ndarray],
               Tuple[torch.Tensor, torch.Tensor]]:
        '''Render body using simple rendering strategy + get mask.'''
        body_rgb, body_mask = self.renderer(verts)
        return body_rgb, body_mask


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
            pose: np.ndarray, 
            shape: np.ndarray, 
            style_vector: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray]:
        ''' Visualize clothed mesh(es).
        
            First, the parametric model is ran to obtain the verts.
            Then, the renderer renders the clothed meshes. The method
            returns an RGB rendered image (without background) and the
            mask of the person's silhouette. Note that the method 
            always expects Numpy arrays because visualizing TailorNet 
            will not be required in training loop, for now.
        '''
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
            garment_classes=self.parametric_model.garment_classes
        )
        return rgb_img, seg_maps
