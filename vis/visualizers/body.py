from typing import Union, Tuple, Optional
import torch
import numpy as np

from models.smpl_official import easy_create_smpl_model
from rendering.body_renderer import BodyRenderer
from vis.visualizers.common import Visualizer2D


class BodyVisualizer(Visualizer2D):

    def __init__(
            self, 
            device: str,
            gender: Optional[str] = None,
            backgrounds_dir_path: str = None
        ) -> None:
        super().__init__(backgrounds_dir_path)

        self.device = device
        self.renderer = BodyRenderer(
            device=self.device,
            batch_size=1
        )
        if gender is not None:
            self.smpl_model = easy_create_smpl_model(
                gender=gender,
                device=device
            )

    def vis(
        self,
        verts: Union[np.ndarray, torch.Tensor],
        cam_t: Optional[np.ndarray] = None
    ) -> Union[Tuple[np.ndarray, np.ndarray],
               Tuple[torch.Tensor, torch.Tensor]]:
        '''Render body using simple rendering strategy + get mask.'''
        body_rgb, body_mask = self.renderer(
            verts=verts,
            cam_t=cam_t
        )
        return body_rgb, body_mask

    def vis_from_params(
            self,
            pose: Union[np.ndarray, torch.Tensor],
            shape: Union[np.ndarray, torch.Tensor],
            cam_t: Optional[Union[np.ndarray, torch.Tensor]] = None,
            gender: Optional[str] = None
    ) -> Union[Tuple[np.ndarray, np.ndarray],
               Tuple[torch.Tensor, torch.Tensor]]:
        '''First run the SMPL body model to get verts and then render.'''
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

        body_vertices: np.ndarray = self.create_body(
            pose=pose,
            shape=shape,
            smpl_model=smpl_model
        ).vertices

        return self.vis(
            verts=body_vertices,
            cam_t=cam_t
        )
