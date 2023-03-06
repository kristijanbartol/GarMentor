from typing import Union, Tuple, Optional
import torch
import numpy as np

from models.smpl_official import SMPL
from rendering.body_renderer import BodyRenderer
from utils.convert_arrays import to_tensors
from vis.visualizers.common import Visualizer2D


class BodyVisualizer(Visualizer2D):

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
            glob_orient: Union[np.ndarray, torch.Tensor] = None,
            cam_t: Optional[np.ndarray] = None
    ) -> Union[Tuple[np.ndarray, np.ndarray],
               Tuple[torch.Tensor, torch.Tensor]]:
        '''First run the SMPL body model to get verts and then render.'''
        if glob_orient is None:
            glob_orient = self.default_glob_orient

        body_vertices: np.ndarray = self.create_body(
            pose=pose,
            shape=shape,
            glob_orient=glob_orient,
            smpl_model=self.smpl_model
        ).vertices

        return self.vis(
            verts=body_vertices,
            cam_t=cam_t
        )
