from typing import Tuple, Union

import torch
import numpy as np

from data.mesh_managers.body import BodyMeshManager
from rendering.common import Renderer


class BodyRenderer(Renderer):

    ''' A body renderer class.
    
        Note that this renderer supports both Numpy and Torch. The Torch
        tensors come in case the rendering is done in the training loop,
        while the Numpy arrays can come in any other case.
    '''

    def __init__(
            self,
            device: str,
            *args,
            **kwargs
        ) -> None:
        ''' The body renderer constructor.'''
        super().__init__(device=device, *args, **kwargs)
        self.mesh_manager = BodyMeshManager(device)
    
    def _extract_seg_map(
            self, 
            body_rgb: Union[np.ndarray, torch.Tensor]
        ) -> Union[np.ndarray, torch.Tensor]:
        '''Extract segmentation map from the body RGB.'''
        if type(body_rgb) == np.ndarray:
            empty_rgb = np.zeros_like(body_rgb)
            return ~np.all(np.isclose(empty_rgb, body_rgb, atol=1e-3), axis=-1)
        else:
            empty_rgb = torch.zeros_like(body_rgb)
            return ~torch.all(torch.isclose(empty_rgb, body_rgb, atol=1e-3), dim=-1)

    def forward(
            self, 
            verts: torch.Tensor,
            *args,
            **kwargs
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        '''Render RGB images of clothed meshes, single-colored piece-wise.'''
        self._process_optional_arguments(*args, **kwargs)
        
        body_mesh = self.mesh_manager.create_mesh_torch(verts)
        fragments = self.rasterizer(
            body_mesh, 
            cameras=self.cameras
        )
        rgb_img = self.rgb_shader(
            fragments, 
            body_mesh, 
            lights=self.lights_rgb_render
        )[:, :, :, :3]
            
        seg_map = self._extract_seg_map(rgb_img)
        return rgb_img, seg_map
