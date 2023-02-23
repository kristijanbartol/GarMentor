from typing import Dict, Tuple, Optional, Union

import torch
import numpy as np

from pytorch3d.structures import Meshes
from pytorch3d.renderer import Textures

from renderer import Renderer
from vis.colors import BodyColors


class BodyRenderer(Renderer):

    ''' A body renderer class.
    
        Note that this renderer supports both Numpy and Torch. The Torch
        tensors come in case the rendering is done in the training loop,
        while the Numpy arrays can come in any other case.
    '''

    def __init__(
            self,
            *args,
            **kwargs
        ) -> None:
        ''' The body renderer constructor.'''
        super().__init__(*args, **kwargs)
        
    def _prepare_body_mesh_numpy(
            self, 
            body_verts: np.ndarray
        ) -> Meshes:
        '''Numpy version of body mesh preparation.'''
        body_colors = np.ones_like(body_verts) * \
            self._random_pallete_color(BodyColors)
        
        body_verts = torch.from_numpy(
            body_verts).float().unsqueeze(0).to(self.device)
        body_faces = torch.from_numpy(
            self.body_faces.astype(np.int32)).unsqueeze(0).to(self.device)
        body_colors = torch.from_numpy(
            body_colors).float().unsqueeze(0).to(self.device)
        
        return Meshes(
            verts=body_verts,
            faces=body_faces,
            textures=Textures(verts_rgb=body_colors)
        )

    def _prepare_body_mesh_torch(
            self, 
            body_verts: torch.Tensor
        ) -> Meshes:
        '''Torch version of body mesh preparation.'''
        body_colors = (torch.ones_like(body_verts) * \
            torch.tensor(BodyColors.WHITE_SKIN)).float().to(self.device)
        
        return Meshes(
            verts=body_verts,
            faces=self.body_faces_torch,
            textures=Textures(verts_rgb=body_colors)
        )

    def _prepare_body_mesh(
        self,
        body_verts: Union[np.ndarray, torch.Tensor]
    ) -> Meshes:
        '''Extract trimesh Meshes for the body mesh.'''
        if type(body_verts) == np.ndarray:
            self._prepare_body_mesh_numpy(body_verts)
        else:
            self._prepare_body_mesh_torch(body_verts)
    
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
            body_verts: np.ndarray,
            *args,
            **kwargs
        ) -> Union[Tuple[np.ndarray, np.ndarray],
                   Tuple[torch.Tensor, torch.Tensor]]:
        '''Render RGB images of clothed meshes, single-colored piece-wise.'''
        self._process_optional_arguments(*args, **kwargs)
        
        body_mesh = self._prepare_body_mesh(body_verts)
        fragments = self.rasterizer(
            body_mesh, 
            cameras=self.cameras
        )
        rgb_img = self.rgb_shader(
            fragments, 
            body_mesh, 
            lights=self.lights_rgb_render
        )[:, :, :, :3]

        if type(body_verts) == np.ndarray:
            rgb_img = rgb_img[0].cpu().numpy()
            
        seg_map = self._extract_seg_map(rgb_img)
        return rgb_img, seg_map
