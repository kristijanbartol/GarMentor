from typing import Union
from pytorch3d.structures import Meshes
from pytorch3d.renderer import Textures
import numpy as np
import torch

from data.const import BODY_FACES
from data.mesh_managers.common import (
    MeshManager,
    random_pallete_color
)
from vis.colors import BodyColors


class BodyMeshManager(MeshManager):

    def __init__(
            self,
            device: str = 'cpu'
        ) -> None:
        super().__init__()
        self.device = device
        self.faces = torch.from_numpy(
            BODY_FACES.astype(np.int32)).unsqueeze(0).to(device)

    def create_mesh_numpy(
            self,
            verts: np.ndarray,
            device: str = 'cpu'
    ) -> Meshes:
        '''Numpy version of body mesh preparaation.'''
        colors = np.ones_like(verts) * \
            random_pallete_color(BodyColors)
        
        verts = torch.from_numpy(verts).float().unsqueeze(0).to(device)
        
        colors = torch.from_numpy(
            colors).float().unsqueeze(0).to(device)
        
        return Meshes(
            verts=verts,
            faces=self.faces,
            textures=Textures(verts_rgb=colors)
        )

    def create_mesh_torch(
                self, 
                verts: torch.Tensor
    ) -> Meshes:
        '''Torch version of body mesh preparation.'''
        color_pallete = torch.tensor(
            BodyColors.WHITE_SKIN.value, 
            dtype=torch.float32
        )
        colors = (torch.ones_like(verts) * color_pallete).float().to(self.device)
        
        return Meshes(
            verts=verts,
            faces=self.faces,
            textures=Textures(verts_rgb=colors)
        )

    def create_meshes(
        self,
        verts: Union[np.ndarray, torch.Tensor]
    ) -> Meshes:
        ''' Extract trimesh Meshes for the body mesh.'''
        if type(verts) == np.ndarray:
            self.create_mesh_numpy(verts)
        else:
            self.create_mesh_torch(verts)
