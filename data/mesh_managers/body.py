from typing import List, Dict, Tuple, Union
from abc import abstractmethod
from os.path import join
from glob import glob
from psbody.mesh import Mesh
from pytorch3d.structures import Meshes
from pytorch3d.renderer import Textures
from PIL import Image
import numpy as np
from random import randint
import torch
import os
from tqdm import tqdm

from data.const import (
    BODY_FACES,
    MGN_CLASSES,
    GARMENT_CLASSES,
    MGN_DATASET,
    UV_MAPS_PATH
)
from data.mesh_managers.common import (
     MeshManager,
     random_pallete_color
)
from utils.garment_classes import GarmentClasses
from utils.mesh_utils import concatenate_meshes
from vis.colors import GarmentColors, BodyColors, norm_color

from tailornet_for_garmentor.models.smpl4garment_utils import SMPL4GarmentOutput


class BodyMeshManager(MeshManager):

    def __init__(
            self,
            device: str = 'cpu'
        ) -> None:
        super().__init__(self)
        self.device = device

    def _create_mesh_numpy(
            verts: np.ndarray,
            device: str = 'cpu'
        ) -> Meshes:
        body_colors = np.ones_like(body_verts) * \
            _random_pallete_color(BodyColors)
        
        body_verts = torch.from_numpy(
            body_verts).float().unsqueeze(0).to(device)
        body_faces = torch.from_numpy(
            BODY_FACES.astype(np.int32)).unsqueeze(0).to(device)
        body_colors = torch.from_numpy(
            body_colors).float().unsqueeze(0).to(device)
        
        return Meshes(
            verts=body_verts,
            faces=body_faces,
            textures=Textures(verts_rgb=body_colors)
        )

    def _create_mesh_torch(
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

    def create_meshes(
        self,
        body_verts: Union[np.ndarray, torch.Tensor]
    ) -> Meshes:
        ''' Extract trimesh Meshes for the body mesh.'''
        if type(body_verts) == np.ndarray:
            self._prepare_body_mesh_numpy(body_verts)
        else:
            self._prepare_body_mesh_torch(body_verts)

    def save_meshes(
            self,
            meshes: List[Mesh]
    ) -> None: ...
