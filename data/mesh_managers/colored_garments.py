from typing import Dict, List, Union
import torch
import numpy as np
from pytorch3d.structures import Meshes
from pytorch3d.renderer import Textures

from configs.poseMF_shapeGaussian_net_config import get_cfg_defaults
from data.mesh_managers.common import (
    MeshManager,
    default_upper_color,
    default_lower_color,
    default_body_color,
    torch_default_body_color,
    torch_default_garment_color,
    random_pallete_color
)
from utils.drapenet_structure import DrapeNetStructure
from utils.mesh_utils import concatenate_mesh_list
from vis.colors import GarmentColors, BodyColors

from tailornet_for_garmentor.models.smpl4garment_utils import SMPL4GarmentOutput



class ColoredGarmentsMeshManager(MeshManager):

    ''' The mesh manager for colored garment meshes.
    
        This class is used for rendering clothed meshes in a numpy environment,
        such as when generating the training data.
    '''

    def __init__(self):
        super().__init__()
        self.config = get_cfg_defaults()

    def create_meshes(
            self,
            garment_output_dict: Dict[str, Union[SMPL4GarmentOutput, DrapeNetStructure]],
            device: str = 'cpu'
    ) -> List[Meshes]:
        ''' Extract trimesh Meshes from SMPL4Garment output (verts and faces).
        
            To construct the Meshes for upper, lower, and both piece of
            clothing on top of the body mesh, the vertices and the faces need
            to be concatenated into corresponding arrays. In particular, the
            body mesh only consists of body vertices and body faces, i.e.,
            is not concatenated with other arrays. The lower+body garment 
            mesh consists of concatenated body and lower mesh vertices and 
            faces. Finally, the complete mesh (body+upper+lower) consists of
            concatenanted body, lower, and upper vertices and faces. The
            three meshes are returnned as a result of this method.
        '''
        verts_list = [
            garment_output_dict['upper'].body_verts,
            garment_output_dict['upper'].garment_verts,
            garment_output_dict['lower'].garment_verts
        ]
        faces_list = [
            garment_output_dict['upper'].body_faces,
            garment_output_dict['upper'].garment_faces,
            garment_output_dict['lower'].garment_faces
        ]
        
        if self.config.VISUALIZATION.DEFAULT_MESH_COLORS:
            body_colors = np.ones_like(verts_list[0]) * \
                default_body_color(BodyColors)
        else:
            body_colors = np.ones_like(verts_list[0]) * \
                random_pallete_color(BodyColors)
        
        if self.config.VISUALIZATION.DEFAULT_MESH_COLORS:
            part_colors_list = [
                np.ones_like(verts_list[1]) * default_upper_color(GarmentColors),
                np.ones_like(verts_list[2]) * default_lower_color(GarmentColors),
            ]
        
        concat_verts_list = [verts_list[0]]
        concat_faces_list = [faces_list[0]]
        concat_color_list = [body_colors]
        for idx in range(len(verts_list)-1):
            concat_verts, concat_faces = concatenate_mesh_list(
                vertices_list=[concat_verts_list[idx], verts_list[idx+1]],
                faces_list=[concat_faces_list[idx], faces_list[idx+1]]
            )
            concat_verts_list.append(concat_verts)
            concat_faces_list.append(concat_faces)
            
            if self.config.VISUALIZATION.DEFAULT_MESH_COLORS:
                part_colors = part_colors_list[idx]
            else:
                part_colors = np.ones_like(verts_list[idx+1]) * \
                    random_pallete_color(GarmentColors)
            
            concat_color_list.append(
                np.concatenate([concat_color_list[idx], part_colors], axis=0))
        
        meshes = []
        for idx in range(len(verts_list)):
            concat_verts_list[idx] = torch.from_numpy(
                concat_verts_list[idx]).float().unsqueeze(0).to(device)
            concat_faces_list[idx] = torch.from_numpy(
                concat_faces_list[idx].astype(np.int32)).unsqueeze(0).to(device)
            concat_color_list[idx] = torch.from_numpy(
                concat_color_list[idx]).float().unsqueeze(0).to(device)
            
            meshes.append(Meshes(
                verts=concat_verts_list[idx],
                faces=concat_faces_list[idx],
                textures=Textures(verts_rgb=concat_color_list[idx])
            ))
        
        return meshes 


def create_meshes_torch(
        verts_list: List[torch.Tensor],     # already on device, 2[1, V, 3]
        faces_list: List[torch.Tensor]      # already on device, 2[1, F, 3]
) -> List[Meshes]:
    textures_list = [
        torch.ones_like(verts_list[0]) * torch_default_body_color(BodyColors, device=verts_list[0].device),
        torch.ones_like(verts_list[1]) * torch_default_garment_color(GarmentColors, device=verts_list[0].device)
    ]

    cat_verts_list = [
        verts_list[0],
        torch.cat([
            verts_list[0],
            verts_list[1]
        ], dim=1)
    ]
    cat_faces_list = [
        faces_list[0],
        torch.cat([
            faces_list[0],
            faces_list[1] + verts_list[0].shape[1],
        ], dim=1)
    ]
    cat_textures_list = [
        textures_list[0],
        torch.cat([
            textures_list[0], 
            textures_list[1]
        ], dim=1)
    ]

    meshes = [
        Meshes(verts=vs, faces=fs, textures=Textures(verts_rgb=ts)) for (vs, fs, ts) in zip(
            cat_verts_list, cat_faces_list, cat_textures_list
        )
    ]
    return meshes
