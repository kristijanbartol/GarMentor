from typing import List, Dict
import torch
import numpy as np
from pytorch3d.structures import Meshes
from pytorch3d.renderer import Textures

from data.mesh_managers.common import (
    MeshManager,
    default_upper_color,
    default_lower_color,
    default_body_color,
    random_pallete_color
)
from utils.mesh_utils import concatenate_meshes
from vis.colors import GarmentColors, BodyColors

from tailornet_for_garmentor.models.smpl4garment_utils import SMPL4GarmentOutput



class ColoredGarmentsMeshManager(MeshManager):

    ''' The mesh manager for colored garment meshes.
    
        This class is used for rendering clothed meshes in a numpy environment,
        such as when generating the training data.
    '''

    def __init__(
            self,
            config):
        super().__init__()
        self.config = config

    def create_meshes(
            self,
            smpl_output_dict: Dict[str, SMPL4GarmentOutput],
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
            smpl_output_dict['upper'].body_verts,
            smpl_output_dict['upper'].garment_verts,
            smpl_output_dict['lower'].garment_verts
        ]
        faces_list = [
            smpl_output_dict['upper'].body_faces,
            smpl_output_dict['upper'].garment_faces,
            smpl_output_dict['lower'].garment_faces
        ]
        
        if self.config.DEFAULT_MESH_COLORS:
            body_colors = np.ones_like(verts_list[0]) * \
                default_body_color(BodyColors)
        else:
            body_colors = np.ones_like(verts_list[0]) * \
                random_pallete_color(BodyColors)
        
        if self.config.DEFAULT_MESH_COLORS:
            part_colors_list = [
                np.ones_like(verts_list[1]) * default_upper_color(GarmentColors),
                np.ones_like(verts_list[2]) * default_lower_color(GarmentColors),
            ]
        
        concat_verts_list = [verts_list[0]]
        concat_faces_list = [faces_list[0]]
        concat_color_list = [body_colors]
        for idx in range(len(verts_list)-1):
            concat_verts, concat_faces = concatenate_meshes(
                vertices_list=[concat_verts_list[idx], verts_list[idx+1]],
                faces_list=[concat_faces_list[idx], faces_list[idx+1]]
            )
            concat_verts_list.append(concat_verts)
            concat_faces_list.append(concat_faces)
            
            if self.config.DEFAULT_MESH_COLORS:
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
