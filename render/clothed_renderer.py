from typing import Dict, Tuple, List, Optional

import torch
import numpy as np
from random import randint

from pytorch3d.structures import Meshes
from pytorch3d.renderer import Textures

from renderer import Renderer
from utils.mesh_utils import concatenate_meshes
from utils.garment_classes import GarmentClasses
from vis.colors import GarmentColors, BodyColors, norm_color

from tailornet_for_garmentor.models.smpl4garment_utils import SMPL4GarmentOutput


class ClothedRenderer(Renderer):

    ''' Clothed meshes renderer.
    
        Note that the class is used to render ClothSURREAL examples.
        Also note that the implementation is Numpy-based, because
        it will not happen that the tensors will arrive as params.
        This is because the current parametric model, TailorNet,
        is used only offline to generate input data and not during
        training.
    '''

    def __init__(
            self,
            *args,
            **kwargs
        ) -> None:
        ''' The clothed renderer constructor.'''
        super().__init__(*args, **kwargs)
        
    def _random_pallete_color(self, pallete):
        return np.array(norm_color(list(pallete)[randint(0, len(pallete) - 1)].value))
        
    def _prepare_meshes(
            self, 
            smpl_output_dict: Dict[str, SMPL4GarmentOutput]
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
        
        body_colors = np.ones_like(verts_list[0]) * \
            self._random_pallete_color(BodyColors)
        
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
            
            part_colors = np.ones_like(verts_list[idx+1]) * \
                self._random_pallete_color(GarmentColors)
            concat_color_list.append(
                np.concatenate([concat_color_list[idx], part_colors], axis=0))
        
        meshes = []
        for idx in range(len(verts_list)):
            concat_verts_list[idx] = torch.from_numpy(
                concat_verts_list[idx]).float().unsqueeze(0).to(self.device)
            concat_faces_list[idx] = torch.from_numpy(
                concat_faces_list[idx].astype(np.int32)).unsqueeze(0).to(self.device)
            concat_color_list[idx] = torch.from_numpy(
                concat_color_list[idx]).float().unsqueeze(0).to(self.device)
            
            meshes.append(Meshes(
                verts=concat_verts_list[idx],
                faces=concat_faces_list[idx],
                textures=Textures(verts_rgb=concat_color_list[idx])
            ))
        
        return meshes
    
    def _extract_seg_maps(
            self, 
            rgbs: List[np.ndarray]
        ) -> np.ndarray:
        ''' Extract segmentation maps from the RGB renders of meshes.

            Note there is a specific algorithm in this procedure. First, take
            the whole clothed body image and use it to create the first map.
            Then, use RGB image with one less piece of garment to get the 
            difference between this image and the previous one. The difference
            is exactly the segmentation map of this piece of garment. Finally,
            apply this procedure for the second piece of clothing.
        '''
        maps = []
        rgb = np.zeros_like(rgbs[-1])
        for rgb_idx in range(len(rgbs) - 1, -1, -1):
            seg_map = ~np.all(np.isclose(rgb, rgbs[rgb_idx], atol=1e-3), axis=-1)
            maps.append(seg_map)
            rgb = rgbs[rgb_idx]
        return np.stack(maps, axis=0)
    
    def _organize_seg_maps(
            self, 
            seg_maps: np.ndarray, 
            garment_classes: GarmentClasses
        ) -> np.ndarray:
        ''' Organize segmentation maps in the form network will expect them.

            In particular, there will always be five maps: the first two for
            the lower garment (depending on the lower label), the second two
            for the upper garment (depending on the upper label), and the
            final for the whole clothed body.
        '''
        feature_maps = np.zeros((5, seg_maps.shape[1], seg_maps.shape[2]))
        feature_maps[-1] = seg_maps[0]
        feature_maps[garment_classes.lower_label] = seg_maps[1]
        feature_maps[garment_classes.upper_label] = seg_maps[2]
        return feature_maps

    def forward(
            self, 
            smpl_output_dict: Dict[str, SMPL4GarmentOutput],
            garment_classes: GarmentClasses,
            cam_t: Optional[np.narray] = None,
            orthographic_scale: float = None,
            lights_rgb_settings: Dict[str, Tuple[float]] = None
        ) -> Tuple(np.ndarray, np.ndarray):
        '''Render RGB images of clothed meshes, single-colored piece-wise.'''
        self._process_optional_arguments(
            cam_t,
            orthographic_scale,
            lights_rgb_settings
        )
        meshes = self._prepare_meshes(smpl_output_dict)
        rgbs = []
        for mesh in meshes:
            fragments = self.rasterizer(
                mesh, 
                cameras=self.cameras
            )
            rgb_image = self.rgb_shader(
                fragments, 
                mesh, 
                lights=self.lights_rgb_render
            )[:, :, :, :3]
            rgbs.append(rgb_image[0].cpu().numpy())
            
        seg_maps = self._extract_seg_maps(rgbs)
        feature_maps = self._organize_seg_maps(seg_maps, garment_classes)

        return rgbs[-1], feature_maps
