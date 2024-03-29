from typing import Dict, Tuple, List
import torch
import numpy as np
from pytorch3d.structures import Meshes
from pytorch3d.renderer import Textures

from data.mesh_managers.colored_garments import ColoredGarmentsMeshManager
from data.mesh_managers.common import (
    default_upper_color,
    default_lower_color
)
from rendering.common import Renderer
from utils.drapenet_structure import DrapeNetStructure
from utils.garment_classes import GarmentClasses
from vis.colors import GarmentColors

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
        self.mesh_manager = ColoredGarmentsMeshManager()

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
        rgb = np.zeros_like(rgbs[-1][0])
        for rgb_idx in range(len(rgbs) - 1, -1, -1):
            seg_map = ~np.all(np.isclose(rgb, rgbs[rgb_idx], atol=1e-3), axis=-1)
            maps.append(seg_map)
            rgb = rgbs[rgb_idx]
        return np.stack(maps, axis=0)


class TNClothedRenderer(ClothedRenderer):

    def __init__(
            self,
            *args,
            **kwargs
        ) -> None:
        ''' The clothed renderer constructor.'''
        super().__init__(*args, **kwargs)
    
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
            device: str,
            *args,
            **kwargs
        ) -> Tuple[torch.Tensor, np.ndarray]:
        '''Render RGB images of clothed meshes, single-colored piece-wise.'''
        self._process_optional_arguments(*args, **kwargs)

        meshes = self.mesh_manager.create_meshes(
            garment_output_dict=smpl_output_dict,
            device=device
        )
        rgbs = []
        for mesh_part, mesh in zip(['body', 'upper', 'lower'], meshes):
            print(f'Rendering {mesh_part} mesh...')
            fragments = self.rasterizer(
                mesh, 
                cameras=self.cameras
            )
            rgb_image = self.rgb_shader(
                fragments, 
                mesh, 
                lights=self.lights_rgb_render
            )[:, :, :, :3]
            rgbs.append(rgb_image)
            
        final_rgb = rgbs[-1][0]     # NOTE: for now, non-batched rendering
        seg_maps = self._extract_seg_maps([x[0].cpu().numpy() for x in rgbs])
        feature_maps = self._organize_seg_maps(seg_maps, garment_classes)

        return final_rgb, feature_maps
    
    @staticmethod
    def _simple_extract_seg_map(rgb):
        return ~np.all(np.isclose(np.zeros_like(rgb), rgb, atol=1e-3), axis=-1)

    def special_rendering(
            self,
            smpl_output_dict: Dict[str, SMPL4GarmentOutput],
            garment_classes: GarmentClasses,
            device: str,
    ):
        upper_verts = torch.from_numpy(smpl_output_dict['upper'].garment_verts).float().unsqueeze(0).to(device)
        upper_faces = torch.from_numpy(smpl_output_dict['upper'].garment_faces.astype(np.int32)).unsqueeze(0).to(device)
        upper_color = torch.tensor(default_upper_color(GarmentColors)).float().to(device)
        upper_mesh_color = (torch.ones_like(upper_verts) * \
            upper_color).float().to(device)
        upper_mesh = Meshes(
            verts=upper_verts,
            faces=upper_faces,
            textures=Textures(verts_rgb=upper_mesh_color)
        )
        lower_verts = torch.from_numpy(smpl_output_dict['lower'].garment_verts).float().unsqueeze(0).to(device)
        lower_faces = torch.from_numpy(smpl_output_dict['lower'].garment_faces.astype(np.int32)).unsqueeze(0).to(device)
        lower_color = torch.tensor(default_lower_color(GarmentColors)).float().to(device)
        lower_mesh_color = (torch.ones_like(lower_verts) * \
            lower_color).float().to(device)
        lower_mesh = Meshes(
            verts=lower_verts,
            faces=lower_faces,
            textures=Textures(verts_rgb=lower_mesh_color)
        )

        rgbs = []
        for mesh in [upper_mesh, lower_mesh]:
            fragments = self.rasterizer(
                mesh, 
                cameras=self.cameras
            )
            rgb_image = self.rgb_shader(
                fragments, 
                mesh, 
                lights=self.lights_rgb_render
            )[:, :, :, :3]
            rgbs.append(rgb_image[0])
            
        #upper_seg_map = self._simple_extract_seg_map(rgbs[0])
        #lower_seg_map = self._simple_extract_seg_map(rgbs[1])

        return rgbs[0], rgbs[1]


class DNClothedRenderer(ClothedRenderer):

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
    
    def _organize_seg_maps(
            self, 
            seg_maps: np.ndarray
        ) -> np.ndarray:
        ''' Organize segmentation maps in the form network will expect them.

            In particular, the maps are created in reverse order (whole body,
            lower cloth, upper cloth), because of the way the maps are extracted
            from RGB images. Therefore, the order of the maps have to be
            inversed.
        '''
        feature_maps = np.zeros((3, seg_maps.shape[1], seg_maps.shape[2]))
        feature_maps[2] = seg_maps[0]
        feature_maps[1] = seg_maps[1]
        feature_maps[0] = seg_maps[2]
        return feature_maps

    def forward(
            self, 
            drapenet_dict: Dict[str, DrapeNetStructure],
            device: str,
            *args,
            **kwargs
        ) -> Tuple[torch.Tensor, np.ndarray]:
        '''Render RGB images of clothed meshes, single-colored piece-wise.'''
        self._process_optional_arguments(*args, **kwargs)

        meshes = self.mesh_manager.create_meshes(
            garment_output_dict=drapenet_dict,
            device=device
        )
        rgbs = []
        # NOTE: Need this particular order because of the way I produce segmaps.
        for mesh_part, mesh in zip(['body', 'upper', 'lower'], meshes):
            print(f'Rendering {mesh_part} mesh...')
            fragments = self.rasterizer(
                mesh, 
                cameras=self.cameras
            )
            try:
                mesh.verts_list()
            except RuntimeError as cuda_err:
                return None, None
            rgb_image = self.rgb_shader(
                fragments, 
                mesh, 
                lights=self.lights_rgb_render
            )[:, :, :, :3]
            rgbs.append(rgb_image)
            
        final_rgb = rgbs[-1][0]     # NOTE: for now, non-batched rendering
        seg_maps = self._extract_seg_maps([x[0].cpu().numpy() for x in rgbs])
        feature_maps = self._organize_seg_maps(seg_maps)

        return final_rgb, feature_maps
    

class TorchClothedRenderer(ClothedRenderer):

    def __init__(
            self,
            *args,
            **kwargs
        ) -> None:
        ''' The clothed renderer constructor.'''
        super().__init__(*args, **kwargs)

    @staticmethod
    def _get_img_diff(
        rgb1,
        rgb2
    ):
        return torch.logical_not(torch.all(torch.isclose(rgb1, rgb2, atol=1e-3), dim=-1))

    def forward(
            self, 
            meshes: List[Meshes],
            *args,
            **kwargs
        ) -> torch.Tensor:
        '''Render RGB images of clothed meshes, single-colored piece-wise.'''
        self._process_optional_arguments(*args, **kwargs)
        rgbs = []
        # NOTE: Need this particular order because of the way I produce segmaps.
        for mesh_part, mesh in zip(['body', 'garment'], meshes):
            print(f'Rendering {mesh_part} mesh...')
            fragments = self.rasterizer(
                mesh, 
                cameras=self.cameras
            )
            #try:
            #    mesh.verts_list()
            #except RuntimeError as cuda_err:
            #    return None, None
            rgb_image = self.rgb_shader(
                fragments, 
                mesh, 
                lights=self.lights_rgb_render
            )[:, :, :, :3]
            rgbs.append(rgb_image)

        return self._get_img_diff(rgbs[-2], rgbs[-1]).swapaxes(1, 2)
        # NOTE: You need the bottom part when fitting pose and shape params also.
        #return torch.cat([
        #    self._get_img_diff(rgbs[-1], torch.zeros_like(rgbs[-1])),
        #    self._get_img_diff(rgbs[-2], rgbs[-1])
        #], dim=0).swapaxes(1, 2)    # TODO: Verify whether the order is ['<garment>', 'whole']
