from typing import Dict, Tuple, List
import torch
import numpy as np
from pytorch3d.structures import Meshes
from pytorch3d.renderer import Textures

from data.mesh_managers.colored_garments.dn import ColoredGarmentsMeshManager
from data.mesh_managers.common import (
    MeshManager,
    default_upper_color,
    default_lower_color
)
from rendering.common import Renderer
from utils.garment_classes import GarmentClasses
from vis.colors import GarmentColors

from utils.drapenet_structure import DrapeNetStructure


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
    
    def _organize_seg_maps(
            self, 
            seg_maps: np.ndarray
        ) -> np.ndarray:
        ''' Organize segmentation maps in the form network will expect them.

            In particular, there will always be five maps: the first two for
            the lower garment (depending on the lower label), the second two
            for the upper garment (depending on the upper label), and the
            final for the whole clothed body.
        '''
        feature_maps = np.zeros((3, seg_maps.shape[1], seg_maps.shape[2]))
        feature_maps[-1] = seg_maps[0]
        feature_maps[0] = seg_maps[1]
        feature_maps[1] = seg_maps[2]
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
            drapenet_dict=drapenet_dict,
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
        feature_maps = self._organize_seg_maps(seg_maps)

        return final_rgb, feature_maps
