from typing import Dict, Tuple, List
import torch
import numpy as np

from data.mesh_managers.dig_colored_garments import DigColoredGarmentsMeshManager
from rendering.common import Renderer
from models.dig_parametric_model import DigOutput


class DigClothedRenderer(Renderer):

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
        self.mesh_manager = DigColoredGarmentsMeshManager()
    
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

    def forward(
            self, 
            dig_output: DigOutput,
            device: str,
            *args,
            **kwargs
        ) -> Tuple[torch.Tensor, np.ndarray]:
        '''Render RGB images of clothed meshes, single-colored piece-wise.'''
        self._process_optional_arguments(*args, **kwargs)

        meshes = self.mesh_manager.create_meshes(
            dig_output=dig_output,
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

        return final_rgb, seg_maps  # 3 x seg_map for DIG
