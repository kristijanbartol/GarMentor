from typing import Tuple, Optional
import numpy as np

from models.parametric_model import ParametricModel
from rendering.clothed_renderer import ClothedRenderer
from utils.garment_classes import GarmentClasses
from utils.convert_arrays import to_tensors
from vis.visualizers.common import Visualizer2D

from tailornet_for_garmentor.models.smpl4garment_utils import SMPL4GarmentOutput


class ClothedVisualizer(Visualizer2D):

    ''' Visualize a parametric model with clothing.
    
        Note that this class does not support texture mapping because
        then I would need psbody.Mesh object and I am only using the
        pytorch3d.structures.Meshes. I can't render psbody.Mesh using
        my current renderers. The ClothedVisualizer class is a subclass
        of Visualizer2D, therefore, it has a method for adding a
        background.
    '''

    def __init__(
            self, 
            gender: str,
            upper_class: str,
            lower_class: str,
            device: str,
            backgrounds_dir_path: str = None,
            img_wh=256
        ) -> None:
        ''' Initialize the visualizer using parameter specification.'''
        super().__init__(backgrounds_dir_path)

        _garment_classes = GarmentClasses(
            upper_class, 
            lower_class
        )
        self.parametric_model = ParametricModel(
            gender=gender, 
            garment_classes=_garment_classes
        )
        self.device = device
        self.renderer = ClothedRenderer(
            device=self.device,
            batch_size=1
        )
        self.img_wh = img_wh

    def vis(self,
            smpl_output_dict: SMPL4GarmentOutput,
            cam_t: Optional[np.ndarray] = None
        ) -> Tuple[np.ndarray, np.ndarray]:
        ''' Visualize clothed mesh(es), given SMPL4GarmentOutput info.'''
        rgb_img, seg_maps = self.renderer(
            smpl_output_dict,
            self.parametric_model.garment_classes,
            cam_t
        )
        return rgb_img, seg_maps

    def vis_from_params(
            self,
            pose: np.ndarray, 
            shape: np.ndarray, 
            style_vector: np.ndarray,
            cam_t: Optional[np.ndarray] = None
        ) -> Tuple[np.ndarray, np.ndarray]:
        ''' Visualize clothed mesh(es).
        
            First, the parametric model is ran to obtain the verts.
            Then, the renderer renders the clothed meshes. The method
            returns an RGB rendered image (without background) and the
            mask of the person's silhouette. Note that the method 
            always expects Numpy arrays because visualizing TailorNet 
            will not be required in training loop, for now.
        '''
        pose, shape, style_vector = to_tensors(
            arrays=[pose, shape, style_vector]
        )
        smpl_output_dict = self.parametric_model.run(
            pose=pose,
            shape=shape,
            style_vector=style_vector
        )
        rgb_img, seg_maps = self.renderer(
            smpl_output_dict,
            self.parametric_model.garment_classes,
            cam_t
        )
        return rgb_img, seg_maps
