from typing import Tuple, Optional
import numpy as np
from PIL import (
    Image, 
    ImageOps
)

from models.parametric_model import ParametricModel
from rendering.clothed import ClothedRenderer
from utils.garment_classes import GarmentClasses
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
            device: str,
            gender: str,
            upper_class: Optional[str] = None,
            lower_class: Optional[str] = None,
            garment_classes: Optional[GarmentClasses] = None,
            backgrounds_dir_path: str = None,
            img_wh=256
        ) -> None:
        """
        Initialize the ClothedVisualizer class.

        Either garment_classes object or upper&lower class should be 
        provided as arguments. The ClothedVisualizer contains
        ClothedRenderer, which makes it more convenient for the user not
        to think about the rendering details.
        """
        super().__init__(backgrounds_dir_path)

        if garment_classes is None:
            assert(upper_class is not None and lower_class is not None)
            self.garment_classes = GarmentClasses(
                upper_class, 
                lower_class
            )
        else:
            self.garment_classes = garment_classes

        self.parametric_model = ParametricModel(
            gender=gender, 
            garment_classes=self.garment_classes
        )
        self.device = device
        self.renderer = ClothedRenderer(device=self.device)
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
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Visualize clothed mesh(es).
        
        First, the parametric model is ran to obtain the verts.
        Then, the renderer renders the clothed meshes. The method
        returns an RGB rendered image (without background) and the
        mask of the person's silhouette. Note that the method 
        always expects Numpy arrays because visualizing TailorNet 
        will not be required in training loop, for now.
        """
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
        return (
            rgb_img, 
            seg_maps,
            smpl_output_dict['upper'].joints
        )
    
    def save_vis(
            self,
            rgb_img: np.ndarray,
            save_path: str
    ) -> None:
        """
        Save RGB clothed image.
        """
        rgb_img = (rgb_img * 255).astype(np.uint8)
        rgb_img = ImageOps.flip(Image.fromarray(rgb_img))
        rgb_img.save(save_path)
        print(f'Saved clothed image: {save_path}...')

    def save_masks(
            self,
            seg_masks: np.ndarray,
            save_path: str
    ) -> None:
        """
        Save segmentation masks as a single .npz file.
        """
        np.savez_compressed(
            save_path, 
            seg_maps=seg_masks.astype(bool)
        )
        print(f'Saved clothing segmentation masks: {save_path}...')
