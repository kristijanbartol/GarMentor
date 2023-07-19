from typing import (
    Tuple, 
    Optional, 
    List
)
from torch import Tensor
import numpy as np
from PIL import (
    Image, 
    ImageOps
)

from models.parametric.dn import ParametricModel
from rendering.clothed.dn import ClothedRenderer
from utils.garment_classes import GarmentClasses
from vis.visualizers.common import Visualizer2D

from DrapeNet.draping.drape import main


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
            backgrounds_dir_path: Optional[str] = None,
            img_wh=256
        ) -> None:
        """
        Initialize the ClothedVisualizer class.

        Either garment_classes object or upper&lower class should be 
        provided as arguments. The ClothedVisualizer contains
        ClothedRenderer, which makes it more convenient for the user not
        to think about the rendering details.
        """
        super().__init__(
            img_wh=img_wh,
            backgrounds_dir_path=backgrounds_dir_path
        )
        assert(device != 'cpu')
        self.device = device
        self.parametric_model = ParametricModel(
            device=self.device,
            gender=gender
        )
        self.renderer = ClothedRenderer(
            device=self.device,
            img_wh=img_wh    
        )
        self.img_wh = img_wh

    def vis_from_params(
            self,
            pose: np.ndarray, 
            shape: np.ndarray, 
            style_vector: np.ndarray,
            cam_t: Optional[np.ndarray] = None
        ) -> Tuple[Tensor, np.ndarray, np.ndarray]:
        """
        Visualize clothed mesh(es).
        
        First, the parametric model is ran to obtain the verts.
        Then, the renderer renders the clothed meshes. The method
        returns an RGB rendered image (without background) and the
        mask of the person's silhouette. Note that the method 
        always expects Numpy arrays because visualizing TailorNet 
        will not be required in training loop, for now.
        """
        drapenet_dict = self.parametric_model.run(
            pose=pose,
            shape=shape,
            style_vector=style_vector
        )
        rgb_img, seg_maps = self.renderer(
            drapenet_dict=drapenet_dict,
            device=self.device
        )
        return (
            rgb_img, 
            seg_maps,
            drapenet_dict['upper'].joints
        )

    @staticmethod
    def save_vis(
            rgb_img: np.ndarray,
            save_path: str
    ) -> None:
        """
        Save RGB clothed image.
        """
        rgb_img = (rgb_img * 255).astype(np.uint8)
        #pil_img = ImageOps.flip(Image.fromarray(rgb_img))
        pil_img = Image.fromarray(rgb_img)
        pil_img.save(save_path)
        print(f'Saved clothed image: {save_path}...')

    @staticmethod
    def save_masks(
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

    @staticmethod
    def save_masks_as_images(
            seg_masks: np.ndarray,
            save_paths: List[str]
    ) -> None:
        """
        Save segmentation masks as .png images for verification.
        """
        for seg_idx in range(5):
            mask_img = (seg_masks[seg_idx] * 255).astype(np.uint8)
            #pil_img = ImageOps.flip(Image.fromarray(mask_img))
            pil_img = Image.fromarray(mask_img)
            pil_img.save(save_paths[seg_idx])
