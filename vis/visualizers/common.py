from abc import abstractmethod
from typing import Union, Tuple, Optional
import torch
import numpy as np
from psbody.mesh import Mesh

try:
    from smplx.body_models import ModelOutput as SMPLOutput
except ImportError:
    from smplx.utils import SMPLOutput

from data.datasets.common import (
    get_background_paths,
    load_background
)
from data.mesh_managers.textured_garments import TexturedGarmentsMeshManager
from models.smpl_official import SMPL
from utils.convert_arrays import to_smpl_model_params
from utils.image_utils import add_rgb_background

from tailornet_for_garmentor.models.smpl4garment_utils import SMPL4GarmentOutput


class Visualizer(object):

    """ 
    The abstract Visualizer class.
    
    All of the subclasses should implement at least `Visualizer.vis` and 
    `Visualizer.vis_from_params`.
    """
    
    @abstractmethod
    def vis(self,
            kpts: Optional[np.ndarray] = None,
            back_img: Optional[np.ndarray] = None,
            skeleton: Optional[bool] = None,
            verts: Optional[Union[np.ndarray, torch.Tensor]] = None,
            smpl_output_dict: Optional[SMPL4GarmentOutput] = None,
            cam_t: Optional[np.ndarray] = None
        ) -> Union[Tuple[Mesh, Mesh, Mesh],                     # Visualizer3D
               Union[Tuple[np.ndarray, np.ndarray],         # BodyVisualizer
                     Tuple[torch.Tensor, torch.Tensor]],    
               Tuple[np.ndarray, np.ndarray],               # ClothedVisualizer
               np.ndarray]: ...                             # KeypointsVisualizer

    @abstractmethod
    def vis_from_params(
            self,
            pose: Union[np.ndarray, torch.Tensor], 
            shape: Union[np.ndarray, torch.Tensor], 
            style_vector: Optional[Union[np.ndarray, torch.Tensor]] = None,
            cam_t: Optional[np.ndarray] = None
    ) -> Union[Tuple[Mesh, Mesh, Mesh],                     # Visualizer3D
               Union[Tuple[np.ndarray, np.ndarray],         # BodyVisualizer
                     Tuple[torch.Tensor, torch.Tensor]],    
               Tuple[np.ndarray, np.ndarray],               # ClothedVisualizer
               np.ndarray]: ...                             # KeypointsVisualizer

    @abstractmethod
    def save_vis(
            self,
            vis_object: Union[Tuple[Mesh, Mesh, Mesh],  # Visualizer3D
                        np.ndarray],                    # Visualizer2D
            save_path: str
    ) -> None: ...


class Visualizer2D(Visualizer):

    """
    The Visualizer2D class for rendering 2D images of humans.
    
    The subclasses render the people (clothed or body) and optionally
    add 2D background image using `Visualizer2D.add_background`. 
    """

    def __init__(
            self,
            backgrounds_dir_path: str = None
    ) -> None:
        """
        All the 2D visualizers need background paths prepared.
        """
        super().__init__()
        self.backgrounds_dir_path = backgrounds_dir_path

        self.background_paths = None
        if backgrounds_dir_path is not None:
            self.backgrounds_paths = get_background_paths(
                backgrounds_dir_path=backgrounds_dir_path,
                num_backgrounds=1000
            )

    @staticmethod
    def create_body(
            pose: np.ndarray,
            shape: np.ndarray,
            smpl_model: SMPL
    ) -> SMPLOutput:
        """
        Create the body output based on the pose and shape parameters
        and the given SMPL body model.
        """
        glob_rotmat, pose_rotmat, shape = to_smpl_model_params(
            pose=pose, 
            shape=shape
        )
        return smpl_model(
            global_orient=glob_rotmat,
            body_pose=pose_rotmat,
            betas=shape,
            pose2rot=False
        )

    def add_background(
            self, 
            rgb_img: Union[np.ndarray, torch.Tensor],
            mask: Union[np.ndarray, torch.Tensor],
            back_img: Union[np.ndarray, torch.Tensor]
        ) -> Union[np.ndarray, torch.Tensor]:
        """
        Add random 2D background "behind" rendered person based on mask.
        """
        if self.background_paths is not None and back_img is not None:
            if back_img is None:
                back_img = load_background(
                    backgrounds_paths=self.backgrounds_paths,
                    img_wh=self.img_wh
                ).to(self.device)

            rgb_img = add_rgb_background(
                backgrounds=back_img,
                rgb=rgb_img,
                seg=mask
            )
        else:
            print('WARNING: The background paths not provided.'\
                ' Returning the original image.')
        return rgb_img
    
    @abstractmethod
    def save_vis(
            self,
            vis_object: np.ndarray,
            save_path: str
    ) -> None: ...


class Visualizer3D(Visualizer):

    """ 
    An abstract Visualizer3D to create and store 3D meshes.
    
    The main purpose of the class is to support creating and storing
    textured clothed meshes because they can't be simply rendered, but
    it is important to have 3D meshes to verify interpenetration resolution
    and texture application procedures.
    """

    def __init__(self):
        super().__init__()
        self.mesh_manager = TexturedGarmentsMeshManager(
            save_maps_to_disk=False
        )

    def save_vis(
            self,
            meshes: Tuple[Mesh, Mesh, Mesh],
            save_basepath: str
    ) -> None:
        """
        Save the visualization of 3D meshes to disk (only way to observe).
        """
        self.mesh_manager.save_meshes(
            meshes=meshes,
            save_basepath=save_basepath
        )
