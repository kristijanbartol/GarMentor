from typing import Tuple
import numpy as np
from psbody.mesh import Mesh

from data.mesh_managers.textured_garments import TexturedGarmentsMeshManager
from models.parametric.tn import ParametricModel
from utils.garment_classes import GarmentClasses
from vis.visualizers.common import Visualizer3D

from tailornet_for_garmentor.models.smpl4garment_utils import SMPL4GarmentOutput


class ClothedVisualizer3D(Visualizer3D):

    """
    Visualize 3D clothed parametric mesh (texture-only).
    
    Note that this class can only produce textured meshes and not simply
    single-colored meshes because pytorch3d.structures.Meshes are required
    for single-color meshes, but then you can't easily properly save them
    as obj + texture image. On the other hand, psbody.Mesh can be saved,
    but the current functionalities do not allow creating texture maps for
    single-color meshes. This class is intended to be used in a way that
    it produces 3D meshes at the end which are then stored to the disk.
    """

    def __init__(
            self,
            gender: str,
            upper_class: str,
            lower_class: str,
        ) -> None:
        """
        Prepare texture mesh manager and parametric model.
        """
        self.gender = gender
        self.garment_classes = GarmentClasses(
            upper_class, 
            lower_class
        )
        self.mesh_manager = TexturedGarmentsMeshManager(save_maps_to_disk=True)
        self.parametric_model = ParametricModel(
            gender=gender, 
            garment_classes=self.garment_classes
        )

    def vis(self,
            smpl_output_dict: SMPL4GarmentOutput
        ) -> Tuple[Mesh, Mesh, Mesh]:
        """
        Visualize clothed mesh(es), given SMPL4GarmentOutput info.
        """
        meshes = self.mesh_manager.create_meshes(
            smpl_output_dict=smpl_output_dict
        )
        meshes = self.mesh_manager.texture_meshes(
            meshes=meshes,
            garment_classes=self.garment_classes
        )
        return (
            meshes[0],  # body mesh
            meshes[1],  # upper garment mesh
            meshes[2]   # lower garment mesh
        )

    def vis_from_params(
            self,
            pose: np.ndarray, 
            shape: np.ndarray, 
            style_vector: np.ndarray,
        ) -> Tuple[Mesh, Mesh, Mesh]:
        """
        Visualize clothed mesh(es), given pose, shape, and style params.
        """
        smpl_output_dict = self.parametric_model.run(
            pose=pose,
            shape=shape,
            style_vector=style_vector
        )
        meshes = self.vis(smpl_output_dict)
        return (
            meshes[0],  # body mesh
            meshes[1],  # upper garment mesh
            meshes[2]   # lower garment mesh
        )
