from typing import List, Tuple, Union
from abc import abstractmethod
from psbody.mesh import Mesh
from pytorch3d.structures import Meshes
import numpy as np
from random import randint

from vis.colors import norm_color

from tailornet_for_garmentor.models.smpl4garment_utils import SMPL4GarmentOutput


def random_pallete_color(pallete):
    return np.array(norm_color(list(pallete)[randint(0, len(pallete) - 1)].value))


def create_psbody_meshes(smpl_output_dict: SMPL4GarmentOutput
                    ) -> Tuple[Mesh, Mesh, Mesh]:
    ''' Create psbody.Meshes objects given SMPL4Garment output.
    
        This function is used both creating the clothed meshes which
        will be saved to disk as obj files. This is useful for the
        textured clothed meshes and also the clothed meshes without
        textures (geometry-only).
    '''
    body_mesh = Mesh(
        v=smpl_output_dict['upper'].body_verts, 
        f=smpl_output_dict['upper'].body_faces
    )
    upper_mesh = Mesh(
        v=smpl_output_dict['upper'].garment_verts, 
        f=smpl_output_dict['upper'].garment_faces
    )
    lower_mesh = Mesh(
        v=smpl_output_dict['lower'].garment_verts, 
        f=smpl_output_dict['lower'].garment_faces
    )
    return (
        body_mesh,
        upper_mesh,
        lower_mesh
    )


class MeshManager(object):

    """
    An abstract mesh manager with common declaration/definitions.
    """

    @abstractmethod
    def create_meshes(
        self,
        smpl_output: Union[np.ndarray, SMPL4GarmentOutput]
    ) -> Union[Tuple[Union[Mesh, Meshes], 
               Union[Mesh, Meshes], 
               Union[Mesh, Meshes]
               ],
               Meshes]:
        """
        Implemented for various types for different mesh manager subclasses.
        """
        pass

    def save_meshes(
            self,
            meshes: List[Mesh],
            save_basepath: str
    ) -> None:
        """
        Simply save the provided body and garment meshes as obj.
        """
        meshes[0].write_obj(f'{save_basepath}-body.obj')
        meshes[1].write_obj(f'{save_basepath}-upper.obj')
        meshes[2].write_obj(f'{save_basepath}-lower.obj')
