from typing import Tuple
from psbody.mesh import Mesh

from data.mesh_managers.common import (
    MeshManager,
    create_psbody_meshes
)

from tailornet_for_garmentor.models.smpl4garment_utils import SMPL4GarmentOutput


class NoColorGarmentsMeshManager(MeshManager):

    def __init__(self):
        super().__init__(self)

    @staticmethod
    def create_meshes(
            smpl_output_dict: SMPL4GarmentOutput
    ) -> Tuple[Mesh, Mesh, Mesh]:
        return create_psbody_meshes(smpl_output_dict)
