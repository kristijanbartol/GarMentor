from typing import Dict
import numpy as np
import os
import torch
from smplx import SMPL

import configs.paths as paths
from utils.drapenet_structure import DrapeNetStructure

from DrapeNet.utils_drape import (
    draping,
    load_lbs,
    load_udf,
    reconstruct
)
from DrapeNet.smpl_pytorch.body_models import SMPL


class ParametricModel(object):

    '''Gathers functionalities of TailorNet and SMPL4Garment.'''

    def __init__(
            self,
            device,
            gender
        ) -> None:
        self.device = device
        self.models = load_lbs(
            checkpoints_dir=paths.DRAPENET_CHECKPOINTS,
            device=self.device
        )
        _, self.latent_codes_top, self.decoder_top = load_udf(
            checkpoints_dir=paths.DRAPENET_CHECKPOINTS, 
            code_file_name=paths.TOP_CODES_FNAME, 
            model_file_name=paths.TOP_MODEL_FNAME, 
            device=self.device
        )
        self.coords_encoder, self.latent_codes_bottom, self.decoder_bottom = load_udf(
            checkpoints_dir=paths.DRAPENET_CHECKPOINTS, 
            code_file_name=paths.BOTTOM_CODES_FNAME, 
            model_file_name=paths.BOTTOM_MODEL_FNAME, 
            device=self.device
        )
        data_body = np.load(os.path.join(paths.DRAPENET_EXTRADIR, 'body_info_f.npz'))
        self.tfs_c_inv = torch.FloatTensor(data_body['tfs_c_inv']).to(device)
        self.smpl_server = SMPL(
            model_path=paths.DRAPENET_SMPLDIR, 
            gender='f' if gender == 'female' else 'm'
        ).to(device)

    def run(self, 
            pose: np.ndarray, 
            shape: np.ndarray, 
            style_vector: np.ndarray
            ) -> Dict[str, DrapeNetStructure]:
        '''Run the parametric model (TN, SMPL) and solve interpenetrations.'''
        drapenet_dict = {}
        style_tensor = torch.from_numpy(style_vector).to(self.device)
        mesh_top, vertices_top_T, faces_top = reconstruct(
            coords_encoder=self.coords_encoder, 
            decoder=self.decoder_top, 
            lat=style_tensor[[0]], 
            udf_max_dist=0.1, 
            resolution=256, 
            differentiable=False
        )
        mesh_bottom, vertices_bottom_T, faces_bottom = reconstruct(
            coords_encoder=self.coords_encoder, 
            decoder=self.decoder_bottom, 
            lat=style_tensor[[1]], 
            udf_max_dist=0.1, 
            resolution=256, 
            differentiable=False
        )
        vertices_Ts = [vertices_top_T, vertices_bottom_T]
        faces_garments = [faces_top.cpu().numpy(), faces_bottom.cpu().numpy()]

        top_mesh, bottom_mesh, bottom_mesh_layer, body_mesh, joints = draping(
            vertices_Ts=vertices_Ts, 
            faces_garments=faces_garments, 
            latent_codes=[style_tensor[[0]], style_tensor[[1]]], 
            pose=torch.from_numpy(pose).unsqueeze(0).float().to(self.device), 
            beta=torch.from_numpy(shape).unsqueeze(0).float().to(self.device), 
            models=self.models, 
            smpl_server=self.smpl_server, 
            tfs_c_inv=self.tfs_c_inv
        )

        drapenet_dict['upper'] = DrapeNetStructure(
            garment_verts=top_mesh.vertices,
            garment_faces=faces_garments[0],
            body_verts=body_mesh.vertices,
            body_faces=body_mesh.faces,
            joints=joints
        )
        drapenet_dict['lower'] = DrapeNetStructure(
            garment_verts=bottom_mesh.vertices,
            garment_faces=faces_garments[1],
            body_verts=body_mesh.vertices,
            body_faces=body_mesh.faces,
            joints=joints
        )
        return drapenet_dict
