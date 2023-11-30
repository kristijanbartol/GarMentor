from abc import abstractmethod
from typing import Dict
from tailornet_for_garmentor.models.smpl4garment_utils import SMPL4GarmentOutput
import torch
import numpy as np
from utils.garment_classes import GarmentClasses
import time
import os

from configs import paths
from utils.drapenet_structure import DrapeNetStructure
from utils.mesh_utils import concatenate_meshes

from drapenet_for_garmentor.utils_drape import (
    draping,
    load_lbs,
    load_udf,
    reconstruct
)
from drapenet_for_garmentor.smpl_pytorch.body_models import SMPL

from tailornet_for_garmentor.models.smpl4garment import SMPL4Garment
from tailornet_for_garmentor.models.tailornet_model import get_best_runner as get_tn_runner
from tailornet_for_garmentor.utils.rotation import normalize_y_rotation
from tailornet_for_garmentor.utils.interpenetration import remove_interpenetration_fast


class ParametricModel(object):

    @abstractmethod
    def run(self):
        pass


class TNParametricModel(ParametricModel):

    '''Gathers functionalities of TailorNet and SMPL4Garment.'''

    def __init__(self, 
                 gender: str, 
                 garment_classes: GarmentClasses,
                 eval: bool = False):
        '''Initialize TailorNet and SMPL dictionaries for each garment and gender.'''

        self.garment_classes = garment_classes
        self.classes = {
            'upper': garment_classes.upper_class,
            'lower': garment_classes.lower_class
        }
        self.labels = {
            'upper': garment_classes.upper_label,
            'lower': garment_classes.lower_label
        }
        print(f'Initializing ({gender}, {self.classes["upper"]}, ' \
            f'{self.classes["lower"]}) model...')

        self.smpl_model = SMPL4Garment(gender=gender)
        self.tn_models = {
            'upper': get_tn_runner(gender=gender, garment_class=self.classes['upper']),
            'lower': get_tn_runner(gender=gender, garment_class=self.classes['lower'])
        }
        # TODO: Create a decorator for measuring (eval) time, instead of if blocks.
        self.eval = eval
        if eval:
            self.exec_times = {}

    def _run_tailornet(self, 
                       garment_part: str, 
                       pose: np.ndarray, 
                       shape: np.ndarray, 
                       style_vector: np.ndarray) -> np.ndarray:
        '''Estimate garment displacement given the parameters.'''

        assert(garment_part in ['upper', 'lower'])
        if self.classes[garment_part] is None:
            return None

        style = style_vector[self.labels[garment_part]]
        norm_pose = normalize_y_rotation(pose)

        print(f'Running TailorNet ({garment_part} -> {self.classes[garment_part]}' \
            f'(label={self.labels[garment_part]}))...')

        with torch.no_grad():
            garment_disp = self.tn_models[garment_part].forward(
                thetas=torch.from_numpy(norm_pose[None, :].astype(np.float32)).cuda(),
                betas=torch.from_numpy(shape[None, :].astype(np.float32)).cuda(),
                gammas=torch.from_numpy(style[None, :].astype(np.float32)).cuda(),
            )
        return garment_disp[0].cpu().numpy()

    def _run_smpl4garment(self, 
                          garment_part: str, 
                          pose: np.ndarray, 
                          shape: np.ndarray, 
                          garment_disp: np.ndarray) -> SMPL4GarmentOutput:
        '''Run SMPL4Garment model given the parameters and garment displacements.'''

        if self.classes[garment_part] is None:
            return None

        print(f'Running SMPL4Garment ({self.classes[garment_part]})...')

        return self.smpl_model.run(
            beta=shape, 
            theta=pose, 
            garment_class=self.classes[garment_part], 
            garment_d=garment_disp)

    @staticmethod
    def _remove_interpenetrations(
        smpl_outputs: Dict[str, SMPL4GarmentOutput]) -> Dict[str, SMPL4GarmentOutput]:
        ''' Resolve complex interpenetrations between the meshes.
        
            First resolve interpenetrations between the body and lower garment
            (if provided), and then between the body and upper garment (if provided).
            Then, if both upper and lower garment meshes are provided, resolve the
            interpenetrations between the concatenated body-lower mesh and the
            upper mesh.
        '''
        lower = smpl_outputs['lower']
        upper = smpl_outputs['upper']

        if lower is not None:
            lower.garment_verts = remove_interpenetration_fast(
                garment_verts=lower.garment_verts,
                garment_faces=lower.garment_faces,
                body_verts=lower.body_verts,
                body_faces=lower.body_faces
            )
        if upper is not None:
            upper.garment_verts = remove_interpenetration_fast(
                garment_verts=upper.garment_verts,
                garment_faces=upper.garment_faces,
                body_verts=upper.body_verts,
                body_faces=upper.body_faces
            )
        
        if lower is not None and upper is not None:
            body_lower_verts, body_lower_faces = concatenate_meshes(
                [lower.body_verts, lower.garment_verts],
                [lower.body_faces, lower.garment_faces]
            )

            upper.garment_verts = remove_interpenetration_fast(
                garment_verts=upper.garment_verts,
                garment_faces=upper.garment_faces,
                body_verts=body_lower_verts,
                body_faces=body_lower_faces
            )

        return {
            'upper': upper,
            'lower': lower
        }

    def run(self, 
            pose: np.ndarray, 
            shape: np.ndarray, 
            style_vector: np.ndarray
            ) -> Dict[str, SMPL4GarmentOutput]:
        '''Run the parametric model (TN, SMPL) and solve interpenetrations.'''

        smpl_output_dict = {}
        for garment_part in ['upper', 'lower']:
            if self.eval:
                start_time = time.time()
            garment_disp = self._run_tailornet(
                garment_part=garment_part,
                pose=pose,
                shape=shape,
                style_vector=style_vector
            )
            if self.eval:
                self.exec_times['tailornet-time'] = time.time() - start_time
                start_time = time.time()
            smpl_output = self._run_smpl4garment(
                garment_part=garment_part,
                pose=pose,
                shape=shape,
                garment_disp=garment_disp
            )
            if self.eval:
                self.exec_times['smpl-time'] = time.time() - start_time
            smpl_output_dict[garment_part] = smpl_output

        if self.eval:
            start_time = time.time()
        smpl_output_dict = self._remove_interpenetrations(smpl_output_dict)
        if self.eval:
            self.exec_times['interpenetrations-time'] = time.time() - start_time

        return smpl_output_dict


class DNParametricModel(ParametricModel):

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


class TorchParametricModel(DNParametricModel):

    def __init__(
            self,
            device: str,
            gender: str
        ) -> None:
        super().__init__(
            device=device,
            gender=gender
        )

    def _draping(
            self, 
            vertices_T, 
            faces_garments, 
            latent_codes, 
            pose, 
            beta, 
            models, 
            smpl_server, 
            tfs_c_inv
        ):
        faces_top, faces_bottom = faces_garments
        latent_code_top, latent_code_bottom = latent_codes
        embedder, _lbs, _lbs_shape, _lbs_deform_top, _lbs_deform_bottom, _lbs_deform_layer = models
        with torch.no_grad():
            output_smpl = smpl_server(betas=beta, body_pose=pose[:, 3:], global_orient=pose[:, :3], return_verts=True)
            tfs = output_smpl.T
            smpl_verts = output_smpl.vertices
            joints = output_smpl.joints

            _, top_mesh = deforming(vertices_top_T, faces_top, pose, beta, latent_code_top, embedder, _lbs, _lbs_shape, _lbs_deform_top, tfs, tfs_c_inv)

            _, _, bottom_mesh, bottom_mesh_layer = deforming_layer(vertices_bottom_T, faces_bottom, pose, beta, latent_code_bottom, latent_code_top, embedder, _lbs, _lbs_shape, _lbs_deform_bottom, _lbs_deform_layer, tfs, tfs_c_inv)

        body_mesh = trimesh.Trimesh(smpl_verts.squeeze().cpu().numpy(), smpl_server.faces)

        colors_f_body = np.ones((len(body_mesh.faces), 4))*np.array([255, 255, 255, 200])[np.newaxis,:]
        colors_f_top = np.ones((len(top_mesh.faces), 4))*np.array([160, 160, 255, 200])[np.newaxis,:]
        colors_f_bottom = np.ones((len(bottom_mesh.faces), 4))*np.array([100, 100, 100, 200])[np.newaxis,:]
        body_mesh.visual.face_colors = colors_f_body
        top_mesh.visual.face_colors = colors_f_top
        bottom_mesh.visual.face_colors = colors_f_bottom
        bottom_mesh_layer.visual.face_colors = colors_f_bottom
            
        return top_mesh, bottom_mesh, bottom_mesh_layer, body_mesh, joints
    
    def run(self,
            pose: torch.Tensor, 
            shape: torch.Tensor, 
            style_vector: torch.Tensor,
            garment_part: str
            ) -> Dict[str, DrapeNetStructure]:
        drapenet_dict = {}
        decoder = self.decoder_top if garment_part == 'upper' else self.decoder_bottom
        _, vertices_T, faces = reconstruct(
            coords_encoder=self.coords_encoder, 
            decoder=decoder, 
            lat=style_vector, 
            udf_max_dist=0.1, 
            resolution=256, 
            differentiable=False
        )
        #vertices_Ts = [vertices_T, vertices_bottom_T]
        #faces_garments = [faces_top.cpu().numpy(), faces_bottom.cpu().numpy()]

        top_mesh, bottom_mesh, _, body_mesh, joints = draping(
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
