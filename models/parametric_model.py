from abc import abstractmethod
from typing import Dict, Tuple
from tailornet_for_garmentor.models.smpl4garment_utils import SMPL4GarmentOutput
import torch
import torch.nn.functional as F
import numpy as np
from utils.garment_classes import GarmentClasses
import time
import os

from configs import paths
from utils.drapenet_structure import DrapeNetStructure
from utils.mesh_utils import concatenate_mesh_list

from drapenet_for_garmentor.utils_drape import (
    draping,
    load_lbs,
    load_udf,
    reconstruct
)
from drapenet_for_garmentor.smpl_pytorch.body_models import SMPL, ModelOutput
from drapenet_for_garmentor.meshudf.meshudf import get_mesh_from_udf

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
            body_lower_verts, body_lower_faces = concatenate_mesh_list(
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

    @staticmethod
    def _skinning(x, w, tfs, tfs_c_inv):
        """Linear blend skinning
        Args:
            x (tensor): deformed points. shape: [B, N, D]
            w (tensor): conditional input. [B, N, J]
            tfs (tensor): bone transformation matrices. shape: [B, J, D+1, D+1]
            tfs_c_inv (tensor): bone transformation matrices. shape: [J, D+1, D+1]
        Returns:
            x (tensor): skinned points. shape: [B, N, D]
        """
        tfs = torch.einsum('bnij,njk->bnik', tfs, tfs_c_inv)

        x_h = F.pad(x, (0, 1), value=1.0)
        x_h = torch.einsum("bpn,bnij,bpj->bpi", w, tfs, x_h)

        return x_h[:, :, :3]

    def _deforming(
            self, 
            verts_T, 
            pose, 
            beta, 
            latent_code, 
            embedder, 
            model_lbs, 
            model_lbs_shape, 
            model_lbs_deform, 
            tfs, 
            tfs_c_inv
        ):
        # vertices_garment_T - (#P, 3) 
        # tfs - (1, 24, 4, 4) 
        # tfs_c_inv - (24, 4, 4) 
        # pose - (1, 72) 
        # beta - (1, 10) 
        # latent_code - (1, 32) 

        num_v = verts_T.shape[0]
        points = verts_T.unsqueeze(0)
        points_embed = embedder(points*5)
            
        latent_code = latent_code.unsqueeze(1).repeat(1, num_v, 1)
        points_embed = torch.cat((points_embed, latent_code), dim=-1)
            
        pose_input = pose.unsqueeze(1).repeat(1, num_v, 1)
        beta_input = beta.unsqueeze(1).repeat(1, num_v, 1)

        input_lbs_deform = torch.cat((pose_input, beta_input), dim=-1)
        
        x_deform = model_lbs_deform(input_lbs_deform, points_embed)/100
        garment_deform = points + x_deform

        input_lbs_shape = torch.cat((points, beta_input), dim=-1)
        delta_shape_pred = model_lbs_shape(input_lbs_shape)
        garment_deform += delta_shape_pred

        lbs_weight = model_lbs(points)
        lbs_weight = lbs_weight.softmax(dim=-1)
        garment_skinning = self._skinning(garment_deform, lbs_weight, tfs, tfs_c_inv)

        verts_deformed = garment_skinning.squeeze() # (#P, 3) 
        return verts_deformed

    def _draping(
            self, 
            verts_T, 
            latent_codes, 
            pose, 
            beta, 
            tfs,
            tfs_c_inv,
            garment_part
        ):
        latent_code = latent_codes
        embedder, lbs, lbs_shape, lbs_deform_top, lbs_deform_bottom, _ = self.models
        lbs_deform = lbs_deform_top if garment_part == 'upper' else lbs_deform_bottom
        verts_deformed = self._deforming(
            verts_T, 
            pose, 
            beta, 
            latent_code, 
            embedder, 
            lbs, 
            lbs_shape, 
            lbs_deform, 
            tfs, 
            tfs_c_inv
        )
        return verts_deformed
    
    def get_body_output(
            self,
            pose: torch.Tensor,
            shape: torch.Tensor,
            global_orient: torch.Tensor,
            return_verts: bool
        ) -> ModelOutput:
        return self.smpl_server(
            betas=shape,
            body_pose=pose,
            global_orient=global_orient,
            return_verts=return_verts
        )
    
    def _reconstruct(
            self,
            coords_encoder, 
            decoder, 
            lat, 
            udf_max_dist=0.1, 
            resolution=256, 
            differentiable=False
        ):
        def udf_func(c):
            c = coords_encoder.encode(c.unsqueeze(0))
            p = decoder(c, lat).squeeze(0)
            p = torch.sigmoid(p)
            p = (1 - p) * udf_max_dist
            return p

        return get_mesh_from_udf(
            udf_func,
            coords_range=(-1, 1),
            max_dist=udf_max_dist,
            N=resolution,
            max_batch=2**16,
            differentiable=differentiable,
            use_fast_grid_filler=True
        )
    
    def run(self,
            pose: torch.Tensor, 
            shape: torch.Tensor, 
            style_vector: torch.Tensor,
            smpl_output: ModelOutput,
            garment_part: str
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        decoder = self.decoder_top if garment_part == 'upper' else self.decoder_bottom
        verts_T, cloth_faces = self._reconstruct(
            coords_encoder=self.coords_encoder, 
            decoder=decoder, 
            lat=style_vector, 
            udf_max_dist=0.1, 
            resolution=256, 
            differentiable=True     # crucial! (originally False)
        )
        cloth_verts = self._draping(
            verts_T=verts_T, 
            latent_codes=style_vector, 
            pose=pose, 
            beta=shape, 
            tfs=smpl_output.T,
            tfs_c_inv=self.tfs_c_inv,
            garment_part=garment_part
        )
        return cloth_verts, cloth_faces
