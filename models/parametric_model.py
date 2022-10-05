from typing import Tuple, Union
from tailornet_for_garmentor.models.smpl4garment_utils import SMPL4GarmentOutput
import torch
import numpy as np
from utils.garment_classes import GarmentClasses

from utils.mesh_utils import concatenate_meshes

from tailornet_for_garmentor.models.smpl4garment import SMPL4Garment
from tailornet_for_garmentor.models.tailornet_model import get_best_runner as get_tn_runner
from tailornet_for_garmentor.utils.rotation import normalize_y_rotation
from tailornet_for_garmentor.utils.interpenetration import remove_interpenetration_fast


class ParametricModel(object):

    '''Gathers functionalities of TailorNet and SMPL4Garment.'''

    def __init__(self):
        '''Initialize TailorNet and SMPL dictionaries for each garment and gender.'''

        self.smpl_model_dict = dict()
        self.tailornet_model_dict = dict()
        for gender in ['male', 'female']:
            self.smpl_model[gender] = SMPL4Garment(gender=gender)
            self.tailornet_model[gender] = dict()
            for garment_class in ['t-shirt']:   # for now, use only T-shirt
                self.tailornet_model[gender][garment_class] = get_tn_runner(
                    gender=gender, garment_class=garment_class)

    def _run_tailornet(self, 
                       gender: str, 
                       garment_class: str, 
                       pose: np.ndarray, 
                       shape: np.ndarray, 
                       style: np.ndarray) -> np.ndarray:
        '''Estimate garment displacement given the parameters.'''

        if garment_class is None:
            return None
        norm_pose = normalize_y_rotation(pose)

        with torch.no_grad():
            garment_disp = self.tailornet_model_dict[gender][garment_class].forward(
                thetas=torch.from_numpy(norm_pose[None, :].astype(np.float32)).cuda(),
                betas=torch.from_numpy(shape[None, :].astype(np.float32)).cuda(),
                gammas=torch.from_numpy(style[None, :].astype(np.float32)).cuda(),
            )
        return garment_disp[0].cpu().numpy()

    def _run_smpl4garment(self, 
                          gender: str, 
                          garment_class: str, 
                          pose: np.ndarray, 
                          shape: np.ndarray, 
                          garment_disp: np.ndarray) -> SMPL4GarmentOutput:
        '''Run SMPL4Garment model given the parameters and garment displacements.'''

        if garment_class is None:
            return None

        return self.smpl_model_dict[gender].run(
            beta=shape, 
            theta=pose, 
            garment_class=garment_class, 
            garment_d=garment_disp)

    @staticmethod
    def _remove_interpenetrations(upper: SMPL4GarmentOutput, 
                                  lower: SMPL4GarmentOutput
                                  ) -> Tuple[SMPL4GarmentOutput, SMPL4GarmentOutput]:
        ''' Resolve complex interpenetrations between the meshes.
        
            First resolve interpenetrations between the body and lower garment
            (if provided), and then between the body and upper garment (if provided).
            Then, if both upper and lower garment meshes are provided, resolve the
            interpenetrations between the concatenated body-lower mesh and the
            upper mesh.
        '''
        if lower is not None:
            lower.garment_verts = remove_interpenetration_fast(
                garment_verts=lower.garment_verts,
                garment_faces=lower.garment_faces,
                base_verts=lower.body_verts,
                base_faces=lower.body_faces
            )
        if upper is not None:
            upper.garment_verts = remove_interpenetration_fast(
                garment_verts=upper.garment_verts,
                garment_faces=upper.garment_faces,
                base_verts=upper.body_verts,
                base_faces=upper.body_faces
            )
        
        if lower is not None and upper is not None:
            body_lower_verts, body_lower_faces = concatenate_meshes(
                [lower.body_verts, lower.garment_verts],
                [lower.body_faces, lower.garment_faces]
            )

            upper.garment_verts = remove_interpenetration_fast(
                garment_verts=upper.garment_verts,
                garment_faces=upper.garment_faces,
                base_verts=body_lower_verts,
                base_faces=body_lower_faces
            )

        return upper, lower

    def run(self, 
            gender: str, 
            garment_classes: GarmentClasses, 
            pose: np.ndarray, 
            shape: np.ndarray, 
            style_vector: np.ndarray
            ) -> Tuple[SMPL4GarmentOutput, SMPL4GarmentOutput]:
        '''Run the parametric model (TN, SMPL) and solve interpenetrations.'''

        upper_garment_disp = self._run_tailornet(
            gender=gender,
            garment_class=garment_classes.upper_class,
            pose=pose,
            shape=shape,
            style=style_vector[garment_classes.upper_label]
        )
        lower_garment_disp = self._run_tailornet(
            gender=gender,
            garment_class=garment_classes.lower_class,
            pose=pose, 
            shape=shape, 
            style=style_vector[garment_classes.lower_label]
        )

        upper_smpl_output = self._run_smpl4garment(
            gender=gender,
            garment_class=garment_classes.upper_class,
            pose=pose,
            shape=shape,
            garment_disp=upper_garment_disp
        )
        lower_smpl_output = self._run_smpl4garment(
            gender=gender,
            garment_class=garment_classes.lower_class,
            pose=pose,
            shape=shape,
            garment_disp=lower_garment_disp
        )

        upper_smpl_output, lower_smpl_output = self._remove_interpenetrations(
            upper_smpl_output, lower_smpl_output)

        return upper_smpl_output, lower_smpl_output
