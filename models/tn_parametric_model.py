from typing import Dict, Tuple, Union
from tailornet_for_garmentor.models.smpl4garment_utils import SMPL4GarmentOutput
import torch
import numpy as np
from utils.garment_classes import GarmentClasses
import time

from utils.mesh_utils import concatenate_meshes

from tailornet_for_garmentor.models.smpl4garment import SMPL4Garment
from tailornet_for_garmentor.models.tailornet_model import get_best_runner as get_tn_runner
from tailornet_for_garmentor.utils.rotation import normalize_y_rotation
from tailornet_for_garmentor.utils.interpenetration import remove_interpenetration_fast


class TnParametricModel(object):

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
