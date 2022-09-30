import torch
import numpy as np

from utils.mesh_utils import concatenate_meshes

from tailornet_for_garmentor.models.smpl4garment import SMPL4Garment
from tailornet_for_garmentor.models.tailornet_model import get_best_runner as get_tn_runner
from tailornet_for_garmentor.utils.rotation import normalize_y_rotation
from tailornet_for_garmentor.utils.interpenetration import remove_interpenetration_fast


class ParametricModel(object):

    def __init__(self):
        self.smpl_model_dict = dict()
        self.tailornet_model_dict = dict()
        for gender in ['male', 'female']:
            self.smpl_model[gender] = SMPL4Garment(gender=gender)
            self.tailornet_model[gender] = dict()
            for garment_class in ['t-shirt']:   # for now, use only T-shirt
                self.tailornet_model[gender][garment_class] = get_tn_runner(
                    gender=gender, garment_class=garment_class)

    def _run_tailornet(self, gender, garment_class, pose, shape, style):
        if garment_class is None:
            return None
        
        norm_pose = normalize_y_rotation(pose)

        with torch.no_grad():
            # Run TailorNet model to get the displacements based on the provided parameters.
            garment_disp = self.tailornet_model_dict[gender][garment_class].forward(
                thetas=torch.from_numpy(norm_pose[None, :].astype(np.float32)).cuda(),
                betas=torch.from_numpy(shape[None, :].astype(np.float32)).cuda(),
                gammas=torch.from_numpy(style[None, :].astype(np.float32)).cuda(),
            )[0].cpu().numpy()

        return garment_disp

    def _run_smpl4garment(self, gender, garment_class, pose, shape, garment_disp):
        if garment_class is None:
            return None
        
        # Get the predicted body and garment meshes - simply SMPL + the above displacements.
        smpl_output = self.smpl_model_dict[gender].run(
            beta=shape, 
            theta=pose, 
            garment_class=garment_class, 
            garment_d=garment_disp)

        return smpl_output

    def _remove_interpenetrations(upper, lower):
        # Resolve interpenetrations between body and garment meshes.
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
        
        # Remove interpenetrations between the upper and body-lower meshes.
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

    def run(self, gender, garment_classes, pose, shape, style_vector):
        # Run TailorNet to compute parameterized clothing displacements.
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

        # Run SMPL4Garment to obtain parameterized body and garment meshes.
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
