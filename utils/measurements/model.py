import torch
import numpy as np

import smplx
from smplx.body_models import SMPL
from smplx.utils import SMPLOutput

from configs.const import NUM_SHAPE_PARAMS
from configs import paths


SMPL_NUM_KPTS = 23
SMPLX_NUM_KPTS = 21


def set_shape(
        model: SMPL, 
        shape_coefs: np.ndarray
    ) -> SMPLOutput:
    shape_tensor = torch.tensor(shape_coefs, dtype=torch.float32)
    return model(
        betas=shape_tensor, 
        return_verts=True
    )


def create_model(
        gender: str, 
        model_type='smpl'
    ) -> SMPL:
    if model_type == 'star':
        return smplx.star.STAR()
    else:
        if model_type == 'smpl':
            body_pose = torch.zeros((1, SMPL_NUM_KPTS * 3))
        elif model_type == 'smplx':
            body_pose = torch.zeros((1, SMPLX_NUM_KPTS * 3))
        else:
            body_pose = torch.zeros((1, SMPL_NUM_KPTS * 3))
        return smplx.create(
            model_path=paths.SMPL_PATH_TEMPLATE.format(gender=gender.upper()), 
            model_type=model_type,
            gender=gender, use_face_contour=False,
            num_betas=NUM_SHAPE_PARAMS,
            body_pose=body_pose,
            ext='npz'
        )
