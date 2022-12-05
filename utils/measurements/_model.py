import os
import numpy as np
import torch
import trimesh
from PIL import Image
from pathlib import Path
import argparse

import smplx


SMPL_NUM_KPTS = 23
SMPLX_NUM_KPTS = 21

MODELS_DIR = '/data/hierprob/'


def set_shape(model, shape_coefs):
    if type(model) == smplx.star.STAR:
        return model(pose=torch.zeros((1, 72), device='cpu'), betas=shape_coefs, trans=torch.zeros((1, 3), device='cpu'))


    shape_coefs = torch.tensor(shape_coefs, dtype=torch.float32)
    
    return model(betas=shape_coefs, return_verts=True)


def create_model(gender, body_pose=None, num_coefs=10, model_type='smpl'):
    if model_type == 'star':
        return smplx.star.STAR()
    else:
        if model_type == 'smpl':
            if body_pose is None:
                body_pose = torch.zeros((1, SMPL_NUM_KPTS * 3))
        elif model_type == 'smplx':
            body_pose = torch.zeros((1, SMPLX_NUM_KPTS * 3))
        return smplx.create(MODELS_DIR, model_type=model_type,
                            gender=gender, use_face_contour=False,
                            num_betas=num_coefs,
                            body_pose=body_pose,
                            ext='npz')
