from typing import Dict, Tuple, Union
from tailornet_for_garmentor.models.smpl4garment_utils import SMPL4GarmentOutput
import torch
import numpy as np
from utils.garment_classes import GarmentClasses
import time
import os
from dataclasses import dataclass, fields

from utils.mesh_utils import concatenate_meshes

from DIG_for_garmentor.networks import IGR, lbs_mlp, learnt_representations
from DIG_for_garmentor.smpl_pytorch.smpl_server import SMPLServer
from DIG_for_garmentor.utils.deform import rotate_root_pose_x, infer

from tailornet_for_garmentor.models.smpl4garment import SMPL4Garment
from tailornet_for_garmentor.models.tailornet_model import get_best_runner as get_tn_runner
from tailornet_for_garmentor.utils.rotation import normalize_y_rotation
from tailornet_for_garmentor.utils.interpenetration import remove_interpenetration_fast


@dataclass
class DigOutput:

    # Mandatory values.
    body_verts: Union[torch.Tensor, np.ndarray]
    body_faces: Union[torch.Tensor, np.ndarray]
    upper_verts: Union[torch.Tensor, np.ndarray]
    upper_faces: Union[torch.Tensor, np.ndarray]
    lower_verts: Union[torch.Tensor, np.ndarray]
    lower_faces: Union[torch.Tensor, np.ndarray]
    joints_3d: np.ndarray
    upper_style: np.ndarray
    lower_style: np.ndarray

    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __iter__(self):
        return self.keys()

    def keys(self):
        keys = [t.name for t in fields(self)]
        return iter(keys)

    def values(self):
        values = [getattr(self, t.name) for t in fields(self)]
        return iter(values)

    def items(self):
        data = [(t.name, getattr(self, t.name)) for t in fields(self)]
        return iter(data)


class DigParametricModel(object):

    '''Gathers functionalities of TailorNet and SMPL4Garment.'''

    def __init__(self, 
                 gender: str
        ) -> None:
        '''Initialize TailorNet and SMPL dictionaries for each garment and gender.'''
        print('Initializing DIG parametric model...')
        ''' dir to dump mesh '''
        self.output_folder = './output'
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        ''' Load pretrained models and necessary files '''
        data = np.load('extra-data/shapedirs_f.npz')
        self.shapedirs = torch.FloatTensor(data['shapedirs']).cuda()
        self.tfs_weighted_zero = torch.FloatTensor(data['tfs_weighted_zero']).cuda()
        lbs_weights = torch.FloatTensor(data['lbs_weights']).cuda()
        num_v = len(self.shapedirs)

        self.model_blend_weight = lbs_mlp.lbs_mlp_scanimate(d_in=3, d_out=num_v, width=512, depth=8, skip_layer=[4]).cuda().eval()
        self.model_blend_weight.load_state_dict(torch.load('extra-data/pretrained/blend_weight.pth'))

        numG = 100
        dim_latentG = 12

        model_G_shirt = IGR.ImplicitNet_multiG(d_in=3+dim_latentG, skip_in=[4]).cuda().eval()
        model_G_pants = IGR.ImplicitNet_multiG(d_in=3+dim_latentG, skip_in=[4]).cuda().eval()
        model_G_shirt.load_state_dict(torch.load('extra-data/pretrained/shirt.pth'))
        model_G_pants.load_state_dict(torch.load('extra-data/pretrained/pants.pth'))

        self.model_G_shirt = model_G_shirt.cuda().eval()
        self.model_G_pants = model_G_pants.cuda().eval()

        model_rep_shirt = learnt_representations.Network(cloth_rep_size=dim_latentG, samples=numG)
        model_rep_pants = learnt_representations.Network(cloth_rep_size=dim_latentG, samples=numG)
        model_rep_shirt.load_state_dict(torch.load('extra-data/pretrained/shirt_rep.pth'))
        model_rep_pants.load_state_dict(torch.load('extra-data/pretrained/pants_rep.pth'))
        model_rep_shirt = model_rep_shirt.cuda().eval()
        model_rep_pants = model_rep_pants.cuda().eval()
        print('Load SDF model done!')

        self.embedder, embed_dim = lbs_mlp.get_embedder(4)
        d_width = 512
        dim_theta = 72
        dim_theta_p = 128 
        model_lbs = lbs_mlp.lbs_mlp_scanimate(d_in=3, d_out=24, width=d_width, depth=8, skip_layer=[4])
        model_lbs.load_state_dict(torch.load('extra-data/pretrained/lbs_shirt.pth'))
        self.model_lbs = model_lbs.cuda().eval()

        model_lbs_p = lbs_mlp.lbs_mlp_scanimate(d_in=3, d_out=24, width=d_width, depth=8, skip_layer=[4])
        model_lbs_p.load_state_dict(torch.load('extra-data/pretrained/lbs_pants.pth'))
        self.model_lbs_p = model_lbs_p.cuda().eval()
        print('Load lbs model done!')

        model_lbs_delta = lbs_mlp.lbs_pbs(d_in_theta=dim_theta, d_in_x=embed_dim, d_out_p=dim_theta_p, skip=True, hidden_theta=d_width, hidden_matrix=d_width)
        self.model_lbs_delta = model_lbs_delta.cuda().eval()
        self.model_lbs_delta.load_state_dict(torch.load('extra-data/pretrained/lbs_delta_shirt.pth'))

        model_lbs_delta_p = lbs_mlp.lbs_pbs(d_in_theta=dim_theta, d_in_x=embed_dim, d_out_p=dim_theta_p, skip=True, hidden_theta=d_width, hidden_matrix=d_width)
        self.model_lbs_delta_p = model_lbs_delta_p.cuda().eval()
        self.model_lbs_delta_p.load_state_dict(torch.load('extra-data/pretrained/lbs_delta_pants.pth'))
        print('Load lbs_delta model done!')

        ''' Initialize SMPL model '''
        rest_pose = np.zeros((24,3), np.float32)
        rest_pose[1,2] = 0.15
        rest_pose[2,2] = -0.15
        rest_pose = rotate_root_pose_x(rest_pose)

        param_canonical = torch.zeros((1, 86),dtype=torch.float32).cuda()
        param_canonical[0, 0] = 1
        param_canonical[:,4:76] = torch.FloatTensor(rest_pose).reshape(-1)
        self.smpl_server = SMPLServer(param_canonical, gender='f', betas=None, v_template=None)
        self.tfs_c_inv = self.smpl_server.tfs_c_inv.detach()

        # NOTE: DIG provides a 100 fixed parameters for 100 shirts and 100 pants - this is their limitation.
        self.z_style_shirt = model_rep_shirt.weights[0] # these weights are pretrained latent code for the 100 shirts
        self.z_style_pants = model_rep_pants.weights[0] # these weights are pretrained latent code for the 100 pants

    def run(self, 
            pose: np.ndarray, 
            shape: np.ndarray
        ) -> DigOutput:
        '''Estimate garment displacement given the parameters.'''
        shirt_randint, pants_randint = torch.randint(100, size=(2,))
        upper_style = self.z_style_shirt[shirt_randint]
        lower_style = self.z_style_pants[pants_randint]

        body_mesh, shirt_mesh, pants_mesh, joints_3d = infer(
            pose, 
            shape, 
            self.model_G_shirt,
            self.model_G_pants, 
            upper_style, 
            lower_style, 
            self.tfs_c_inv, 
            self.shapedirs, 
            self.tfs_weighted_zero, 
            self.embedder, 
            self.model_lbs, 
            self.model_lbs_delta, 
            self.model_lbs_p, 
            self.model_lbs_delta_p, 
            self.model_blend_weight, 
            self.smpl_server, 
            self.output_folder
        )
        return DigOutput(
            body_verts=body_mesh.vertices,
            body_faces=body_mesh.faces,
            upper_verts=shirt_mesh.vertices,
            upper_faces=shirt_mesh.faces,
            lower_verts=pants_mesh.vertices,
            lower_faces=pants_mesh.faces,
            joints_3d=joints_3d.cpu().detach().numpy(),
            upper_style=upper_style.cpu().detach().numpy(),
            lower_style=lower_style.cpu().detach().numpy()
        )
