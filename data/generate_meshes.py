import numpy as np
import trimesh
import sys
import pickle
import os
from random import randint
from scipy.spatial.transform import Rotation as R

sys.path.append('/garmentor/')
os.environ["CUDA_VISIBLE_DEVICES"]="1"

from models.parametric_model import ParametricModel
from models.smpl_conversions import smplx2smpl
from utils.garment_classes import GarmentClasses
from utils.augmentation.smpl_augmentation import normal_sample_style_numpy
from utils.colors import GarmentColors, BodyColors, N
from utils.mesh_utils import concatenate_meshes
from configs.poseMF_shapeGaussian_net_config import get_cfg_defaults


OUTPUT_DIR = '/garmentor/data/output/'


def create_colored_mesh(smpl_output_dict):
    
    def random_pallete_color(pallete):
        return np.array(N(list(pallete)[randint(0, len(pallete) - 1)].value))
    
    verts_list = [
        smpl_output_dict['upper'].body_verts,
        smpl_output_dict['upper'].garment_verts,
        smpl_output_dict['lower'].garment_verts    
    ]
    faces_list=[
        smpl_output_dict['upper'].body_faces,
        smpl_output_dict['upper'].garment_faces,
        smpl_output_dict['lower'].garment_faces
    ]
    body_colors = np.ones_like(verts_list[0]) * random_pallete_color(BodyColors)
    upper_garment_colors = np.ones_like(verts_list[1]) * random_pallete_color(GarmentColors)
    lower_garment_colors = np.ones_like(verts_list[2]) * random_pallete_color(GarmentColors)
    
    concat_verts, concat_faces = concatenate_meshes(
        vertices_list=verts_list,
        faces_list=faces_list
    )
    concat_colors = np.concatenate([body_colors, upper_garment_colors, lower_garment_colors], axis=0)
    
    return trimesh.Trimesh(
        vertices=concat_verts,
        faces=concat_faces,
        vertex_colors=concat_colors
    )
    

def apply_transform(mesh, transl, global_orient):
    rotation_matrix = R.from_rotvec(global_orient[0]).as_matrix()
    transformation_matrix = np.concatenate([
        rotation_matrix, 
        np.swapaxes(transl, 0, 1)
    ], axis=1)
    transformation_matrix = np.concatenate([
        transformation_matrix,
        np.array([[0., 0., 0., 1.]])
    ], axis=0)
    mesh.apply_transform(transformation_matrix)
    return mesh


if __name__ == '__main__':
    parametric_models = {
        'male': ParametricModel(gender='male', 
                                garment_classes=GarmentClasses(
                                upper_class='t-shirt',
                                lower_class='pant')
                                ),
        'female': ParametricModel(gender='female',
                                  garment_classes=GarmentClasses(
                                  upper_class='t-shirt',
                                  lower_class='pant')
                                 )   
    }

    cfg = get_cfg_defaults()

    mean_style = np.zeros(cfg.MODEL.NUM_STYLE_PARAMS, 
                        dtype=np.float32)
    delta_style_std_vector = np.ones(cfg.MODEL.NUM_STYLE_PARAMS, 
                                    dtype=np.float32) * cfg.TRAIN.SYNTH_DATA.AUGMENT.GARMENTOR.STYLE_STD
    
    PKL_DIR = '/garmentor/notebooks/POSA_rp_poses/'


    for pkl_idx, pkl_fname in enumerate([x for x in os.listdir(PKL_DIR) if 'pkl' in x]):
        with open(os.path.join(PKL_DIR, pkl_fname), 'rb') as pkl_file:
            data = pickle.load(pkl_file)
            
            theta = np.concatenate([data['global_orient'], data['body_pose']], axis=1)
            #theta = np.concatenate([data['body_pose'], np.zeros((1, 9))], axis=1)
            beta = data['betas']
            global_orient = data['global_orient']
            transl = data['transl']
            gender = data['gender']
        
        beta, theta = smplx2smpl(
            '/data/hierprob3d/',
            '/garmentor/data/output/',
            beta,
            theta,
            overwrite_previous_output=True
        )
        
        theta = theta[0]
        beta = beta[0]
            
        style_vector = normal_sample_style_numpy(
            num_garment_classes=GarmentClasses.NUM_CLASSES,
            mean_params=mean_style,
            std_vector=delta_style_std_vector)
        
        smpl_output_dict = parametric_models[gender].run(
            pose=theta,
            shape=beta,
            style_vector=style_vector
        )
        
        mesh = create_colored_mesh(smpl_output_dict)
        trans_mesh = apply_transform(mesh, transl, global_orient)
        print(f'Exporing mesh #{pkl_idx}...')
        trans_mesh.export(os.path.join(OUTPUT_DIR, f'{pkl_idx:03d}.obj'), 
                    file_type='obj')
