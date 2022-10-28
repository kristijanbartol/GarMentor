import os
import sys
import pickle
import random
import numpy as np

from psbody.mesh import Mesh

sys.path.append('/garmentor/')

from models.parametric_model import ParametricModel
from models.smpl_conversions import smplx2smpl
from utils.augmentation.smpl_augmentation import normal_sample_style_numpy
from utils.garment_classes import GarmentClasses
from configs.poseMF_shapeGaussian_net_config import get_cfg_defaults

from data.const import (
    GARMENTOR_DIR,
    SUBJECT_OBJ_SAVEDIR,
    UV_MAPS_PATH,
    MGN_DATASET,
    SCANS_DIR
)


def texture_meshes(meshes: list, texture_paths: list, garment_tag: list, uv_maps_pth: str):
    '''
    Texture the [body, upper garment, lower garment] meshes from list meshes
    with textures from texture_paths -- mesh can be None if no mesh is available
    Arguments:
        meshes: list of 3 psbody Mesh classes for the body, upper garment and lower garment.
                a mesh can be None if you don't want to texture that part of mesh
        texture_paths: list of 3 paths to texture images for body, upper garment and lower garment
        garment_tag: list of 2 garment labels for the upper garment and lower garment
        uv_maps_pth: folder path where the uv maps are stored for each garment type
    Returns:
        textured_meshes: list of 3 textured meshes from the input
    '''

    body_mesh = meshes[0]
    ug_mesh = meshes[1]
    lg_mesh = meshes[2]

    # load pre-defined uv maps
    general_vt = np.load(f'{uv_maps_pth}/general_vt.npy')

    # texture body
    if body_mesh is not None:
        body_ft = np.load(f'{uv_maps_pth}/body_ft.npy')
        
        body_mesh.vt = general_vt
        body_mesh.ft = body_ft
        body_mesh.set_texture_image(texture_paths[0])

    # texture upper garment
    if ug_mesh is not None:
        
        ug_tag = garment_tag[0]
        ug_ft = np.load(f'{uv_maps_pth}/{ug_tag}_ft.npy')

        ug_mesh.vt = general_vt
        ug_mesh.ft = ug_ft
        ug_mesh.set_texture_image(texture_paths[1])

    # texture lower garment
    if lg_mesh is not None:

        lg_tag = garment_tag[1]
        lg_ft = np.load(f'{uv_maps_pth}/{lg_tag}_ft.npy')

        lg_mesh.vt = general_vt
        lg_mesh.ft = lg_ft
        lg_mesh.set_texture_image(texture_paths[2])


    return [body_mesh, ug_mesh, lg_mesh]



if __name__ == '__main__':
    cfg = get_cfg_defaults()

    mean_style = np.zeros(cfg.MODEL.NUM_STYLE_PARAMS, 
                        dtype=np.float32)
    delta_style_std_vector = np.ones(cfg.MODEL.NUM_STYLE_PARAMS, 
                                    dtype=np.float32) * cfg.TRAIN.SYNTH_DATA.AUGMENT.GARMENTOR.STYLE_STD
    
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
    
    texture_dirpaths = [os.path.join(MGN_DATASET, x) for x in os.listdir(MGN_DATASET)]
    
    if not os.path.exists(GARMENTOR_DIR):
        os.makedirs(GARMENTOR_DIR)
    
    for scan_dir in [x for x in os.listdir(SCANS_DIR) if 'kids' not in x]:
        scan_dirpath = os.path.join(SCANS_DIR, scan_dir)
        for pkl_fname in [x for x in os.listdir(scan_dirpath) if x.split('.')[1] == 'pkl']:
            pkl_fpath = os.path.join(scan_dirpath, pkl_fname)
            with open(pkl_fpath, 'rb') as pkl_f:
                metadata = pickle.load(pkl_f)
                
                theta = np.concatenate([metadata['global_orient'], metadata['body_pose']], axis=1)
                beta = metadata['betas']
                gender = metadata['gender']
                
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

            body_mesh = Mesh(
                v=smpl_output_dict['upper'].body_verts, 
                f=smpl_output_dict['upper'].body_faces
            )
            upper_mesh = Mesh(
                v=smpl_output_dict['upper'].garment_verts, 
                f=smpl_output_dict['upper'].garment_faces
            )
            lower_mesh = Mesh(
                v=smpl_output_dict['lower'].garment_verts, 
                f=smpl_output_dict['lower'].garment_faces
            )
            
            random_texture_dirpath = texture_dirpaths[random.randint(0, len(texture_dirpaths) - 1)]
            
            textured_meshes = texture_meshes(
                meshes=[
                    body_mesh, 
                    upper_mesh, 
                    lower_mesh
                ], 
                texture_paths=[
                    f'{random_texture_dirpath}/body_tex.jpg',
                    f'{random_texture_dirpath}/multi_tex.jpg',
                    f'{random_texture_dirpath}/multi_tex.jpg'
                ], 
                garment_tag=['t-shirt', 'pant'], 
                uv_maps_pth=UV_MAPS_PATH
            )
            
            mesh_basename = pkl_fname.split(".")[0]
            mesh_dir = os.path.join(SUBJECT_OBJ_SAVEDIR, scan_dir)
            mesh_basepath = os.path.join(SUBJECT_OBJ_SAVEDIR, scan_dir, mesh_basename)
            
            os.makedirs(mesh_dir, exist_ok=True)
            
            textured_meshes[0].write_obj(f'{mesh_basepath}-body.obj')
            textured_meshes[1].write_obj(f'{mesh_basepath}-upper.obj')
            textured_meshes[2].write_obj(f'{mesh_basepath}-lower.obj')
