import os
import sys
import pickle
import random
import numpy as np
from typing import Set, List
from os import path as osp
import pandas as pd

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
    AGORA_DIR,
    CAM_DIR,
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


def __infer_subjects_for_scene(cam_file_directory: str, scene_name: str, verbosity: int = 0) -> Set[str]:
    """For the given scene, returns a list that contains the (relative) paths to all non-kid subjects that appear in this scene.
    Currently unused.
    """
    cam_files = [file for file in os.listdir(cam_file_directory)  if osp.splitext(file)[1] == '.pkl']
    data = []
    for cam_file in cam_files:
        df = pd.read_pickle(cam_file).reset_index()
        for idx, row in df.iterrows():
            if scene_name in row.imgPath:
                data.append(row)
    if verbosity > 0:
        print(f"Found {len(data)} images for scene {scene_name}")
    unique_smplx_subjects = set()
    for entry in data:
        unique_smplx_subjects.update([entry.gt_smplx_path[i] for i in range(len(entry.gt_smplx_path)) if not entry.kid[i]])
    if verbosity > 0:
        print(f"Found {len(unique_smplx_subjects)} unique subjects for scene {scene_name}")
    return unique_smplx_subjects


def _is_already_processed(output_dir: str, pickle_fpath: str) -> bool:
    """Checks whether the given pickle file has already been processed by 
    looking for the corresponding output files in the provided folder
    """
    filename = osp.basename(osp.splitext(pickle_fpath)[0])  # e.g. 10004_w_Amaya_0_0
    # These files must be present if the subject has already been processed
    required_extensions = [
        '-body.jpg', '-body.mtl', '-body.obj',
        '-upper.jpg', '-upper.mtl', '-upper.obj',
        '-lower.jpg', '-lower.mtl', '-lower.obj'
    ]
    required_files = [filename+extension for extension in required_extensions]  # e.g. 10004_w_Amaya_0_0-body.jpg
    for file in required_files:
        if not osp.isfile(osp.join(output_dir, file)):    # e.g. /data/garmentor/agora/subjects/shirt-pant/trainset_3dpeople_adults_bfh/10004_w_Amaya_0_0-body.jpg
            return False
    return True


def _save_style_parameters(
    style_upper: np.ndarray,
    style_lower: np.ndarray,
    output_dir: str,
    #dir_hierarchy: List[str],
    base_dir: str,
    subject_fname: str
    ) -> None:
    """Saves the garment style parameters to a file where they can be retrieved
    from later on. Currently hardcoded to work with 4 parameters (first 2 for
    upper and last 2 for lower body garments)
    Args:
        style_upper (np.ndarray): Style parameters for the upper garment
        style_lower (np.ndarray): Style parameters for the lower garment
        output_dir (str): The directory where the style parameter file should
            be saved to
        dir_hierarchy (List[str]): Each element represents one directory in the
            hierarchy that the subject, for which the style parameters are
            saved, is contained in. The first entry represents the root of this
            hierarchy. Example: ['trainset_3dpeople_adults_bfh'] will read as
            'trainset_3dpeople_adults_bfh/subject.obj' hierarchy.
        base_dir (str): Directory where the subject resides in. Currently only
            supports exactly one directory.

    """
    pkl_fpath = osp.join(output_dir, 'style_parameters.pkl')
    if osp.isfile(pkl_fpath):
        with open(pkl_fpath, 'rb') as pkl_file:
            param_file = pickle.load(pkl_file)
    else:
        param_file = {}
    if not base_dir in param_file.keys():
        param_file[base_dir] = {}
    param_file[base_dir][subject_fname] = {
        'upper': style_upper,
        'lower': style_lower
    }
    with open(pkl_fpath, 'wb') as pkl_file:
        pickle.dump(param_file, pkl_file)


if __name__ == '__main__':

    cfg = get_cfg_defaults()

    mean_style = np.zeros(cfg.MODEL.NUM_STYLE_PARAMS, 
                        dtype=np.float32)
    delta_style_std_vector = np.ones(cfg.MODEL.NUM_STYLE_PARAMS, 
                                    dtype=np.float32) * cfg.TRAIN.SYNTH_DATA.AUGMENT.GARMENTOR.STYLE_STD

    UPPER_GARMENT_TYPE = "t-shirt"
    LOWER_GARMENT_TYPE = "pant"

    parametric_models = {
        'male': ParametricModel(gender='male', 
                                garment_classes=GarmentClasses(
                                upper_class=UPPER_GARMENT_TYPE,
                                lower_class=LOWER_GARMENT_TYPE)
                                ),
        'female': ParametricModel(gender='female',
                                garment_classes=GarmentClasses(
                                upper_class=UPPER_GARMENT_TYPE,
                                lower_class=LOWER_GARMENT_TYPE)
                                )   
    }
    
    texture_dirpaths = [os.path.join(MGN_DATASET, x) for x in os.listdir(MGN_DATASET)]
    
    if not os.path.exists(GARMENTOR_DIR):
        os.makedirs(GARMENTOR_DIR)
    
    invalid_subjects = []
    for scan_dir in [x for x in os.listdir(SCANS_DIR) if 'kids' not in x]:
        scan_dirpath = os.path.join(SCANS_DIR, scan_dir)
        for pkl_fname in [x for x in os.listdir(scan_dirpath) if osp.splitext(x)[1] == '.pkl']:
            pkl_fpath = os.path.join(scan_dirpath, pkl_fname)   # e.g. /data/agora/smplx_gt/trainset_3dpeople_adults_bfh/10004_w_Amaya_0_0.pkl

            mesh_dir = os.path.join(SUBJECT_OBJ_SAVEDIR, f"{UPPER_GARMENT_TYPE}-{LOWER_GARMENT_TYPE}", scan_dir)
            os.makedirs(mesh_dir, exist_ok=True)
            mesh_basename = osp.splitext(pkl_fname)[0]
            mesh_basepath = os.path.join(mesh_dir, mesh_basename)          

            # Check if this subject (with the current garment combination) is already present in the output directory
            if _is_already_processed(mesh_dir, pkl_fpath):
                print(f"Subject {osp.join(scan_dir, mesh_basename)} already processed, skipping...")
                continue

            with open(pkl_fpath, 'rb') as pkl_f:
                metadata = pickle.load(pkl_f)
                try:
                    theta = np.concatenate([metadata['global_orient'], metadata['body_pose']], axis=1)
                    beta = metadata['betas']
                    gender = metadata['gender']
                except KeyError as e:
                    print(
                        f"Subject {osp.join(scan_dir, mesh_basename)} is missing a required attribute "
                        f"({e}), skipping...")
                    invalid_subjects.append(osp.join(scan_dir, mesh_basename))
                    continue
                
            beta, theta = smplx2smpl(
                '/data/hierprob3d/',
                '/data/garmentor/conversion_output/',
                beta,
                theta,
                gender=gender,
                overwrite_previous_output=True
            )
            
            theta = theta[0]
            beta = beta[0]
                
            style_vector = normal_sample_style_numpy(
                num_garment_classes=GarmentClasses.NUM_CLASSES,
                mean_params=mean_style,
                std_vector=delta_style_std_vector)

            # decode style parameter for saving
            style_upper = \
                style_vector[parametric_models[gender].labels['upper']]
            style_lower = \
                style_vector[parametric_models[gender].labels['lower']]

            _save_style_parameters(
                style_upper,
                style_lower,
                osp.join(
                    SUBJECT_OBJ_SAVEDIR,
                    f"{UPPER_GARMENT_TYPE}-{LOWER_GARMENT_TYPE}"
                ),
                scan_dir,
                mesh_basename
            )

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
            
            random_texture_dirpath = \
                texture_dirpaths[random.randint(0, len(texture_dirpaths) - 1)]
            
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

            textured_meshes[0].write_obj(f'{mesh_basepath}-body.obj')
            textured_meshes[1].write_obj(f'{mesh_basepath}-upper.obj')
            textured_meshes[2].write_obj(f'{mesh_basepath}-lower.obj')

            # modify obj files to support material in blender
            # by default, psbody mesh's write_obj() function does not utilize
            # the `usemtl` keyword, which results in blender not showing the
            # material
            for mesh_type in ["body", "upper", "lower"]:
                obj_fpath = f"{mesh_basepath}-{mesh_type}.obj"
                content = ""
                with open(obj_fpath, "r") as obj_file:
                    first_face = True
                    usemtl_encountered = False
                    for line in obj_file:
                        if not usemtl_encountered and \
                            line.strip().split(' ')[0] == 'usemtl':
                            usemtl_encountered = True
                        if first_face and line.strip().split(' ')[0] == 'f':
                            first_face = False
                            if not usemtl_encountered:
                                content += \
                                    f"usemtl {mesh_basename}-{mesh_type}\n"
                                usemtl_encountered = True
                        content += line
                with open(obj_fpath, "w") as obj_file:
                    obj_file.write(content)

    if len(invalid_subjects) > 0:
        print(
            "The following subjects were invalid and could not be processed, "
            "they have been excluded from the dataset:")
        with open(
            os.path.join(
                SUBJECT_OBJ_SAVEDIR,
                f"{UPPER_GARMENT_TYPE}-{LOWER_GARMENT_TYPE}",
                "invalid_subjects.txt"
            ),
            "w"
        ) as file:
            for sub in invalid_subjects:
                print(sub)
                file.write(f"{sub}\n")
