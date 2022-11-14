import os
from os import path as osp
import argparse
import numpy as np
from itertools import compress
import pickle
import sys
import pandas as pd
import random
from typing import Tuple
from tqdm import tqdm
from random import randrange
import time
import shutil

sys.path.append('/garmentor/')

from psbody.mesh import Mesh

from data.const import (
    SCENE_OBJ_SAVEDIR,
    SUBJECT_OBJ_SAVEDIR,
    TRAIN_CAM_DIR
)

def _sample_scene_data(
    path_cam_files: str,
    scene_name: str,
    elements_to_sample: int = -1
    ) -> Tuple[pd.DataFrame, int]:
    """Samples the given amount of scene descriptions from the provided files.
    If more elements are requested than exist, the maximum number of elements
    is returned.
    Args:
        path_cam_files (str): Path to the directory that contains all camera
            files.
        scene_name (str): Name of the scene for which scene data should be
            sampled
        elements_to_sample (int, Optional): Maximum amount of elements that
            should be sampled. If a value <= 0 is provided, samples the maximum
            amount of elements possible. Defaults to -1
    Returns:
        pd.DataFrame: The sampled scene information
        int: Number of sampled elements
    """
    next_index = 0
    df_scene = pd.DataFrame()
    for pkl_file in [file for file in os.listdir(path_cam_files) \
        if osp.splitext(file)[1] == '.pkl']:
        pkl_fpath = osp.join(path_cam_files, pkl_file)
        df = pd.read_pickle(pkl_fpath)
        # Unify indices among all cam files
        number_elements = len(df.index)
        df.index = np.arange(next_index, number_elements + next_index)
        next_index = number_elements + next_index
        # Only get the rows which contain entries for the specified scene
        df_scene = pd.concat([
            df_scene,
            df[df['imgPath'].str.contains(scene_name)]
        ])
    # df_scene now contains all entries for the given scene
    number_rows = len(df_scene.index)
    elements_to_sample = min(elements_to_sample, number_rows) \
        if elements_to_sample > 0 else number_rows
    sample_indices = random.sample(
        np.arange(number_rows).tolist(),
        elements_to_sample
    )
    df_return = df_scene.iloc[sample_indices]
    return df_return, elements_to_sample


def _select_garment(garment_combination_occurences: np.ndarray):
    """Selects a garment combination based on the already selected garments
    with the goal of evenly selecting all garments.
    Args:
        garment_combination_occurences (np.ndarray): Array that, for each
            garment combination, holds the number of times this garment
            combination has already been selected.
    Returns:
        int: The index of the garment that should be selected, based on the
            input array.
    """
    least_garment_occurences = np.where(
        garment_combination_occurences == garment_combination_occurences.min()
    )[0]
    return least_garment_occurences[
        randrange(least_garment_occurences.shape[0])]


def _read_subject_style_params(
    garment_combination: str,
    subject_base_dir: str,
    subject_fname: str
    ) -> dict:
    """Returns the garment style parameters for the given subject.
    Args:
        garment_combinations (str): Garment combinations used for this subject.
            Has to correspond to the folder name where the subject is located.
            E.g. garmentor/agora/subjects/<garment_combination>/<base_dir>/
        subject_base_dir (str): Base directory where the subject is located in.
            E.g. garmentor/agora/subjects/shirt_pant/<base_dir>/
        subject_fname (str): Filename of the subject without mesh-specific
            suffix and file extension (e.g. -body.obj, -upper.obj, -lower.obj).
    Returns:
        dict: Key 'upper' contains np.ndarray with the style parameters for the
            upper body garment type, 'lower' analog for the lower body garment.
    """
    style_fpath = osp.join(
        SUBJECT_OBJ_SAVEDIR, garment_combination, 'style_parameters.pkl'
    )
    if not osp.isfile(style_fpath):
        raise ValueError(f"No valid file: {style_fpath}")
    style_info = {}
    with open(style_fpath, 'rb') as style_file:
        style_info = pickle.load(style_file)
    if not isinstance(style_info, dict) or len(style_info.keys()) == 0:
        raise ValueError(f"Invalid garment style file: {style_fpath}")
    return style_info[subject_base_dir][subject_fname]


def _subject_idx_formatting(subject_idx: int) -> str:
    return f"{subject_idx:.3d}"


def _process_mtl_files(
    subject_idx: int,
    subject_fname: str,
    original_obj_location: str
    ) -> str:
    mtl_content = "# File automatically created by scene generation script\n"
    for mesh_type in ['body', 'upper', 'lower']:
        mtl_content += "\n"
        with open(
            osp.join(
                original_obj_location,
                f"{subject_fname}-{mesh_type}.mtl"
            ),
            'r'
        ) as mtl_file:
            for line in mtl_file:
                to_process = line.split('#')[0]
                if "newmtl" in to_process:
                    mtl_content += f"newmtl {mesh_type}\n"
                elif "map_Ka" in to_process:
                    mtl_content += f"map_Ka {_subject_idx_formatting(subject_idx)}-{mesh_type}.jpg\n"
                elif "map_Kd" in to_process:
                    mtl_content += f"map_Kd {_subject_idx_formatting(subject_idx)}-{mesh_type}.jpg\n"
                elif "map_Ks" in to_process:
                    mtl_content += f"map_Ks {_subject_idx_formatting(subject_idx)}-{mesh_type}.jpg\n"
                else:
                    mtl_content += line
    return mtl_content


def _process_obj_files(
    subject_idx: int,
    subject_fname: str,
    original_obj_location: str,
    scene_output_dir: str
    ) -> None:
    """As the obj files produced by psbody mesh do not fully support the obj
    standard, we have to process them to do so. After processing, the obj
    files, including their assigned textures, can be imported in blender.
    Args:
        subject_idx (int): Current index of the subject, will determine the
            file names.
        subject_fname (str): Base name of the subject in AGORA, determines
            which obj files will be used. Suffixes like '-body' and extensions
            should be omitted.
        original_obj_location (str): Path to the directory that contains the
            original obj files.
        scene_output_dir (str): Path to the directory where the processed obj
            files should be saved to.
    """
    if not osp.isdir(original_obj_location):
        raise ValueError("The provided source directory does not exist: "
            f"{original_obj_location}")
    os.makedirs(scene_output_dir, exist_ok=True)

    for mesh_type in ['body', 'upper', 'lower']:
        # Copy and rename texture files
        shutil.copy(
            osp.join(
                original_obj_location,
                f"{subject_fname}-{mesh_type}.jpg"
            ),
            osp.join(
                scene_output_dir,
                f"{_subject_idx_formatting(subject_idx)}-{mesh_type}.jpg"
            )
        )
    # Modify and merge material files
    mtl_content = _process_mtl_files(
        subject_idx,
        subject_fname,
        original_obj_location
    )
    with open(
        osp.join(
            scene_output_dir,
            f"{_subject_idx_formatting(subject_idx)}.mtl"
        ),
        'w'
    ) as mtl_file:
        mtl_file.write(mtl_content)
    # Modify and merge obj files



def generate_scene_samples(
    cam_info_base_dir: str,
    scene_name: str,
    scenes_to_generate: int = -1
    ):
    df, number_samples = _sample_scene_data(
        cam_info_base_dir,
        scene_name,
        scenes_to_generate
    )
    # add columns for garments and style parameters
    df['garment_combinations'] = None
    df['garment_styles'] = None

    # make sure that each garment combination is represented evenly in the
    # dataset
    garment_combinations = [
        element for element in os.listdir(SUBJECT_OBJ_SAVEDIR) \
            if osp.isdir(osp.join(SUBJECT_OBJ_SAVEDIR, element))
    ]
    num_garment_combinations = len(garment_combinations)
    gar_comb_occurences = np.zeros(num_garment_combinations, np.int32)

    camera_information = {}

    for idx in tqdm(range(number_samples), desc=f"Generating {scene_name} "
    "scenes"):
        # Generate scene as given by the data
        scene = df.iloc[idx]    # this returns a copy, don't assign to it !!!
        kids = scene['kid']
        adults = [not kid for kid in kids]

        # select only the subjects that are adults
        if np.asarray(kids).all():
            print(f"Skipping index {idx} due to only containing children")
            continue
        Xs = list(compress(scene['X'], adults))
        Ys = list(compress(scene['Y'], adults))
        Zs = list(compress(scene['Z'], adults))

        gt_paths_smplx = list(compress(scene['gt_path_smplx'], adults))
        img_path = scene['imgPath']

        camX = scene['camX']
        camY = scene['camY']
        camZ = scene['camZ']
        camYaw = scene['camYaw']

        camera_information[osp.splitext(img_path)[0]] = {
            'x': camX,
            'y': camY,
            'z': camZ,
            'yaw': camYaw
        }

        output_dirpath = osp.join(
            SCENE_OBJ_SAVEDIR,
            scene_name,
            osp.splitext(img_path)[0]
        )
        os.makedirs(output_dirpath, exist_ok=True)
        subject_garments = []   # Stores the garments of the respective subjets
        subject_styles = []     # Stores the garment style parameters of the
                                # respective subject
        for subject_idx, gt_path_smplx in enumerate(gt_paths_smplx):
            # Select garment combination
            garment_combination_index = _select_garment(gar_comb_occurences)
            # this index can be e.g. used into garment_combinations to get the name
            selected_garments = garment_combinations[garment_combination_index]
            subject_garments.append({
                'upper': selected_garments.split('_')[0],
                'lower': selected_garments.split('_')[1]
            })
            # Load garment style vectors
            subject_styles.append(
                _read_subject_style_params(
                    selected_garments,
                    osp.dirname(osp.relpath(gt_path_smplx, 'smplx_gt')),
                    osp.splitext(
                        osp.basename(osp.relpath(gt_path_smplx, 'smplx_gt'))
                    )[0]
                )
            )

            mesh_basepath = osp.join(
                SUBJECT_OBJ_SAVEDIR,
                selected_garments,
                osp.splitext(osp.relpath(gt_path_smplx, 'smplx_gt'))[0]
            )

            trans = np.array([
                Xs[subject_idx],
                Ys[subject_idx],
                Zs[subject_idx]
            ])

            for mesh_type in ['body', 'upper', 'lower']:
                mesh = Mesh(filename=f"{mesh_basepath}-{mesh_type}.obj")
                mesh.translate_vertices(trans)
                mesh.write_obj(osp.join(
                    output_dirpath,
                    f"{_subject_idx_formatting(subject_idx)}-{mesh_type}.obj"
                ))
                # TODO: make sure that obj files utilize usemtl
                # TODO: merge body, upper, and lower mesh into one obj file
        # Add garment information to dataframe
        df.iat[
            idx,
            df.columns.get_loc('garment_combinations')
        ] = subject_garments
        df.iat[idx, df.columns.get_loc('garment_styles')] = subject_styles
    # Save dataframe and camera info
    output_dir = osp.join(SCENE_OBJ_SAVEDIR, scene_name)
    with open(osp.join(output_dir, 'ground_truth.pkl'), 'wb') as file:
        df.to_pickle(file)
    with open(osp.join(output_dir, 'camera_info.pkl'), 'wb') as file:
        pickle.dump(camera_information, file)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--part-idx', '-I', type=int,
    #    help='Part index specifying PKL file index for camera information.')
    #parser.add_argument('--scene-idx', '-S', type=int,
    #    help='Scene index inside the specified part.')

    parser.add_argument('-i', '--input', type=str, help="Path to the directory"
        " that contains the .pkl files with scene informations.")
    parser.add_argument('-s', '--scene-name', type=str,
        help="Name of the scene for which scenes should be generated. E.g. '"
        "archivz' or 'brushifygrasslands'. Has to be part of the 'imgPath' "
        "attribute inside of the provided cam .pkl files.")
    parser.add_argument('-n', '--number-scenes', type=int,
        help="Maximum number of scenes that should be generated. If less data "
        "is available, fewer scenes are generated. Specify -1 to generate all "
        "possible scenes.")

    args = parser.parse_args()

    generate_scene_samples(args.input, args.scene_name, args.number_scenes)
    sys.exit(0)
    
    pkl_fname = f'train_{args.part_idx}.pkl'
    pkl_fpath = os.path.join(TRAIN_CAM_DIR, pkl_fname)

    with open(pkl_fpath, 'rb') as pkl_f:
        data = pickle5.load(pkl_f)
        num_scenes = data['X'].shape[0]
        
        if args.scene_idx >= num_scenes:
            print(f'ERROR: The selected part has only {num_scenes} scenes. '
                  f'Adjust your scene index to be < {num_scenes}.')
        
        kids = data['kid'].iloc[args.scene_idx]
        not_kids = [not kid for kid in kids]
        
        # Select only the non-kids.
        Xs = list(compress(
            data['X'].iloc[args.scene_idx],
            not_kids)
        )
        Ys = list(compress(
            data['Y'].iloc[args.scene_idx],
            not_kids)
        )
        Zs = list(compress(
            data['Z'].iloc[args.scene_idx],
            not_kids)
        )
        gt_paths_smplx = list(compress(
            data['gt_path_smplx'].iloc[args.scene_idx],
            not_kids)
        )
        
        camX = data['camX'].iloc[args.scene_idx]
        camY = data['camY'].iloc[args.scene_idx]
        camZ = data['camZ'].iloc[args.scene_idx]
        camYaw = data['camYaw'].iloc[args.scene_idx]
        img_path = data['imgPath'].iloc[args.scene_idx]
        
        output_dirpath = os.path.join(
            SCENE_OBJ_SAVEDIR,
            img_path.split('.')[0]
        )
        
        for subject_idx, gt_path_smplx in enumerate(gt_paths_smplx):
            rel_gt_path_smplx = os.path.relpath(gt_path_smplx, 'smplx_gt')
            rel_gt_path_smplx = rel_gt_path_smplx.split('.')[0]
            mesh_basepath = os.path.join(SUBJECT_OBJ_SAVEDIR, rel_gt_path_smplx)
            
            body_mesh = Mesh(filename=f'{mesh_basepath}-body.obj')
            upper_mesh = Mesh(filename=f'{mesh_basepath}-upper.obj')
            lower_mesh = Mesh(filename=f'{mesh_basepath}-lower.obj')
            
            trans = np.array([
                Xs[subject_idx],
                Ys[subject_idx],
                Zs[subject_idx]
            ])

            body_mesh.translate_vertices(trans)
            upper_mesh.translate_vertices(trans)
            lower_mesh.translate_vertices(trans)
            
            body_mesh.write_obj(output_dirpath, f'{subject_idx:02d}-body.obj')
            upper_mesh.write_obj(output_dirpath, f'{subject_idx:02d}-upper.obj')
            lower_mesh.write_obj(output_dirpath, f'{subject_idx:02d}-lower.obj')
