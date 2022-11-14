import os
from os import path as osp
import argparse
import numpy as np
from itertools import compress
import pickle5
import sys
import pandas as pd
import random
from typing import Tuple
from tqdm import tqdm
from random import randrange

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
    ) -> Tuple[pd.Dataframe, int]:
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
        pkl_fpath = osp.join(path_cam_files, 'pkl_file')
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


def generate_scene_samples(
    subjects_base_dir: str,
    scene_name: str,
    scenes_to_generate: int = -1
    ):
    df, number_samples = _sample_scene_data(
        subjects_base_dir,
        scene_name,
        scenes_to_generate
    )

    # make sure that each garment combination is represented evenly in the
    # dataset
    garment_combinations = [
        element for element in os.listdir(SUBJECT_OBJ_SAVEDIR) \
            if osp.isdir(osp.join(SUBJECT_OBJ_SAVEDIR, element))
    ]
    num_garment_combinations = len(garment_combinations)
    gar_comb_occurences = np.zeros(num_garment_combinations, np.int32)

    for idx in tqdm(range(number_samples), desc="Generating scenes"):
        # Generate scene as given by the data
        scene = df.iloc[idx]
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

        output_dirpath = osp.join(
            SCENE_OBJ_SAVEDIR,
            scene_name,
            osp.splitext(img_path)[0]
        )

        for subject_idx, gt_path_smplx in enumerate(gt_paths_smplx):
            # Select garment combination
            selected_garment_combination = _select_garment(gar_comb_occurences)
            # this index can be e.g. used into garment_combinations to get the name



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--part-idx', '-I', type=int,
        help='Part index specifying PKL file index for camera information.')
    parser.add_argument('--scene-idx', '-S', type=int,
        help='Scene index inside the specified part.')

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
