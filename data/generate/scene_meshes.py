import os
from os import path as osp
import argparse
import numpy as np
from itertools import compress
import pickle
import sys
import pandas as pd
import random
from typing import Tuple, List
from tqdm import tqdm
from random import randrange
import time
import shutil

sys.path.append('/garmentor/')

from psbody.mesh import Mesh

from data.const import (
    SCENE_OBJ_SAVEDIR,
    SUBJECT_OBJ_SAVEDIR,
    TRAIN_CAM_DIR,
    VAL_CAM_DIR
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
    print(f"Found {number_rows} possible configurations for scene {scene_name}")
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
    try:
        style_params = style_info[subject_base_dir][subject_fname]
    except Exception as e:
        return e
    return style_params


def _subject_idx_formatting(subject_idx: int) -> str:
    return f"{subject_idx:03d}"


def _increase_vertex_ids(
    line_split: List[str],
    previous_vertices: int,
    previous_normals: int,
    previous_textures: int
    ) -> str:
    """Based on already added number of vertices, normals, and textures, the
    current indices have to be increased to ensure uniqueness
    """
    line = "f"
    for idx in range(1, len(line_split)):
        entry = line_split[idx].strip()
        if entry == "":
            continue
        entry_split = entry.split("/")
        # Case 1: only vertex id: "f v1 v2 v3"
        if len(entry_split) == 1:
            line += f" {int(entry_split[0]) + previous_vertices}"
        # Case 2: vertex id + texture id: "f v1/vt1 v2/vt2 v3/vt3"
        elif len(entry_split) == 2:
            line += f" {int(entry_split[0]) + previous_vertices}/{int(entry_split[1]) + previous_textures}"
        elif len(entry_split) == 3:
            # Case 3: vertex id + normal id: "f v1//vn1 v2//vn2 v3//vn3"
            if entry_split[1] == '':
                line += f" {int(entry_split[0]) + previous_vertices}//{int(entry_split[2]) + previous_normals}"
            # Case 4: vertex id + texture id + normal id: "f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3"
            else:
                line += f" {int(entry_split[0]) + previous_vertices}/{int(entry_split[1]) + previous_textures}/{int(entry_split[2]) + previous_normals}"
        else:
            raise ValueError(f"Could not parse the given face definition {line_split}")
    line += "\n"
    return line


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
                    mtl_content += f"newmtl {_subject_idx_formatting(subject_idx)}-{mesh_type}\n"
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
    scene_output_dir: str,
    mtl_fname: str
    ) -> str:
    """Assumes from obj files:
        - no negative indices
        - a single object per file (especially no object definition with "o")
    """
    obj_content = "# File automatically created by scene generation script\n\n"
    obj_content += f"mtllib {mtl_fname}" # missing \n is intended
    previous_vertices = 0
    previous_normals = 0
    previous_textures = 0
    for mesh_type in ['body', 'upper', 'lower']:
        loop_vertex_counter = 0
        loop_normal_counter = 0
        loop_texture_counter = 0
        encountered_first_face = False
        encountered_first_vertex = False
        obj_content += "\n"
        obj_fpath = osp.join(
            scene_output_dir,
            f"{_subject_idx_formatting(subject_idx)}-{mesh_type}.obj"
        )
        with open(obj_fpath, 'r') as obj_file:
            for line in obj_file:
                to_process = line.split('#')[0].strip()
                line_split = to_process.split(' ')
                # Check whether this line defines a vertex, normal, or texture
                # and increase the respective counter
                if not encountered_first_vertex and line_split[0] == 'v':
                    if "nan" in to_process:
                        raise ValueError(f"Ecountered nan in file {obj_fpath}")
                    else:
                        encountered_first_vertex = True
                if line_split[0] == 'v':
                    loop_vertex_counter += 1
                elif line_split[0] == 'vn':
                    loop_normal_counter += 1
                elif line_split[0] == 'vt':
                    loop_texture_counter += 1
                # Remove any previous definition of material files
                if "mtllib" in to_process:
                    continue
                # Remove any previous usemtl statements
                if "usemtl" in to_process:
                    continue
                # Check whether we have to insert usemtl and object
                # definition statements
                if not encountered_first_face and line_split[0] == 'f':
                    obj_content += f"o {_subject_idx_formatting(subject_idx)}-{mesh_type}\n"
                    obj_content += f"usemtl {_subject_idx_formatting(subject_idx)}-{mesh_type}\n"
                    encountered_first_face = True
                # Need to modify vertex, normal, and texture indices if we have
                # already added other meshes
                if line_split[0] == 'f':
                    line = _increase_vertex_ids(
                        line_split,
                        previous_vertices,
                        previous_normals,
                        previous_textures
                    )
                obj_content += line
        previous_vertices += loop_vertex_counter
        previous_normals += loop_normal_counter
        previous_textures += loop_texture_counter
    return obj_content


def _process_mesh_files(
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

    # Copy and rename texture files
    for mesh_type in ['body', 'upper', 'lower']:
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
    obj_content = _process_obj_files(
        subject_idx,
        scene_output_dir,
        f"{_subject_idx_formatting(subject_idx)}.mtl"
    )
    with open(
        osp.join(
            scene_output_dir,
            f"{_subject_idx_formatting(subject_idx)}.obj"
        ),
        'w'
    ) as obj_file:
        obj_file.write(obj_content)
    return


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
    if number_samples == 0:
        print(f"Could not find information for scene {scene_name}, skipping...")
        return
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

    transformation_info = {}
    invalid_subjects = []

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
        Yaws = list(compress(scene['Yaw'], adults))

        gt_paths_smplx = list(compress(scene['gt_path_smplx'], adults))
        img_path = scene['imgPath']

        camX = scene['camX']
        camY = scene['camY']
        camZ = scene['camZ']
        camYaw = scene['camYaw']

        transformation_info[osp.splitext(img_path)[0]] = {
            'camera': {'x': camX, 'y': camY, 'z': camZ, 'yaw': camYaw}
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
            mesh_basepath = osp.join(
                SUBJECT_OBJ_SAVEDIR,
                selected_garments,
                osp.splitext(osp.relpath(gt_path_smplx, 'smplx_gt'))[0]
            )
            subject_garments.append({
                'upper': selected_garments.split('+')[0],
                'lower': selected_garments.split('+')[1]
            })
            # Load garment style vectors
            style_params = _read_subject_style_params(
                selected_garments,
                osp.dirname(osp.relpath(gt_path_smplx, 'smplx_gt')),
                osp.splitext(
                    osp.basename(osp.relpath(gt_path_smplx, 'smplx_gt'))
                )[0]
            )
            if isinstance(style_params, Exception):
                # Error, most likely, the referenced subject is invalid
                # e.g. the case for subject "rp_patrick_posed_003_0_0"
                print(f"Subject {mesh_basepath} invalid, skipping")
                invalid_subjects.append(f"Image: {img_path}")
                invalid_subjects.append(f"Subject: {mesh_basepath}")
                invalid_subjects.append(f"Error: {style_params}")
                subject_garments[-1] = None
                subject_styles.append(None)
                continue
            subject_styles.append(style_params)

            for mesh_type in ['body', 'upper', 'lower']:
                shutil.copy(
                    f"{mesh_basepath}-{mesh_type}.obj",
                    osp.join(
                        output_dirpath,
                        f"{_subject_idx_formatting(subject_idx)}-"
                        f"{mesh_type}.obj"
                    )
                )
            try:
                _process_mesh_files(
                    subject_idx,
                    osp.splitext(
                        osp.basename(osp.relpath(gt_path_smplx, 'smplx_gt'))
                    )[0],
                    osp.join(
                        SUBJECT_OBJ_SAVEDIR,
                        selected_garments,
                        osp.dirname(osp.relpath(gt_path_smplx, 'smplx_gt'))
                    ),
                    output_dirpath
                )
            except Exception as e:
                # Some error in our generated subject mesh, e.g. nan values
                # We want to ignore the subject --> delete all files we created
                # for it and continue with next subject iteration
                print(f"Subject {mesh_basepath} invalid, skipping...")
                invalid_subjects.append(f"Image: {img_path}")
                invalid_subjects.append(f"Subject: {mesh_basepath}")
                invalid_subjects.append(f"Error: {e}\n")
                for element in os.listdir(output_dirpath):
                    if _subject_idx_formatting(subject_idx) in element:
                        os.remove(osp.join(output_dirpath, element))
                # Also set last entries for garment information to None
                subject_garments[-1] = None
                subject_styles[-1] = None
                continue

            transformation_info[osp.splitext(img_path)[0]][
                _subject_idx_formatting(subject_idx)
            ] = {
                'x': Xs[subject_idx],
                'y': Ys[subject_idx],
                'z': Zs[subject_idx],
                'yaw': Yaws[subject_idx]
            }

            # Delete intermediate obj files
            for mesh_type in ['body', 'upper', 'lower']:
                intermediate_fpath = osp.join(
                    output_dirpath,
                    f"{_subject_idx_formatting(subject_idx)}-{mesh_type}.obj"
                )
                if osp.isfile(intermediate_fpath):
                    os.remove(intermediate_fpath)
        # Update garment information with None elements for kids
        for element_in_row, adult in enumerate(adults):
            if not adult:
                subject_garments.insert(element_in_row, None)
                subject_styles.insert(element_in_row, None)
        # Add garment information to dataframe
        df.iat[
            idx,
            df.columns.get_loc('garment_combinations')
        ] = subject_garments
        df.iat[idx, df.columns.get_loc('garment_styles')] = subject_styles
        # Save dataframe and camera info
        output_dir = osp.join(SCENE_OBJ_SAVEDIR, scene_name)
        with open(osp.join(output_dir, 'ground_truth_buffer.pkl'), 'wb') as file:
            df.to_pickle(file)
        with open(osp.join(output_dir, 'transformation_info_buffer.pkl'), 'wb') as file:
            pickle.dump(transformation_info, file, 4)   # protocol 4 important for compatibility with Unreal Engine
        os.replace(
            osp.join(output_dir, 'ground_truth_buffer.pkl'),
            osp.join(output_dir, 'ground_truth.pkl')
        )
        os.replace(
            osp.join(output_dir, 'transformation_info_buffer.pkl'),
            osp.join(output_dir, 'transformation_info.pkl')
        )
    with open(
        osp.join(
            SCENE_OBJ_SAVEDIR,
            scene_name,
            'invalid_subjects.txt'
        ),
        'w'
    ) as file:
        for entry in invalid_subjects:
            file.write(f"{entry}\n")
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--scene-names', type=str, nargs="+",
        help="Name of the scenes for which subjects should be generated. E.g. "
        "'archivz' or 'brushifygrasslands'. Has to be part of the 'imgPath' "
        "attribute inside of the provided cam .pkl files.")
    parser.add_argument('-n', '--number-scenes', type=int, nargs="+",
        help="Maximum number of scenes that should be generated. If less data "
        "is available, fewer scenes are generated. Specify -1 to generate all "
        "possible scenes. If one value is provided, this counts for all scenes"
        ". If more than one value is provided, each value is used for the "
        "corresponding scene.")
    parser.add_argument('--validation', action='store_true', help="Generate "
        "data from the validation set.")

    args = parser.parse_args()

    if len(args.number_scenes) == 1:
        args.number_scenes = [
            args.number_scenes[0] for _ in range(len(args.scene_names))
        ]
    else:
        assert len(args.number_scenes) == len(args.scene_names), f"number-scenes must have length 1 or same length as scene-names: {len(args.number_scenes)} vs {len(args.scene_names)}"

    for scene_idx in range(len(args.scene_names)):
        generate_scene_samples(
            VAL_CAM_DIR if args.validation else TRAIN_CAM_DIR,
            args.scene_names[scene_idx],
            args.number_scenes[scene_idx]
        )
