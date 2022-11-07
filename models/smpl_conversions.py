import sys
import smplx
import torch
import numpy as np
from typing import Union, Tuple, List
from tqdm import tqdm
import os
from os import path as osp
import trimesh
import argparse
import time
import pickle
import pathlib
from math import ceil

def smpl2smplx(
    path_models: str,
    output_dir: str,
    smpl_betas: Union[np.ndarray, torch.Tensor],
    smpl_thetas: Union[np.ndarray, torch.Tensor],
    gender: str = 'male',
    device: Union[str, torch.device] = None,
    overwrite_previous_output: bool = False,
    save_conversion_results: bool = True,
    conversion_results_filename: str = 'conv_results.npz',
    check_already_converted: bool = True,
    explicit_result_file: str = None,
    verbosity: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """Converts the given SMPL parameters into SMPL-X parameters. Use
    split_smplx_full_body_pose to get the sub-poses for different body parts.
    Wraps the conversion code from the smplx repository.
    Args:
        path_models (str): Path to the directory that contains the smpl and
            smplx model sub-directories
        output_dir (str): Output directory
        smpl_betas (Union[np.ndarray, torch.Tensor]): SMPL body shape
            parameters of shape (B, 10), where B is the number of different
            bodies for which the parameters should be converted
        smpl_thetas (Union[np.ndarray, torch.Tensor]): SMPL body pose
            parameters of shape (B, 72), where the first three entries encode
            the global orientation
        gender (str, optional): Gender of the SMPL body. Supported are
            'male', 'female', and 'neutral'. Defaults to 'male'
        device (Union[str, torch.device], optional): The device to use for
            computations. If not provided, decides automatically based on
            availability. Defaults to None
        overwrite_previous_output (bool, optional): Whether previously created
            meshes should be overwritten. The results of conversions are
            not affected by this. Defaults to False
        save_conversion_Results (bool, optional): Whether the SMPL and
            corresponding SMPL-X parameters should be saved together in a file
            in the output directory. This allows to retrieve them later on.
        conversion_results_filename (str, optional): Filename of the file where
            the converted parameters will be saved to. Defaults to
            'conv_results.npz'
        check_already_converted (bool, optional): Enables a check to see
            whether the given parameters have already been converted before.
            If so, they are not converted again. Instead, their previous
            conversion result is loaded and returned. Checking is done by
            searching through the conversion results file. Defaults to True.
        explicit_result_file (str, optional): When specified, the conversion
            results will be saved into this file instead of the file at the
            default location.
        verbosity (int, optional): Verbosity setting. Defaults to 1
    Returns:
        np.ndarray: SMPL-X body shape parameters (betas) of shape (B, 10)
        np.ndarray: SMPL-X full-body pose parameters (thetas) of shape
            (B, 165). Use split_smplx_full_body_pose to get the sub-poses for
            different body parts
    """
    if check_already_converted:
        previous_results = _check_already_converted(
            smpl_betas,
            smpl_thetas,
            gender,
            "smpl",
            "smplx",
            osp.join(output_dir, conversion_results_filename)
        )
        indices_unmatched = [
            idx for idx, entry in enumerate(previous_results) if entry is None
        ]
        if verbosity > 0:
                print(
                    f"{smpl_betas.shape[0] - len(indices_unmatched)} / "
                    f"{smpl_betas.shape[0]} parameter pairs have already been "
                    "converted."
                )
                if len(indices_unmatched) != 0:
                    print(
                        f"Converting the remaining {len(indices_unmatched)} "
                        "SMPL parameters..."
                    )
        if len(indices_unmatched) == 0:
            # all SMPL parameters have corresponding SMPL-X parameters
            smplx_betas = np.asarray(
                [entry['betas'] for entry in previous_results]
            )
            smplx_thetas = np.asarray(
                [entry['thetas'] for entry in previous_results]
            )
            return smplx_betas, smplx_thetas
        # we need to convert some SMPL parameters
        smpl_betas = smpl_betas[indices_unmatched]
        smpl_thetas = smpl_thetas[indices_unmatched]

    # Parameters for creating the SMPL model
    smpl_params = dict(
        model_path=path_models,
        model_type='smpl',
        betas=smpl_betas,
        global_orient=smpl_thetas[:, :3],
        body_pose=smpl_thetas[:, 3:],
        batch_size=smpl_betas.shape[0],
        gender=gender
    )
    # Conversion
    smplx_betas, smplx_thetas = _perform_conversion(
        'smplx',
        smpl_betas,
        smpl_thetas,
        smpl_params,
        output_dir,
        overwrite_previous_output,
        device,
        verbosity
    )
    smplx_thetas = _remove_pose_joint_dimension(smplx_thetas)
    # Write results to file
    if explicit_result_file is not None:
        explicit_result_file = pathlib.Path(explicit_result_file)
        output_dir = explicit_result_file.parent
        conversion_results_filename = explicit_result_file.name

    if save_conversion_results:
        _save_conversion_results(
            output_dir,
            smpl_betas,
            smpl_thetas,
            smplx_betas,
            smplx_thetas,
            gender,
            conversion_results_filename
        )
    if check_already_converted:
        # Combine with the already converted parameters we obtained earlier
        for idx, entry in enumerate(previous_results):
            if entry is not None:
                smplx_betas = np.insert(
                    smplx_betas,
                    idx,
                    entry['betas'],
                    axis=0
                )
                smplx_thetas = np.insert(
                    smplx_thetas,
                    idx,
                    entry['thetas'],
                    axis=0
                )
    return smplx_betas, smplx_thetas


def smplx2smpl(
    path_models: str,
    output_dir: str,
    smplx_betas: Union[np.ndarray, torch.Tensor],
    smplx_thetas: Union[np.ndarray, torch.Tensor],
    gender: str = 'male',
    device: Union[str, torch.device] = None,
    overwrite_previous_output: bool = False,
    save_conversion_results: bool = True,
    conversion_results_filename: str = 'conv_results.npz',
    check_already_converted: bool = True,
    explicit_result_file: str = None,
    verbosity: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """Converts the given SMPL-X parameters into SMPL parameters. Hand and face
    information will be lost during conversion. For ease of use, they can still
    be provided as part of the input. Wraps the conversion code from the smplx
    repository.
    Args:
        path_models (str): Path to the directory that contains the smpl and
            smplx model sub-directories
        output_dir (str): Output directory
        smplx_betas (Union[np.ndarray, torch.Tensor]): SMPL-X body shape
            parameters of shape (B, 10), where B is the number of different
            bodies for which parameters should be converted.
        smplx_thetas (Union[np.ndarray, torch.Tensor]): SMPL-X body pose
            parameters of shape (B, 165) or (B, 66), where the first three
            entries encode the global orientation. In case of (B, 66) shaped
            input, the remaining parameters will be set to zero automatically.
        gender (str, optional): Gender of the SMPL-X body. Supported are
            'male', 'female', and 'neutral'. Defaults to 'male'.
        device (Union[str, torch.device], optional): The device to use for
            computations. If not provided, devices automatically based on
            availability. Defaults to None.
        overwrite_previous_output (bool, optional): Whether previously created
            meshes should be overwritten. The results of conversions are
            not affected by this. Defaults to False
        save_conversion_results (bool, optional): Whether the SMPL and
            corresponding SMPL-X parameters should be saved together in a file
            in the output directory. This allows to retrieve them later on.
        conversion_results_filename (str, optional): Filename of the file where
            the converted parameters will be saved to. Defaults to
            'conv_results.npz'
        check_already_converted (bool, optional): Enables a check to see
            whether the given parameters have already been converted before.
            If so, they are not converted again. Instead, their previous
            conversion result is loaded and returned. Checking is done by
            searching through the conversion results file. Defaults to True.
        explicit_result_file (str, optional): When specified, the conversion
            results will be saved into this file instead of the file at the
            default location.
        verbosity (int, optional): Verbosity setting. Defaults to 1
    Returns:
        np.ndarray: SMPL body shape parameters (betas) of shape (B, 10)
        np.ndarray: SMPL body pose parameters (thetas) of shape (B, 72)
    """
    if smplx_thetas.shape[1] != 165:
        smplx_thetas = np.append(
            smplx_thetas,
            np.zeros((smplx_thetas.shape[0], 165 - smplx_thetas.shape[1])),
            axis=1
        )
    if check_already_converted:
        previous_results = _check_already_converted(
            smplx_betas,
            smplx_thetas,
            gender,
            "smplx",
            "smpl",
            osp.join(output_dir, conversion_results_filename)
        )
        indices_unmatched = [
            idx for idx, entry in enumerate(previous_results) if entry is None
        ]
        if verbosity > 0:
                print(
                    f"{smplx_betas.shape[0] - len(indices_unmatched)} / "
                    f"{smplx_betas.shape[0]} parameter pairs have already been"
                    " converted."
                )
                if len(indices_unmatched) != 0:
                    print(
                        f"Converting the remaining {len(indices_unmatched)} "
                        "SMPL-X parameters..."
                    )
        if len(indices_unmatched) == 0:
            # all SMPL-X parameters have corresponding SMPL parameters
            smpl_betas = np.asarray(
                [entry['betas'] for entry in previous_results]
            )
            smpl_thetas = np.asarray(
                [entry['thetas'] for entry in previous_results]
            )
            return smpl_betas, smpl_thetas
        # we need to convert some SMPL-X parameters
        smplx_betas = smplx_betas[indices_unmatched]
        smplx_thetas = smplx_thetas[indices_unmatched]
    # Parameters for creating the SMPL-X model
    smplx_params = dict(
        model_path=path_models,
        model_type='smplx',
        betas=smplx_betas,
        global_orient=smplx_thetas[:, :3],
        body_pose=smplx_thetas[:, 3 : 22*3],    # (B, 63)
        # remaining pose parameters are 0-initialized by default
        use_pca=False,
        flat_hand_mean=True,
        batch_size=smplx_betas.shape[0],
        gender=gender
    )
    # Conversion
    smpl_betas, smpl_thetas = _perform_conversion(
        'smpl',
        smplx_betas,
        smplx_thetas,
        smplx_params,
        output_dir,
        overwrite_previous_output,
        device,
        verbosity
    )
    smpl_thetas = _remove_pose_joint_dimension(smpl_thetas)
    # Write results to file
    if explicit_result_file is not None:
        explicit_result_file = pathlib.Path(explicit_result_file)
        output_dir = explicit_result_file.parent
        conversion_results_filename = explicit_result_file.name
    if save_conversion_results:
        _save_conversion_results(
            output_dir,
            smpl_betas,
            smpl_thetas,
            smplx_betas,
            smplx_thetas,
            gender,
            conversion_results_filename
        )
    if check_already_converted:
        # Combine with the already converted parameters we obtained earlier
        for idx, entry in enumerate(previous_results):
            if entry is not None:
                smpl_betas = np.insert(
                    smpl_betas,
                    idx,
                    entry['betas'],
                    axis=0
                )
                smpl_thetas = np.insert(
                    smpl_thetas,
                    idx,
                    entry['thetas'],
                    axis=0
                )
    return smpl_betas, smpl_thetas


def _perform_conversion(
    conv_to: str,
    betas: Union[np.ndarray, torch.Tensor],
    thetas: Union[np.ndarray, torch.Tensor],
    model_params: dict,
    output_dir: str,
    overwrite_previous_output: bool,
    device: Tuple[str, torch.device] = None,
    verbosity: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
    """Converts the given parameters to the specified model type.
    The origin model type is determined based on the given model parameters.
    Args:
        conv_to (str): Model type to which the parameters should be converted.
        betas (Union[np.ndarray, torch.tensor]): Body shape parameters of shape
            (B, 10), where B is the number of different bodies for which the
            parameters should be converted.
        thetas (Union[np.ndarray, torch.tensor]): Body pose parameters of shape
            (B, P), where P depends on the original body model. For example,
            for SMPL P=72 and for SMPL-X P=165. In general, P=3+2*Number of
            body joints. The first three entries encode the global orientation.
        model_params (dict): The parameters used for creating the original
            body model. Specifically, the original body model type is inferred
            from this dict.
        output_dir (str): The base output directory for files that will be
            created during the conversion.
        overwrite_previous_output (bool): Whether previously created
            intermediate files should be overwritten.
        device (Tuple[str, torch.device], optional): The device to use for
            computations. If not provided, decides automatically based on
            availablility. Defaults to None.
        verbosity (int, optional): Verbosity setting. Defaults to 1.
    Raises:
        ValueError: When the output directory already contains intermediate
            files and overwriting of these files is disabled.
        RuntimeError: When the os.system call to the conversion script returned
            a non-zero exit status.
    Returns:
        np.ndarray: Shape parameters converted to the specified body model
            type. Shape is (B, 10)
        np.ndarray: Pose parameters converted to the specified body model type.
            Shape is (B, N, 3), where N=J+1 with J=number of body joints of
            the target body model and 1 for the global orientation. As an
            example, for SMPL N=24 and for SMPL-X N=55
    """
    conv_to = conv_to.lower()
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    if isinstance(device, str):
        device = torch.device(device)
    conv_from = model_params['model_type']
    batch_size = model_params['batch_size']
    gender = model_params['gender']
    path_models = model_params['model_path']

    from_output_path = osp.join(output_dir, conv_from)
    to_output_path = osp.join(output_dir, conv_to)

    # Do output directories contain files?
    files = []
    if osp.exists(from_output_path):
        files.extend([f for f in os.listdir(from_output_path)])
    if osp.exists(to_output_path):
        files.extend([f for f in os.listdir(to_output_path)])
    if len(files) > 0 and not overwrite_previous_output:
        raise ValueError(
            "The provided output directory is not empty. Conversion would "
            "overwrite existing data. Provide a different directory or "
            "confirm overwriting by setting overwrite_previous_output to "
            "True")

    for dir in [from_output_path, to_output_path]:
        os.makedirs(dir, exist_ok=True)

    # Create body model
    if verbosity > 0:
        print(f"Creating {conv_from.upper()} body model... ", end='', 
              flush=True)
    time_model_creation_start = time.perf_counter()
    body_model = smplx.create(**model_params).to(device)
    model_output = body_model()
    if verbosity > 0:
        print("Finished, took "
              f"{(time.perf_counter() -time_model_creation_start):.2f} "
              "seconds")

    # Create meshes
    for idx in tqdm(range(batch_size), desc="Creating Meshes"):
        vertices = model_output.vertices[idx].detach().cpu().numpy()
        mesh = trimesh.Trimesh(vertices, body_model.faces, process=False)
        mesh_bytes = trimesh.exchange.ply.export_ply(mesh)
        with open(osp.join(from_output_path, str(idx)+'.ply'), 'wb') as file:
            file.write(mesh_bytes)

    if verbosity > 0:
        print(f"Executing the conversion script from {conv_from.upper()} to "
              f"{conv_to.upper()}, this may take a while")
    time_conversion_script_start = time.perf_counter()
    base_dir = pathlib.Path(__file__).parent.parent
    smplx_directory = osp.join(base_dir, 'smplx')
    transfer_file = osp.join(
        path_models,
        'transfer',
        f"{conv_from}2{conv_to}_deftrafo_setup.pkl"
    )
    
    
    smplx_directory = '/garmentor/smplx/'
    os.chdir(smplx_directory)
    
    # Call script
    return_value = \
        os.system(
            #f"cd {smplx_directory} && python -m transfer_model --exp-cfg "
            f"python -m transfer_model --exp-cfg "
            f"config_files/{conv_from}2{conv_to}.yaml --exp-opts "
            f"datasets.mesh_folder.data_folder={from_output_path} "
            f"body_model.gender={gender} body_model.folder={path_models} "
            f"output_folder={to_output_path} log_file={to_output_path} "
            f"deformation_transfer_path={transfer_file}"
        )
    if return_value != 0:
        raise RuntimeError("The conversion script exited with code "
                           f"{return_value}. Please check the traceback above "
                           "for the reason.")
    if verbosity > 0:
        print("Finished conversion, took "
              f"{(time.perf_counter() - time_conversion_script_start):.2f} "
              "seconds")

    # Load output format parameters from the created pickle files
    output_betas = []
    output_thetas = []
    for idx in tqdm(range(batch_size), desc="Reading Output Parameters"):
        with open(osp.join(to_output_path, f"{idx}.pkl"), 'rb') as file:
            conv_results = pickle.load(file)
        # Shape parameters
        output_betas.append(
            conv_results['betas'][0, :10].detach().cpu().numpy()
        )
        # Poses are in rotation matrix format
        poses = _batch_rot2aa(conv_results['full_pose'].squeeze())
        output_thetas.append(poses.detach().cpu().numpy())

    output_betas = np.asarray(output_betas)
    output_thetas = np.asarray(output_thetas)
    return output_betas, output_thetas


def _check_already_converted(
    betas: np.ndarray,
    thetas: np.ndarray,
    gender: str,
    provided_parameter_type: str,
    result_parameter_type: str,
    file_path: str
    ) -> List[Union[dict, None]]:
    """Checks whether the provided parameters are present in the provided
    conversion results file. If so, their counterparts are returned. Otherwise,
    None is returned.
    Args:
        betas (np.ndarray): Beta parameters of shape (B, 10)
        thetas (np.ndarray): Theta parameters of shape (B, 72) for SMPL pose
            parameters and (B, 165) for SMPL-X pose parameters.
        gender (str): Gender of the bodies. Can be "male", "female" or
            "neutral".
        provided_parameter_type (str): Type of the provided parameters, e.g.
            smpl for SMPL parameters and smplx for SMPL-X parameters.
        result_parameter_type (str): Type of parameters for which a conversion
            result should be searched.
        file_path (str): Path to the conversion result file where the
            parameters should be searched in
    Returns:
        A list that, for each element in B, contains one of the following:
            None: If the parameters are not present in the conversion result
                file
            dict: If the parameters are present in the conversion result file,
                the key "betas" will contain the converted beta parameters and
                the key "thetas" will contain the converted theta parameters
        If the provided conversion result file could not be opened, a list with
            all None values is returned.
    """
    error_return_value = [None for _ in range(betas.shape[0])]
    content = load_conversion_results(file_path)
    if content is None:
        return error_return_value
    for key in [
        f"{provided_parameter_type}_betas_{gender}",
        f"{provided_parameter_type}_thetas_{gender}",
        f"{result_parameter_type}_betas_{gender}",
        f"{result_parameter_type}_betas_{gender}"
    ]:
        if not key in content.keys():
            return error_return_value
        if content[key].dtype == 'O' and content[key] == None:
            return error_return_value

    results = []
    for idx in range(betas.shape[0]):
        from_key_betas = f"{provided_parameter_type}_betas_{gender}"
        from_key_thetas = f"{provided_parameter_type}_thetas_{gender}"
        to_key_betas = f"{result_parameter_type}_betas_{gender}"
        to_key_thetas = f"{result_parameter_type}_thetas_{gender}"
        conversion_result_betas = None
        conversion_result_thetas = None

        search_results = np.where(
            (content[from_key_betas] == betas[idx]).all(axis=1)
        )
        # Index of all converted beta parameters that match the provided beta
        # parameter
        matching_beta_indices = search_results[0].tolist()
        for beta_index in matching_beta_indices:
            if (content[from_key_thetas][beta_index] == thetas[idx]).all():
                conversion_result_betas = content[to_key_betas][beta_index]
                conversion_result_thetas = content[to_key_thetas][beta_index]
                break
        if conversion_result_betas is None:
            results.append(None)
        else:
            results.append(dict(
                betas=conversion_result_betas,
                thetas=conversion_result_thetas
            ))
    return results


def load_conversion_results(
    file_path: str
    ) -> dict:
    """Loads the results of previously done conversions.
    Args:
        file_path (str): Path to the file that contains the results which
            should be loaded.
    Returns:
        A dict with keys 'smpl_betas_<gender>', 'smpl_thetas_<gender>', 'smplx_betas_<gender>', and 'smplx_thetas_<gender>' with genders
            male, female, and neutral. The value to each key is either None or a numpy array of following shapes:
            smpl_betas and smplx_betas (B, 10), smpl_thetas (B, 72), smplx_thetas (B, 165).
            Elements in the same row for the same gender correspond to each other.
    """
    if file_path[-4:] != '.npz':
        file_path += '.npz'
    if osp.isfile(file_path):
        content = np.load(file_path, allow_pickle=True)
        # We want to always return smplx thetas with 165 values, so we have to extend 
        for key in ['smplx_thetas_male', 'smplx_thetas_female', 'smplx_thetas_neutral']:
            if content[key].dtype == 'O' and content[key] == None:
                continue
            thetas_shape = content[key].shape
            if thetas_shape[1] != 165:
                content[key] = np.append(
                    content[key],
                    np.zeros((thetas_shape[0], 165-thetas_shape[1])),
                    axis=1
                )
        return content
    else:
        return None


def split_smplx_full_body_pose(
    smplx_pose: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
               np.ndarray, np.ndarray]:
    """Splits the SMPL-X full body pose into the different sub-poses. The
    returned arrays point to the same memory as the input array. Splitting
    points based on this joint order: https://github.com/vchoutas/smplx/blob/
    5fa20519735cceda19afed0beeabd53caef711cd/smplx/joint_names.py#L17
    Args:
        smplx_pose (np.ndarray): The full-body SMPL-X pose with shape of either 
        (B, 55, 3) or (B, 165)
    Returns:
        np.ndarray: Global orientation of shape (B, 3)
        np.ndarray: Body pose of shape (B, 63)
        np.ndarray: Jaw pose of shape (B, 3)
        np.ndarray: Left eye pose of shape (B, 3)
        np.ndarray: Right eye pose of shape (B, 3)
        np.ndarray: Left hand pose of shape (B, 45)
        np.ndarray: Right hand pose of shape (B, 45)
    """
    if len(smplx_pose.shape) == 3:
        smplx_pose = _remove_pose_joint_dimension(smplx_pose)
    global_orient = smplx_pose[:, :3]
    body_pose = smplx_pose[:, 3:66]
    jaw_pose = smplx_pose[:, 66:69]
    l_eye_pose = smplx_pose[:, 69:72]
    r_eye_pose = smplx_pose[:, 72:75]
    l_hand_pose = smplx_pose[:, 75:120]
    r_hand_pose = smplx_pose[:, 120:165]
    return global_orient, body_pose, jaw_pose, l_eye_pose, r_eye_pose, \
        l_hand_pose, r_hand_pose


def _save_conversion_results(
    output_dir: str,
    smpl_betas: np.ndarray,
    smpl_thetas: np.ndarray,
    smplx_betas: np.ndarray,
    smplx_thetas: np.ndarray,
    gender: str,
    filename: str = "conv_results"
    ) -> None:
    """Writes the given SMPL / SMPL-X parameters into the specified file. If
    the file already exists, the provided values will be appended to any
    existing values.
    Args:
        output_dir (str): Path to the directory where the conversion results
            should be saved to.
        smpl_betas (np.ndarray): SMPL shape parameters of shape (B, 10)
        smpl_thetas (np.ndarray): SMPL pose parameters of shape (B, 72)
        smplx_betas (np.ndarray): SMPL-X shape parameters of shape (B, 10)
        smplx_thetas (np.ndarray): SMPL-X pose parameters of shape (B, 165)
        gender (str): Gender of the body model. Supported are male, female, and neutral
        filename (str): Name of the file into which the results should be
            written.
    """
    # Make sure dimensions are as expected
    if len(smpl_betas.shape) == 1:
        smpl_betas = smpl_betas[np.newaxis, ...]
    if len(smpl_thetas.shape) == 1:
        smpl_thetas = smpl_thetas[np.newaxis, ...]
    if len(smpl_thetas.shape) == 3:
        smpl_thetas = _remove_pose_joint_dimension(smpl_thetas)

    if len(smplx_betas.shape) == 1:
        smplx_betas = smplx_betas[np.newaxis, ...]
    if len(smplx_thetas.shape) == 1:
        smplx_thetas = smplx_thetas[np.newaxis, ...]
    if len(smplx_thetas.shape) == 3:
        smplx_thetas = _remove_pose_joint_dimension(smplx_thetas)

    # Always store full-length smplx thetas
    if smplx_thetas.shape[1] != 165:
        smplx_thetas = np.append(
            smplx_thetas,
            np.zeros((smplx_thetas.shape[0], 165-smplx_thetas.shape[1])),
            axis=1
        )

    os.makedirs(output_dir, exist_ok=True)
    if filename[-4:] != '.npz':
        filename += ".npz"
    file_path = osp.join(output_dir, filename)

    if osp.isfile(file_path):
        content = np.load(file_path, allow_pickle=True)
        smpl_betas_male = content['smpl_betas_male']
        smpl_betas_female = content['smpl_betas_female']
        smpl_betas_neutral = content['smpl_betas_neutral']

        smpl_thetas_male = content['smpl_thetas_male']
        smpl_thetas_female = content['smpl_thetas_female']
        smpl_thetas_neutral = content['smpl_thetas_neutral']

        smplx_betas_male = content['smplx_betas_male']
        smplx_betas_female = content['smplx_betas_female']
        smplx_betas_neutral = content['smplx_betas_neutral']

        smplx_thetas_male = content['smplx_thetas_male']
        smplx_thetas_female = content['smplx_thetas_female']
        smplx_thetas_neutral = content['smplx_thetas_neutral']
    else:
        smpl_betas_male = smpl_betas_female = smpl_betas_neutral = \
            smpl_thetas_male = smpl_thetas_female = smpl_thetas_neutral = \
                smplx_betas_male = smplx_betas_female = smplx_betas_neutral = \
                    smplx_thetas_male = smplx_thetas_female = smplx_thetas_neutral = np.asarray(None)

    if gender == "male":
        smpl_betas_male = smpl_betas if smpl_betas_male.dtype == 'O' and smpl_betas_male == None else np.append(smpl_betas_male, smpl_betas, axis=0)
        smpl_thetas_male = smpl_thetas if smpl_thetas_male.dtype == 'O' and smpl_thetas_male == None else np.append(smpl_thetas_male, smpl_thetas, axis=0)
        smplx_betas_male = smplx_betas if smplx_betas_male.dtype == 'O' and smplx_betas_male == None else np.append(smplx_betas_male, smplx_betas, axis=0)
        smplx_thetas_male = smplx_thetas if smplx_thetas_male.dtype == 'O' and smplx_thetas_male == None else np.append(smplx_thetas_male, smplx_thetas, axis=0)
    elif gender == "female":
        smpl_betas_female = smpl_betas if smpl_betas_female.dtype == 'O' and smpl_betas_female == None else np.append(smpl_betas_female, smpl_betas, axis=0)
        smpl_thetas_female = smpl_thetas if smpl_thetas_female.dtype == 'O' and smpl_thetas_female == None else np.append(smpl_thetas_female, smpl_thetas, axis=0)
        smplx_betas_female = smplx_betas if smplx_betas_female.dtype == 'O' and smplx_betas_female == None else np.append(smplx_betas_female, smplx_betas, axis=0)
        smplx_thetas_female = smplx_thetas if smplx_thetas_female.dtype == 'O' and smplx_thetas_female == None else np.append(smplx_thetas_female, smplx_thetas, axis=0)
    elif gender == "neutral":
        smpl_betas_neutral = smpl_betas if smpl_betas_neutral.dtype == 'O' and smpl_betas_neutral == None else np.append(smpl_betas_neutral, smpl_betas, axis=0)
        smpl_thetas_neutral = smpl_thetas if smpl_thetas_neutral.dtype == 'O' and smpl_thetas_neutral == None else np.append(smpl_thetas_neutral, smpl_thetas, axis=0)
        smplx_betas_neutral = smplx_betas if smplx_betas_neutral.dtype == 'O' and smplx_betas_neutral == None else np.append(smplx_betas_neutral, smplx_betas, axis=0)
        smplx_thetas_neutral = smplx_thetas if smplx_thetas_neutral.dtype == 'O' and smplx_thetas_neutral == None else np.append(smplx_thetas_neutral, smplx_thetas, axis=0)

    np.savez(
        file_path,
        smpl_betas_male=smpl_betas_male,
        smpl_betas_female=smpl_betas_female,
        smpl_betas_neutral=smpl_betas_neutral,
        smpl_thetas_male=smpl_thetas_male,
        smpl_thetas_female=smpl_thetas_female,
        smpl_thetas_neutral=smpl_thetas_neutral,
        smplx_betas_male=smplx_betas_male,
        smplx_betas_female=smplx_betas_female,
        smplx_betas_neutral=smplx_betas_neutral,
        smplx_thetas_male=smplx_thetas_male,
        smplx_thetas_female=smplx_thetas_female,
        smplx_thetas_neutral=smplx_thetas_neutral
    )
    return


def _remove_pose_joint_dimension(pose: np.ndarray) -> np.ndarray:
    """Removes the joint dimension from the given posa parameters
    Args:
        pose (np.ndarray): The pose parameters where the pose dimension should
            be removed from. Expected shape (B, J, 3) where B is the batch size
            and J is the number of joints.
    Returns:
        np.ndarray: The pose parameters with removed joint dimension. Shape
            (B, J*3)
    """
    if len(pose.shape) == 2:
        return pose
    return pose.reshape(pose.shape[0], -1)


def _batch_rot2aa(
    Rs: torch.Tensor, epsilon: float = 1e-7
) -> torch.Tensor:
    """
    Rs is B x 3 x 3
    Copied from smplx.transfer_model.utils.pose_utils.py
    """

    cos = 0.5 * (torch.einsum('bii->b', [Rs]) - 1)
    cos = torch.clamp(cos, -1 + epsilon, 1 - epsilon)

    theta = torch.acos(cos)

    m21 = Rs[:, 2, 1] - Rs[:, 1, 2]
    m02 = Rs[:, 0, 2] - Rs[:, 2, 0]
    m10 = Rs[:, 1, 0] - Rs[:, 0, 1]
    denom = torch.sqrt(m21 * m21 + m02 * m02 + m10 * m10 + epsilon)

    axis0 = torch.where(torch.abs(theta) < 0.00001, m21, m21 / denom)
    axis1 = torch.where(torch.abs(theta) < 0.00001, m02, m02 / denom)
    axis2 = torch.where(torch.abs(theta) < 0.00001, m10, m10 / denom)

    return theta.unsqueeze(1) * torch.stack([axis0, axis1, axis2], 1)


def _sample_parameters(
    number_samples: int,
    model_type: str,
    sample_global_orient: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
    """Samples the given number of SMPL shape and pose parameters.
    Args:
        number_samples (int): Number of shape and pose combinations to sample
        model_type (str): For which model the pose parameters should be
            sampled. Can be 'smpl' or 'smplx'.
        sample_global_orient (bool, optional): Whether the global orientation
            should also be sampled. False always returns a global orientation
            of (0, 0, 0). Defaults to False. Currenty unused
            
    Raises:
        NotImplementedError: If the model type is not 'smpl' or 'smplx'
    Returns:
        np.ndarray: The sampled SMPL shape parameters of shape (B, 10) where B
            corresponds to the given number of pairs to sample
        np.ndarray: The sampled SMPL pose parameters of shape (B, N) where N
            is dependent on the selected model type. For example, for SMPL N=
            72 and for SMPL-X N=165
    """
    if model_type.lower() not in ['smpl', 'smplx']:
        raise NotImplementedError("Model types besides 'smpl' and 'smplx' are "
                                  "currently not supported.")
    # For now: just random
    THETA_SAMPLES = {'smpl': 72, 'smplx': 165}
    betas = (np.random.rand(number_samples, 10) - 0.5) * 4
    thetas = np.zeros((number_samples, THETA_SAMPLES[model_type.lower()]))
    return betas, thetas


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Converts SMPL parameters to "
                                     "SMPL-X parameters")
    parser.add_argument(
        '-p',
        '--path-models',
        type=str,
        help="Path to the directory that contains the smpl and smplx sub-"
            "directories"
    )
    parser.add_argument('-o', '--output', type=str, required=True,
                        help="Output directory")
    parser.add_argument('-n', '--number-samples', type=int, required=True,
                        help="Number of different shape-pose parameters that "
                        "should be sampled.")
    parser.add_argument('-c', '--conversion', type=str, required=True,
                        help="Which conversion to perform. Either 'smpl2smplx'"
                        " or 'smplx2smpl'")
    parser.add_argument('-b', '--batch-size', type=int, default=0,
                        help="Batch size. Provide 0 to set batch size equal "
                        "to number of samples. Defaults to 0.")
    parser.add_argument('--overwrite', action='store_true', help="Allow "
                        "overwriting of previously created meshes. Has no "
                        "influence on the stored conversion results.")
    parser.add_argument('-r', '--result-file', type=str, default=None,
                        help="When specified, conversion results will be saved"
                        " into this file instead of the file at the default "
                        "location.")
    args = parser.parse_args()

    if args.batch_size <= 0:
        args.batch_size = args.number_samples

    number_batches = ceil(args.number_samples / args.batch_size)
    for batch_idx in tqdm(range(number_batches), desc="Batch"):
        parameters_to_sample = args.batch_size
        if batch_idx == (number_batches -1):
            parameters_to_sample = \
                args.number_samples - (batch_idx * args.batch_size)
        betas, thetas = _sample_parameters(
            parameters_to_sample,
            args.conversion.lower().split('2')[0]
        )

        if args.conversion.lower() == 'smpl2smplx':
            smpl2smplx(
                args.path_models,
                args.output,
                betas,
                thetas,
                overwrite_previous_output=args.overwrite,
                explicit_result_file=args.result_file
            )
        elif args.conversion.lower() == 'smplx2smpl':
            smplx2smpl(
                args.path_models,
                args.output,
                betas,
                thetas,
                overwrite_previous_output=args.overwrite,
                explicit_result_file=args.result_file
            )
        else:
            raise NotImplementedError("Conversions besides smpl2smplx and "
                                      "smplx2smpl are currently not "
                                      "supported.")
        # At this point, always overwrite the previous output
        args.overwrite = True

    print("Conversion finished. Results were stored to "
          f"{osp.join(args.output, 'conv_results.npz')}")
