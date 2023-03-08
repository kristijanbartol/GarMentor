from typing import Tuple
import torch
import numpy as np

from utils.rigid_transform_utils import pose_to_rotmat


def to_torch(
        *arrays: Tuple[np.ndarray, ...]
    ) -> Tuple[torch.Tensor, ...]:
    '''Convert from np.arrays to torch.Tensors using torch.as_tensor.'''
    tensors = []
    for array in arrays:
        if type(array) != torch.Tensor:
            array = torch.as_tensor(array)
        tensors.append(array)
    return tensors


def to_numpy(
        *tensors: Tuple[torch.Tensor, ...]
    ) -> Tuple[np.ndarray, ...]:
    '''Convert from torch.Tensors to np.ndarrays.'''
    arrays = []
    for tensor in tensors:
        if type(tensor) != torch.Tensor:
            tensor = tensor.cpu().detach().numpy()[0]
        arrays.append(tensor)
    return arrays


def to_float_arrays(
        *arrays: Tuple[np.ndarray, ...]
    ) -> Tuple[np.ndarray]:
    '''Convert np.ndtypes to np.float32.'''
    new_arrays = []
    for array in arrays:
        new_arrays.append(array.astype(np.float32))
    return new_arrays


def expand_dims_arrays(
        *arrays: Tuple[np.ndarray, ...]
    ) -> Tuple[np.ndarray, ...]:
    '''Unsqueeze np.arrays to have the first dummy dim.'''
    new_arrays = []
    for array in arrays:
        new_arrays.append(np.expand_dims(array, axis=0))
    return new_arrays


def unsqueeze_tensors(
        *tensors: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, ...]:
    '''Unsqueeze torch.Tensors to have the first dummy dim.'''
    new_tensors = []
    for tensor in tensors:
        new_tensors.append(torch.unsqueeze(tensor, dim=0))
    return new_tensors


def to_smpl_model_params(
        pose: np.ndarray,
        shape: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    '''Prepare pose- and shape-derived parameters for SMPL run.'''
    pose, shape = to_float_arrays(pose, shape)
    pose, shape = expand_dims_arrays(pose, shape)
    pose, shape = to_torch(pose, shape)
    glob_rotmat, pose_rotmat = pose_to_rotmat(pose)
    return glob_rotmat, pose_rotmat, shape
