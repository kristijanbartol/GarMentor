from typing import List, Union, Tuple
import torch
import numpy as np


def to_tensors(arrays: List[np.ndarray]) -> List[torch.Tensor]:
    tensors = []
    for array in arrays:
        if type(array) != torch.Tensor:
            array = torch.as_tensor(array)
        tensors.append(array)
    return tensors


def to_arrays(tensors: List[torch.Tensor]) -> List[np.ndarray]:
    arrays = []
    for tensor in tensors:
        if type(tensor) != torch.ndarray:
            tensor = tensor.detach().cpu().numpy()
        arrays.append(tensor)
    return arrays


def verify_arrays(arrays: Union[List[np.ndarray], List[torch.Tensor]]
            ) -> Tuple[bool, torch.Tensor]:
    are_numpy = False
    if type(arrays[0]) == np.ndarray:
        are_numpy = True
        tensors = to_tensors(
            arrays=arrays
        )
    return are_numpy, tensors
