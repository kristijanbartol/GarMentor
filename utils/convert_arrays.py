from typing import List
import torch
import numpy as np


def to_tensors(arrays: List[np.ndarray]) -> List[torch.Tensor]:
    tensors = []
    for array in arrays:
        tensors.append(torch.as_tensor(array))
    return tensors


def to_arrays(tensors: List[torch.Tensor]) -> List[np.ndarray]:
    arrays = []
    for tensor in tensors:
        arrays.append(tensor.detach().cpu().numpy())
    return arrays
