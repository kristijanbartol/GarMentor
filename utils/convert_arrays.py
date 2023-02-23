from typing import List
import torch
import numpy as np


def to_tensors(arrays: List[np.ndarray]) -> List[torch.Tensor]:
    tensors = []
    for array in arrays:
        if type(array) != torch.Tensor:
            array = torch.as_tensor(array)
        tensors.append(array)
    return tensors
