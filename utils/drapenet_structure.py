from typing import Optional, Union
import torch
import numpy as np
from dataclasses import dataclass, fields


# NOTE: The same as SMPL4GarmentOutput.
# TODO: Have to resolve this redundancy better.

@dataclass
class DrapeNetStructure:

    # Mandatory values.
    garment_verts: Union[torch.tensor, np.ndarray]
    garment_faces: Union[torch.tensor, np.ndarray]
    body_verts: Union[torch.tensor, np.ndarray]
    body_faces: Union[torch.tensor, np.ndarray]
    joints: Union[torch.tensor, np.ndarray]

    # Optional values.
    betas: Optional[Union[torch.tensor, np.ndarray]] = None
    body_pose: Optional[Union[torch.tensor, np.ndarray]] = None
    full_pose: Optional[Union[torch.tensor, np.ndarray]] = None
    global_orient: Optional[Union[torch.tensor, np.ndarray]] = None
    transl: Optional[Union[torch.tensor, np.ndarray]] = None

    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __iter__(self):
        return self.keys()

    def keys(self):
        keys = [t.name for t in fields(self)]
        return iter(keys)

    def values(self):
        values = [getattr(self, t.name) for t in fields(self)]
        return iter(values)

    def items(self):
        data = [(t.name, getattr(self, t.name)) for t in fields(self)]
        return iter(data)
