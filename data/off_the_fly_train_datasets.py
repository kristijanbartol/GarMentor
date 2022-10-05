import numpy as np
import glob
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, fields
import os
import torch
import cv2
from torch.utils.data import Dataset

from data_pregeneration import SurrealDataPreGenerator, DataPreGenerator


@dataclass
class Sample:

    '''A dataclass used to conveniently create batch sample.'''

    pose: torch.Tensor                      # (B, 72)
    shape: torch.Tensor                     # (B, 10)
    style_vector: torch.Tensor              # (B, 4, 10)
    cam_t: torch.Tensor                     # (B, 3)
    joints: torch.Tensor                    # (B, 17, 3)
    garment_labels_vector: torch.Tensor     # (B, 5)

    seg_maps: torch.Tensor                      # (B, 4, WH, WH)
    background: Optional[torch.Tensor] = None   # (B, C, WH, WH)
    rgb_img: Optional[torch.Tensor] = None      # (B, C, WH, WH)
    texture: Optional[torch.Tensor] = None      # (B, ?)

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


class TrainDataset(Dataset):

    ''' Train dataset abstract class containing only common constants.

        Folder hierarchy of the stored data.
        <DATA_ROOT_DIR>
            {dataset_name}/
                {gender}/
                    <PARAMS_FNAME>
                    <IMG_DIR>/
                        <IMG_NAME_1>
                        ...
                    <SEG_MAP_DIR>/
                        <SEG_MAP_1_1>
                        ...
    '''
    DATA_ROOT_DIR = DataPreGenerator.DATA_ROOT_DIR
    IMG_DIR = 'rgb/'
    SEG_MAPS_DIR = 'segmentations/'

    IMG_NAME_TEMPLATE = DataPreGenerator.IMG_NAME_TEMPLATE
    SEG_MAPS_NAME_TEMPLATE = DataPreGenerator.SEG_MAPS_NAME_TEMPLATE
    VALUES_FNAME = DataPreGenerator.VALUES_FNAME


class SurrealTrainDataset(TrainDataset):

    '''An instance of train dataset specific to SURREAL dataset.'''

    DATASET_NAME = SurrealDataPreGenerator.DATASET_NAME

    def __init__(self,
                 gender: str,
                 backgrounds_dir_path: str,
                 img_wh: int = 256):
        '''Initialize paths, load samples's values, and segmentation maps.'''

        super().__init__()

        self.dataset_dir = os.path.join(
            self.DATA_ROOT_DIR,
            self.DATASET_NAME,
            gender
        )
        values_fpath = os.path.join(
            self.dataset_dir, self.VALUES_FNAME)

        self.values: Dict[np.ndarray] = np.load(values_fpath)
        self.all_seg_maps: np.ndarray[np.bool] = self._load_all_seg_maps()

        self.backgrounds_paths = sorted([os.path.join(backgrounds_dir_path, f)
                                         for f in os.listdir(backgrounds_dir_path)
                                         if f.endswith('.jpg')])
        self.img_wh = img_wh

    def _load_all_seg_maps(self) -> np.ndarray[np.bool]:
        '''Load all segmentation maps in proper order.'''

        seg_maps_dir = os.path.join(self.dataset_dir, self.SEG_MAPS_DIR)
        all_seg_maps = []
        for f in sorted(os.listdir(seg_maps_dir)):
            all_seg_maps.append(np.load(f))
        return np.array(all_seg_maps, dtype=np.bool)

    def __len__(self) -> int:
        '''Get dataset length (used by DataLoader).'''

        return self.values['poses'].shape[0]

    def _load_background(self, num_samples: int) -> np.ndarray:
        '''Load random backgrounds. Adapted from the original HierProb3D code.'''

        bg_samples = []
        for _ in range(num_samples):
            bg_idx = torch.randint(low=0, high=len(self.backgrounds_paths), 
                size=(1,)).item()
            bg_path = self.backgrounds_paths[bg_idx]
            background = cv2.cvtColor(cv2.imread(bg_path), cv2.COLOR_BGR2RGB)
            background = cv2.resize(background, (self.img_wh, self.img_wh), 
                interpolation=cv2.INTER_LINEAR)
            background = background.transpose(2, 0, 1)
            bg_samples.append(background)
        bg_samples = np.stack(bg_samples, axis=0).squeeze()
        return torch.from_numpy(bg_samples / 255.).float()

    def _to_tensor(self, 
                   value: np.ndarray, 
                   type: type = np.float32) -> torch.Tensor:
        '''To torch Tensor from NumPy, given the type.'''
        return torch.from_numpy(value.astype(type))

    def __getitem__(self, index: Union[torch.Tensor, List]) -> Sample:
        '''Get the sample based on index, which can be a list of indices.'''
        
        if torch.is_tensor(index):
            index = index.tolist()
        if isinstance(index, list):
            num_samples = len(index)
        else:
            num_samples = 1

        return Sample(
            pose=self._to_tensor(self.values['poses'][index]),
            shape=self._to_tensor(self.values['shapes'][index]),
            style_vector=self._to_tensor(self.values['style_vectors'][index]),
            cam_t=self._to_tensor(self.values['cam_ts'][index]),
            joints=self._to_tensor(self.values['jointss'][index]),
            garment_labels_vector=self._to_tensor(self.values['garment_labels_vectors'][index]),
            seg_maps=self._to_tensor(self.all_seg_maps[index], np.bool),
            background=self._load_background(num_samples)
        )
