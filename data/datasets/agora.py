from typing import Dict, List
import numpy as np
from tqdm import tqdm
import os
import torch
from torch.utils.data import Dataset
from math import floor, ceil
import imageio

import configs.paths as paths
from configs.const import (
    TRAIN,
    VALID,
    AGORA_DATASET_NAME
)
from data.datasets.common import Values
    

class AgoraDataset(Dataset):

    """
    An instance of train dataset specific to SURREAL dataset.
    """

    DATASET_NAME = AGORA_DATASET_NAME

    def __init__(self,
                 gender: str,
                 data_split: str,
                 train_val_ratio: float
        ) -> None:
        """
        Initialize paths, load samples's values, and segmentation maps.
        """

        super().__init__()
        print(f'Loading {data_split} data...')
        
        gender_dirpath = os.path.join(
            paths.DATA_ROOT_DIR,
            self.DATASET_NAME,
            gender
        )
        values_fpath = os.path.join(
            gender_dirpath,
            paths.VALUES_FNAME
        )
        rgb_dirpath = os.path.join(
            gender_dirpath,
            paths.RGB_DIR
        )
        
        data_split_slice = self._get_slices(
            values_fpath=values_fpath,
            data_split=data_split,
            train_val_ratio=train_val_ratio
        )
        self.values = self._load_values(
            gender_dirpath=gender_dirpath,
            data_slice=data_split_slice
        )
        self.rgb_img_paths = self._get_rgb_img_paths(
            gender_dirpath=gender_dirpath,
            data_slice=data_split_slice
        )

    @staticmethod
    def _get_slices(
            values_fpath: str,
            data_split: str,
            train_val_ratio: float
        ) -> slice:
        """
        Calculate slice for given data split (train/valid).
        """
        values = np.load(
            values_fpath,
            allow_pickle=True
        ).item()
        num_samples = values['poses'].shape[0]
        if data_split == TRAIN:
            data_slice = slice(0, floor(num_samples * train_val_ratio))
        elif data_split == VALID:
            data_slice = slice(ceil(num_samples * train_val_ratio), num_samples)
        else:
            raise Exception('Data split should be either train or val!')
        return data_slice

    @staticmethod
    def _load_values(
            gender_dirpath: str,
            data_slice: slice
        ) -> Values:
        """
        Init and fill out the Values object based on .npy and a data slice.
        """
        values = Values()
        values_fpath = os.path.join(
            gender_dirpath,
            paths.VALUES_FNAME
        )
        values.load(
            np_path=values_fpath, 
            data_slice=data_slice
        )
        values.to_numpy()
        return values
    
    @staticmethod
    def _get_rgb_img_paths(
            gender_dirpath: str,
            data_slice: slice
        ) -> List[str]:
        """
        Get RGB paths, given gender dir and a data slice (for train/valid).
        """
        rgb_img_paths = []
        rgb_dirpath = os.path.join(
            gender_dirpath, 
            paths.RGB_DIR
        )
        rgb_files = sorted(os.listdir(rgb_dirpath))[data_slice]
        for f in tqdm(rgb_files):
            rgb_img_path = os.path.join(rgb_dirpath, f)
            rgb_img_paths.append(rgb_img_path)
        return rgb_img_paths

    def __len__(self) -> int:
        """
        Get dataset length (used by DataLoader).
        """
        return len(self.values)

    @staticmethod
    def _to_tensor(value: np.ndarray, 
                   type: type = np.float32) -> torch.Tensor:
        """
        To torch Tensor from NumPy, given the type.
        """
        return torch.from_numpy(value.astype(type)) # type: ignore

    def __getitem__(self, idx: int) -> Dict:
        """
        Get the sample based on index, which can be a list of indices.
        """
        rgb_img = imageio.imread(self.rgb_img_paths[idx]).transpose(2, 0, 1)

        return {
            'pose': self._to_tensor(self.values.poses[idx]),
            'shape': self._to_tensor(self.values.shapes[idx]),
            'style_vector': self._to_tensor(self.values.style_vectors[idx]),
            'garment_labels': self._to_tensor(self.values.garment_labelss[idx]),
            'joints_3d': self._to_tensor(self.values.joints_3ds[idx]),
            'joints_2d': self._to_tensor(self.values.joints_2ds[idx]),
            'cam_t': None,
            'bbox': self._to_tensor(self.values.bboxs[idx]),
            'rgb_img': self._to_tensor(rgb_img, type=np.uint8),
            'seg_maps': None,
            'background': None
        }
