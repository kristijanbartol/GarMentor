from typing import Dict, List, Tuple
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
    VALID
)
from data.prepare.surreal.tn import SurrealDataGenerator
from data.datasets.common import (
    get_background_paths,
    load_background,
    Values
)
from utils.garment_classes import GarmentClasses
    

class SurrealDataset(Dataset):

    """
    An instance of train dataset specific to SURREAL dataset.
    """

    DATASET_NAME = SurrealDataGenerator.DATASET_NAME

    def __init__(self,
                 gender: str,
                 data_split: str,
                 train_val_ratio: float,
                 backgrounds_dir_path: str,
                 img_wh: int = 256):
        """
        Initialize paths, load samples's values, and segmentation maps.
        """
        super().__init__()
        print(f'Loading {data_split} data...')
        
        dataset_gender_dir = os.path.join(
            paths.DATA_ROOT_DIR,
            self.DATASET_NAME,
            gender)

        self.garment_class_list, garment_dirnames = self._get_all_garment_pairs(
            dataset_gender_dir=dataset_gender_dir
        )
        garment_dirpaths = [
            os.path.join(dataset_gender_dir, x) for x in garment_dirnames]
        
        data_split_slices_list = self._get_slices(
            garment_dirpaths=garment_dirpaths,
            values_fname=paths.VALUES_FNAME,
            data_split=data_split,
            train_val_ratio=train_val_ratio
        )
        
        self.values = self._load_values(
            garment_dirpaths,
            slice_list=data_split_slices_list
        )
        self.seg_maps_paths = self._get_seg_maps_paths(
            garment_dirpaths=garment_dirpaths, 
            seg_maps_dirname=paths.SEG_MAPS_DIR,
            slice_list=data_split_slices_list
        )
        self.rgb_img_paths = self._get_rgb_img_paths(
            garment_dirpaths=garment_dirpaths,
            rgb_imgs_dirname=paths.RGB_DIR,
            slice_list=data_split_slices_list
        )
        self.backgrounds_paths = get_background_paths(
            backgrounds_dir_path=backgrounds_dir_path,
            num_backgrounds=10000
        )
        self.img_wh = img_wh
        
    @staticmethod
    def _get_all_garment_pairs(
            dataset_gender_dir
        ) -> Tuple[List[GarmentClasses], List[str]]:
        """
        Collects all the garment class pairs based on directory names.

        This is specific to SURREAL-like datasets because each garment
        combination is generated separately and it is simpler to determine
        the combination this way.
        """
        garment_class_list, garment_dirnames = [], []
        for garment_dirname in os.listdir(dataset_gender_dir):
            garment_dirnames.append(garment_dirname)
            garment_class_pair = garment_dirname.split('+')
            garment_class_list.append(GarmentClasses(
                upper_class=garment_class_pair[0],
                lower_class=garment_class_pair[1]
            ))
        print(f'Found dirnames: {garment_dirnames}')
        return garment_class_list, garment_dirnames
    
    @staticmethod
    def _get_slices(
            garment_dirpaths: List[str],
            values_fname: str,
            data_split: str,
            train_val_ratio: float
        ) -> List[slice]:
        """
        Calculates data slices for each garment combination based on the number
        of samples for the particular garment combination.

        Train slice is between [0, num_samples * train_val_ratio], while the
        validation slice is between [num_samples * train_val_ratio, NUM].
        """
        slices_per_garment = []
        for garment_dirpath in garment_dirpaths:
            values = np.load(
                os.path.join(garment_dirpath, values_fname),
                allow_pickle=True
            ).item()
            num_samples = values['poses'].shape[0]
            if data_split == TRAIN:
                _slice = slice(0, floor(num_samples * train_val_ratio))
            elif data_split == VALID:
                _slice = slice(ceil(num_samples * train_val_ratio), num_samples)
            else:
                raise Exception('Data split should be either train or val!')
            slices_per_garment.append(_slice)
        return slices_per_garment
            
    @staticmethod
    def _load_values(
            garment_dirpaths: List[str],
            slice_list: List[slice]
        ) -> Values:
        """
        Load values from the disk.

        Specific for SURREAL-like datasets, values.npy is stored in relevant
        directory based on paths to particular garment combination directories.
        """
        values = Values()
        for garment_idx, garment_dirpath in enumerate(garment_dirpaths):
            print(f'Loading values ({garment_dirpath})...')
            values_fpath = os.path.join(garment_dirpath, 'values.npy')
            values.load(values_fpath, slice_list[garment_idx])
        values.to_numpy()
        return values

    @staticmethod
    def _get_seg_maps_paths(
            garment_dirpaths: List[str], 
            seg_maps_dirname: str,
            slice_list: List[slice]
        ) -> List[str]:
        """
        Load all segmentation maps for the specified slices.

        The proper order is determined first by the order of garment dirs and
        then by the file number (alphabetic). The slices are based on the 
        dataset type (train/valid).
        """
        seg_maps_paths = []
        for garment_idx, garment_dirpath in enumerate(garment_dirpaths):
            print(f'Loading segmaps paths ({garment_dirpath})...')
            seg_maps_dir = os.path.join(garment_dirpath, seg_maps_dirname)
            seg_files = sorted(
                os.listdir(seg_maps_dir))[slice_list[garment_idx]]
            for f in tqdm(seg_files):
                seg_maps_path = os.path.join(seg_maps_dir, f)
                seg_maps_paths.append(seg_maps_path)
        return seg_maps_paths
    
    @staticmethod
    def _get_rgb_img_paths(
            garment_dirpaths: List[str],
            rgb_imgs_dirname: str,
            slice_list: List[slice]
        ) -> List[str]:
        """
        Get the paths to all RGB images for the specified slices.

        Note that the slices are based on the dataset type (train/valid).
        """
        rgb_img_paths = []
        for garment_idx, garment_dirpath in enumerate(garment_dirpaths):
            print(f'Loading RGB image paths ({garment_dirpath})...')
            rgb_imgs_dir = os.path.join(garment_dirpath, rgb_imgs_dirname)
            rgb_files = sorted(os.listdir(rgb_imgs_dir))[slice_list[garment_idx]]
            for f in tqdm(rgb_files):
                rgb_img_path = os.path.join(rgb_imgs_dir, f)
                rgb_img_paths.append(rgb_img_path)
        return rgb_img_paths

    def __len__(self) -> int:
        """
        Get dataset length (used by DataLoader).
        """
        return len(self.values)

    def _load_background(self) -> np.ndarray:
        """
        Protected method for loading random background.
        """
        return load_background(
            self.backgrounds_paths,
            self.img_wh
        )

    @staticmethod
    def _to_tensor(
            value: np.ndarray, 
            type: type = np.float32
        ) -> torch.Tensor:
        """
        To torch Tensor from NumPy, given the type.
        """
        return torch.from_numpy(value.astype(type)) # type: ignore

    def __getitem__(
            self, 
            idx: int
        ) -> Dict:
        """
        Get the sample based on index, which can be a list of indices.
        """
        seg_maps = np.load(self.seg_maps_paths[idx])['seg_maps']
        rgb_img = imageio.imread(self.rgb_img_paths[idx]).transpose(2, 0, 1)

        return {
            'pose': self._to_tensor(self.values.poses[idx]),
            'shape': self._to_tensor(self.values.shapes[idx]),
            'style_vector': self._to_tensor(self.values.style_vectors[idx]),
            'garment_labels': self._to_tensor(self.values.garment_labelss[idx]),
            'joints_3d': self._to_tensor(self.values.joints_3ds[idx]),
            'joints_2d': self._to_tensor(self.values.joints_2ds[idx]),
            'cam_t': self._to_tensor(self.values.cam_ts[idx]),
            'bbox': self._to_tensor(self.values.bboxs[idx]),
            'rgb_img': self._to_tensor(rgb_img, type=np.uint8),
            'seg_maps': self._to_tensor(seg_maps, type=bool),
            'background': self._load_background()
        }
