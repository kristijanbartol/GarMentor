from typing import List, Optional, Union
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass, fields
import os
import torch
import cv2
import imageio
from torch.utils.data import Dataset

from data.pregenerate_data import SurrealDataPreGenerator, DataPreGenerator
from utils.garment_classes import GarmentClasses


@dataclass
class Sample:

    '''A dataclass used to conveniently create batch sample.'''

    pose: torch.Tensor                      # (B, 72)
    shape: torch.Tensor                     # (B, 10)
    style_vector: torch.Tensor              # (B, 4, 10)
    cam_t: torch.Tensor                     # (B, 3)
    joints: torch.Tensor                    # (B, 17, 3)
    garment_labelss: torch.Tensor           # (B, 4)

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
    
    
class Values:
    
    def __init__(self):
        self.poses = []
        self.shapes = []
        self.style_vectors = []
        self.cam_ts = []
        self.jointss = []
        self.garment_labelss = []
    
    def load(self, 
             np_path: str, 
             split_slice: slice = slice(None)
             ) -> None:
        data = np.load(np_path, allow_pickle=True).item()
        
        self.poses.append(data['poses'][split_slice])
        self.shapes.append(data['shapes'][split_slice])
        self.style_vectors.append(data['style_vectors'][split_slice])
        self.cam_ts.append(data['cam_ts'][split_slice])
        self.jointss.append(data['jointss'][split_slice])
        self.garment_labelss.append(data['garment_labelss'][split_slice])
        
    def numpy(self) -> None:
        self.poses = np.concatenate(self.poses, axis=0)
        self.shapes = np.concatenate(self.shapes, axis=0)
        self.style_vectors = np.concatenate(self.style_vectors, axis=0)
        self.cam_ts = np.concatenate(self.cam_ts, axis=0)
        self.jointss = np.concatenate(self.jointss, axis=0)
        self.garment_labelss = np.concatenate(self.garment_labelss, axis=0)
    
    def __len__(self):
        return self.poses.shape[0]


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
    IMG_DIRNAME = 'rgb/'
    SEG_MAPS_DIRNAME = 'segmentations/'

    IMG_NAME_TEMPLATE = DataPreGenerator.IMG_NAME_TEMPLATE
    SEG_MAPS_NAME_TEMPLATE = DataPreGenerator.SEG_MAPS_NAME_TEMPLATE
    VALUES_FNAME = DataPreGenerator.VALUES_FNAME


TRAIN = 'train'
VALID = 'valid'


class SurrealTrainDataset(TrainDataset):

    '''An instance of train dataset specific to SURREAL dataset.'''

    DATASET_NAME = SurrealDataPreGenerator.DATASET_NAME

    def __init__(self,
                 gender: str,
                 data_split: str,
                 train_val_ratio: float,
                 backgrounds_dir_path: str,
                 img_wh: int = 256):
        '''Initialize paths, load samples's values, and segmentation maps.'''

        super().__init__()
        print(f'Loading {data_split} data...')
        
        dataset_gender_dir = os.path.join(
            self.DATA_ROOT_DIR,
            self.DATASET_NAME,
            gender)

        self.garment_class_list, garment_dirnames = self._get_all_garment_pairs(
            dataset_gender_dir=dataset_gender_dir
        )
        garment_dirpaths = [
            os.path.join(dataset_gender_dir, x) for x in garment_dirnames]
        
        data_split_slices_list = self._get_slices(
            garment_dirpaths=garment_dirpaths,
            data_split=data_split,
            train_val_ratio=train_val_ratio
        )
        
        self.values = self._load_values(
            garment_dirpaths,
            data_split_slices_list=data_split_slices_list
        )
        self.seg_maps = self._load_seg_maps(
            garment_dirpaths=garment_dirpaths, 
            seg_maps_dir=self.SEG_MAPS_DIRNAME,
            data_split_slices_list=data_split_slices_list
        )
        self.rgb_imgs = self._load_rgb_imgs(
            garment_dirpaths=garment_dirpaths,
            rgb_imgs_dir=self.IMG_DIRNAME,
            data_split_slices_list=data_split_slices_list
        )
        self.backgrounds_paths = self._load_backgrounds(
            backgrounds_dir_path=backgrounds_dir_path,
            num_backgrounds=10000
        )
        self.img_wh = img_wh
        
    @staticmethod
    def _get_all_garment_pairs(dataset_gender_dir) -> List[GarmentClasses]:
        garment_class_list, garment_dirnames = [], []
        for garment_dirname in os.listdir(dataset_gender_dir):
            garment_dirnames.append(garment_dirname)
            garment_class_pair = garment_dirname.split('+')
            garment_class_list.append(GarmentClasses(
                upper_class=garment_class_pair[0],
                lower_class=garment_class_pair[1]
            ))
        return garment_class_list, garment_dirnames
    
    @staticmethod
    def _get_slices(garment_dirpaths: List[str],
                    data_split: str,
                    train_val_ratio: float
                    ) -> List[int]:
        slices_per_garment = []
        for garment_dirpath in garment_dirpaths:
            values = np.load(
                os.path.join(garment_dirpath, 'values.npy'),
                allow_pickle=True
            ).item()
            num_samples = values['poses'].shape[0]
            if data_split == TRAIN:
                _slice = slice(0, num_samples * train_val_ratio)
            elif data_split == VALID:
                _slice = slice(num_samples * train_val_ratio, -1)
            else:
                raise Exception('Data split should be either train or val!')
            slices_per_garment.append(_slice)
        return slices_per_garment
            
    @staticmethod
    def _load_values(garment_dirpaths: List[str],
                     slice_list: List[slice]) -> Values:
        print('Loading values...')
        values = Values()
        for garment_idx, garment_dirpath in enumerate(garment_dirpaths):
            values_fpath = os.path.join(garment_dirpath, 'values.npy')
            values.load(values_fpath, slice_list[garment_idx])
        values.numpy()
        return values

    @staticmethod
    def _load_seg_maps(garment_dirpaths: List[str], 
                       seg_maps_dir: str,
                       slice_list: List[slice]
                       ) -> np.ndarray:
        '''Load all segmentation maps in proper order.'''

        print('Loading segmaps...')
        seg_mapss = []
        for garment_idx, garment_dirpath in enumerate(garment_dirpaths):
            seg_maps_dir = os.path.join(garment_dirpath, seg_maps_dir)
            seg_files = sorted(
                os.listdir(seg_maps_dir))[slice_list[garment_idx]]
            for f in tqdm(seg_files):
                seg_maps_path = os.path.join(seg_maps_dir, f)
                seg_mapss.append(np.load(seg_maps_path, dtype=bool))
        return np.concatenate(seg_mapss, axis=0)
    
    @staticmethod
    def _load_rgb_imgs(garment_dirpaths: List[str],
                       rgb_imgs_dir: str,
                       slice_list: List[slice]
                       ) -> np.ndarray:
        print('Loading RGB images...')
        rgb_imgs = []
        for garment_idx, garment_dirpath in enumerate(garment_dirpaths):
            rgb_imgs_dir = os.path.join(garment_dirpath, rgb_imgs_dir)
            rgb_files = sorted(os.listdir(rgb_imgs_dir))[slice_list[garment_idx]]
            for f in tqdm(rgb_files):
                rgb_img_path = os.path.join(rgb_imgs_dir, f)
                rgb_imgs.append(imageio.imread(rgb_img_path))
        return np.concatenate(rgb_imgs, axis=0)
    
    @staticmethod
    def _load_backgrounds(backgrounds_dir_path: str, 
                          num_backgrounds: int = -1) -> List[str]:
        print('Loading background paths...')
        backgrounds_paths = []
        for f in tqdm(sorted(os.listdir(backgrounds_dir_path)[:num_backgrounds])):
            if f.endswith('.jpg'):
                backgrounds_paths.append(
                    os.path.join(backgrounds_dir_path, f)
                )
        return backgrounds_paths

    def __len__(self) -> int:
        '''Get dataset length (used by DataLoader).'''

        return len(self.values)

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

    @staticmethod
    def _to_tensor(value: np.ndarray, 
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
            garment_labels_vector=self._to_tensor(self.values['garment_labelss'][index]),
            seg_maps=self._to_tensor(self.seg_maps[index], np.bool),
            background=self._load_background(num_samples)
        )
