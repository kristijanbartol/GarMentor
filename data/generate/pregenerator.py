from typing import List, Tuple, Optional
from dataclasses import dataclass, fields
import numpy as np
import torch
import os
from PIL import Image
import sys
import argparse
from random import randrange

_module_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(_module_dir))

from configs import paths
from configs.poseMF_shapeGaussian_net_config import get_cfg_defaults
from models.parametric_model import ParametricModel
from rendering.clothed import ClothedRenderer
from utils.augmentation.cam_augmentation import augment_cam_t_numpy
from utils.augmentation.smpl_augmentation import (
    normal_sample_shape_numpy,
    normal_sample_style_numpy
)
from utils.garment_classes import GarmentClasses


@dataclass
class PreGeneratedSampleValues:

    '''A standard dataclass used to handle pregenerated sample values.'''

    pose: np.ndarray                    # (72,)
    shape: np.ndarray                   # (10,)
    style_vector: np.ndarray            # (4, 10)
    garment_labels: np.ndarray          # (4,)
    cam_t: np.ndarray                   # (3,)
    joints: np.ndarray                  # (17, 3)

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


class PreGeneratedValuesArray():

    '''A class used to keep track of an array of pregenerated sampled values.'''

    def __init__(self, samples_dict: dict = None):
        if samples_dict is not None:
            self._samples_dict = {k: list(v) for k, v in samples_dict.items()}
            self.keys = samples_dict.keys()
        else:
            self._samples_dict = {}
            self.keys = []

    def _set_dict_keys(self, sample_dict_keys: List[str]) -> None:
        '''Add 's' to key name specify plural.'''
        self.keys = [x + 's' for x in sample_dict_keys]

    def append(self, values: PreGeneratedSampleValues) -> None:
        '''Mimics adding to list of values to an array. Saves to latent dict.'''
        if not self._samples_dict:
            self._set_dict_keys(values.keys())
            for k in self.keys:
                self._samples_dict[k] = []
        for ks, k in zip(self.keys, values.keys()):
            self._samples_dict[ks].append(values[k])

    def get(self) -> dict:
        '''Get the dictionary with all np.ndarray items.'''
        return_dict = {k: None for k, _ in self._samples_dict.items()}
        for ks in self.keys:
            return_dict[ks] = np.array(self._samples_dict[ks])
        return return_dict


class DataPreGenerator(object):

    ''' An abstract class which contains useful objects for data pregeneration.

        The paths and template to files and folders where data will be stored.
        /data/garmentor/
            {dataset_name}/
                {gender}/
                    {garment_class_pair}/
                        values.npy
                        rgb/
                            {idx:5d}.png
                        segmentations/
                            {idx:5d}_{garment_class}.png
    '''

    DATA_ROOT_DIR = '/data/garmentor/'
    IMG_DIR = 'rgb/'
    SEG_MAPS_DIR = 'segmentations/'

    IMG_NAME_TEMPLATE = '{idx:05d}.png'
    SEG_MAPS_NAME_TEMPLATE = '{idx:05d}.npz'
    VALUES_FNAME = 'values.npy'

    def __init__(self):
        self.dataset_path_template = os.path.join(
            self.DATA_ROOT_DIR,
            '{dataset_name}',
            '{gender}',
            '{upper_garment_class}+{lower_garment_class}'
        )
        self.cfg = get_cfg_defaults()
        self._init_useful_arrays()
        self.poses = self._load_poses()
        self.num_poses = self.poses.shape[0]

    def _load_poses(self) -> np.ndarray:
        '''Load poses. Adapted from the original HierProb3D code.'''
        data = np.load(paths.TRAIN_POSES_PATH)
        fnames = data['fnames']
        poses = data['poses']
        indices = [i for i, x in enumerate(fnames)
                    if (x.startswith('h36m') or x.startswith('up3d') or x.startswith('3dpw'))]
        return np.stack([poses[i] for i in indices], axis=0)
    
    def _init_useful_arrays(self) -> None:
        '''These useful arrays are used to randomly sample data and transform points.'''
        self.delta_betas_std_vector = np.ones(
            self.cfg.MODEL.NUM_SMPL_BETAS, 
            dtype=np.float32) * \
                self.cfg.TRAIN.SYNTH_DATA.AUGMENT.SMPL.SHAPE_STD
        self.mean_shape = np.zeros(
            self.cfg.MODEL.NUM_SMPL_BETAS, 
            dtype=np.float32)
        self.delta_style_std_vector = np.ones(
            self.cfg.MODEL.NUM_STYLE_PARAMS, 
            dtype=np.float32) * \
                self.cfg.TRAIN.SYNTH_DATA.AUGMENT.GARMENTOR.STYLE_STD
        self.mean_style = np.zeros(
            self.cfg.MODEL.NUM_STYLE_PARAMS, 
            dtype=np.float32)
        self.mean_cam_t = np.array(
            self.cfg.TRAIN.SYNTH_DATA.MEAN_CAM_T, 
            dtype=np.float32)
        
    def generate_random_params(
            self, 
            idx: Optional[int] = None
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        '''Generate random pose, shape, camera T, and style vector.'''
        if idx is None:
            idx = randrange(self.num_poses)
        pose = self.poses[idx % self.num_poses]

        shape = normal_sample_shape_numpy(
            mean_params=self.mean_shape,
            std_vector=self.delta_betas_std_vector
        )
        style_vector = normal_sample_style_numpy(
            num_garment_classes=GarmentClasses.NUM_CLASSES,
            mean_params=self.mean_style,
            std_vector=self.delta_style_std_vector
        )
        cam_t = augment_cam_t_numpy(
            self.mean_cam_t,
            xy_std=self.cfg.TRAIN.SYNTH_DATA.AUGMENT.CAM.XY_STD,
            delta_z_range=self.cfg.TRAIN.SYNTH_DATA.AUGMENT.CAM.DELTA_Z_RANGE
        )
        return (
            pose,
            shape,
            style_vector,
            cam_t
        )


class SurrealDataPreGenerator(DataPreGenerator):

    '''A data pregenerator class specific to SURREAL dataset.'''

    DATASET_NAME = 'surreal'
    CHECKPOINT_COUNT = 100

    def __init__(self):
        '''Initialize superclass and create clothed renderer.'''
        super().__init__()
        self.renderer = ClothedRenderer(
            device='cuda:0',
            batch_size=1
        )

    def generate_sample(self, 
                        idx: int,
                        gender: str,
                        parametric_model: ParametricModel
                        ) -> Tuple[np.ndarray, np.ndarray, PreGeneratedSampleValues]:
        '''Generate a single training sample.'''
        pose, shape, style_vector, cam_t = self.generate_random_params(idx)

        print(f'Sample #{idx} ({gender}):')
        print(f'\tPose: {pose}')
        print(f'\tShape: {shape}')
        print(f'\tCam T: {cam_t}')
        print(f'\tStyle: {style_vector}')

        smpl_output_dict = parametric_model.run(
            pose=pose,
            shape=shape,
            style_vector=style_vector
        )
        rgb_img, seg_maps = self.renderer(
            smpl_output_dict,
            garment_classes=parametric_model.garment_classes,
            cam_t=cam_t
        )
        sample_values = PreGeneratedSampleValues(
            pose=pose,
            shape=shape,
            style_vector=style_vector,
            garment_labels=parametric_model.garment_classes.labels_vector,
            cam_t=cam_t,
            joints=smpl_output_dict['upper'].joints
        )
        return rgb_img, seg_maps, sample_values
    
    @staticmethod
    def _create_dirs(dataset_dir, img_dirname, seg_dirname):
        img_dir = os.path.join(dataset_dir, img_dirname)
        seg_dir = os.path.join(dataset_dir, seg_dirname)
        
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        if not os.path.exists(seg_dir):
            os.makedirs(seg_dir)

    def _save_values(self, 
                     samples_values: PreGeneratedValuesArray, 
                     dataset_dir: str) -> None:
        '''Save all sample values as a dictionary of numpy arrays.'''

        values_path = os.path.join(dataset_dir, self.VALUES_FNAME)
        np.save(values_path, samples_values.get())
        print(f'Saved samples values to {values_path}!')

    def _save_sample(self, 
                     dataset_dir: str, 
                     sample_idx: int, 
                     rgb_img: np.ndarray, 
                     seg_maps: np.ndarray, 
                     sample_values: PreGeneratedSampleValues,
                     samples_values: PreGeneratedValuesArray) -> None:
        '''Save RGB, seg maps (disk), and the values to the array (RAM).'''
        
        self._create_dirs(
            dataset_dir=dataset_dir,
            img_dirname=self.IMG_DIR,
            seg_dirname=self.SEG_MAPS_DIR)

        if rgb_img is not None:
            rgb_img = (rgb_img * 255).astype(np.uint8)
            img = Image.fromarray(rgb_img)
            img_dir = os.path.join(dataset_dir, self.IMG_DIR)
            img_path = os.path.join(
                img_dir, self.IMG_NAME_TEMPLATE.format(idx=sample_idx))
            img.save(img_path)
            print(f'Saved image: {img_path}')

        seg_dir = os.path.join(dataset_dir, self.SEG_MAPS_DIR)
        seg_path = os.path.join(
            seg_dir, self.SEG_MAPS_NAME_TEMPLATE.format(idx=sample_idx))
        np.savez_compressed(seg_path, seg_maps=seg_maps.astype(bool))
        print(f'Saved segmentation maps: {seg_path}')

        samples_values.append(sample_values)
        if sample_idx % self.CHECKPOINT_COUNT == 0 and sample_idx != 0:
            print(f'Saving values on checkpoint #{sample_idx}')
            self._save_values(samples_values, dataset_dir)
            
    def _create_values_array(self, dataset_dir: str
                             ) -> Tuple[PreGeneratedValuesArray, int]:
        values_fpath = os.path.join(dataset_dir, self.VALUES_FNAME)
        if os.path.exists(values_fpath):
            samples_dict = np.load(values_fpath, allow_pickle=True).item()
            num_generated = samples_dict['poses'].shape[0]
            samples_values = PreGeneratedValuesArray(
                samples_dict=samples_dict
            )
        else:
            samples_values = PreGeneratedValuesArray()
            num_generated = 0
        return samples_values, num_generated
    
    def _log_class(self, 
                   gender: str, 
                   upper_class: str, 
                   lower_class: str, 
                   num_samples_per_class: int,
                   num_generated: int):
        total_num_samples = self.num_poses \
            if num_samples_per_class is None else num_samples_per_class
        subset_str = f'{gender}-{upper_class}-{lower_class}'
        num_samples_to_generate = total_num_samples - num_generated
        print(f'Generating {total_num_samples}-{num_generated}='
              f'{num_samples_to_generate} samples for {subset_str}...')
        return total_num_samples

    def generate(self, 
                 gender: str,
                 upper_class: str,
                 lower_class: str,
                 num_samples_per_class: int) -> None:
        '''(Pre-)generate the whole dataset.'''

        garment_classes = GarmentClasses(upper_class, lower_class)
        parametric_model = ParametricModel(
            gender=gender, 
            garment_classes=garment_classes
        )
        dataset_dir = self.dataset_path_template.format(
            dataset_name=self.DATASET_NAME,
            gender=gender,
            upper_garment_class=upper_class,
            lower_garment_class=lower_class
        )  
        samples_values, num_generated = self._create_values_array(
            dataset_dir=dataset_dir
        )
        total_num_samples = self._log_class(
            gender=gender, 
            upper_class=upper_class, 
            lower_class=lower_class,
            num_samples_per_class=num_samples_per_class,
            num_generated=num_generated
        )

        for pose_idx in range(num_generated, total_num_samples):
            rgb_img, seg_maps, sample_values = self.generate_sample(
                idx=pose_idx, 
                gender=gender, 
                parametric_model=parametric_model)
            self._save_sample(
                dataset_dir=dataset_dir, 
                sample_idx=pose_idx, 
                rgb_img=rgb_img, 
                seg_maps=seg_maps, 
                sample_values=sample_values,
                samples_values=samples_values
            )
        torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gender', '-G', type=str, choices=['male', 'female'],
                        help='Gender string.')
    parser.add_argument('--upper_class', '-U', type=str, choices=['t-shirt', 'shirt'],
                        help='Upper class string.')
    parser.add_argument('--lower_class', '-L', type=str, choices=['pant', 'short-pant'],
                        help='Lower class string.')
    parser.add_argument('--num_samples', '-N', type=int, default=None,
                        help='Number of samples to have for the class after the generation is done.')
    args = parser.parse_args()
    
    surreal_pregenerator = SurrealDataPreGenerator()
    surreal_pregenerator.generate(
        gender=args.gender,
        upper_class=args.upper_class,
        lower_class=args.lower_class,
        num_samples_per_class=args.num_samples
    )
