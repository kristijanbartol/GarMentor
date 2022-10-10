from typing import List, Tuple
from dataclasses import dataclass, fields
import numpy as np
import os
from PIL import Image
import sys

_module_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(_module_dir))

from configs import paths
from configs.poseMF_shapeGaussian_net_config import get_cfg_defaults
from utils.augmentation.smpl_augmentation import (
    normal_sample_shape_numpy,
    normal_sample_style_numpy
)
from models.parametric_model import ParametricModel
from utils.augmentation.cam_augmentation import augment_cam_t_numpy
from utils.garment_classes import GarmentClasses
from renderers.surreal_renderer import SurrealRenderer


@dataclass
class PreGeneratedSampleValues:

    '''A standard dataclass used to handle pregenerated sample values.'''

    pose: np.ndarray                    # (72,)
    shape: np.ndarray                   # (10,)
    style_vector: np.ndarray            # (4, 10)
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

    def __init__(self):
        self._samples_dict = {}
        self.keys = []
        self.numpied = False    # to avoid errors if requesting values many times

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
        if not self.numpied:
            for ks in self.keys:
                self._samples_dict[ks] = np.array(self._samples_dict[ks])
        self.numpied = True
        return self._samples_dict


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

    IMG_NAME_TEMPLATE = '{idx:5d}.png'
    SEG_MAPS_NAME_TEMPLATE = '{idx:5d}.npy'
    VALUES_FNAME = 'values.npy'

    def __init__(self):
        self.dataset_path_template = os.path.join(
            self.DATA_ROOT_DIR,
            '{dataset_name}',
            '{gender}',
            '{upper_garment_class}+{lower_garment_class}'
        )


class SurrealDataPreGenerator(DataPreGenerator):

    '''A data pregenerator class specific to SURREAL dataset.'''

    DATASET_NAME = 'surreal'

    def __init__(self):
        '''Initialize useful arrays, renderer, and load poses.'''
        
        super().__init__()
        self.cfg = get_cfg_defaults()
        self._init_useful_arrays()
        self.renderer = SurrealRenderer()
        self.poses = self._load_poses()

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
        self.x_axis = np.array([1., 0., 0.], dtype=np.float32)
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

    def generate_sample(self, 
                        idx: int,
                        gender: str,
                        parametric_model: ParametricModel
                        ) -> Tuple[np.ndarray, np.ndarray, PreGeneratedSampleValues]:
        '''Generate a single training sample.'''

        pose: np.ndarray = self.poses[idx]

        shape: np.ndarray = normal_sample_shape_numpy(
            mean_params=self.mean_shape,
            std_vector=self.delta_betas_std_vector)

        cam_t: np.ndarray = augment_cam_t_numpy(
            self.mean_cam_t,
            xy_std=self.cfg.TRAIN.SYNTH_DATA.AUGMENT.CAM.XY_STD,
            delta_z_range=self.cfg.TRAIN.SYNTH_DATA.AUGMENT.CAM.DELTA_Z_RANGE)

        style_vector: np.ndarray = normal_sample_style_numpy(
            num_garment_classes=GarmentClasses.NUM_CLASSES,
            mean_params=self.mean_style,
            std_vector=self.delta_style_std_vector)

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

        rgb_img, seg_maps = self.renderer.render(
            smpl_output_dict,
            garment_classes=parametric_model.garment_classes,
            cam_t=cam_t
        )

        sample_values = PreGeneratedSampleValues(
            pose=pose,
            shape=shape,
            style_vector=style_vector,
            cam_t=cam_t,
            joints=smpl_output_dict['upper'].joints
        )

        return rgb_img, seg_maps, sample_values

    def _save_sample(self, 
                     dataset_dir: str, 
                     sample_idx: int, 
                     rgb_img: np.ndarray, 
                     seg_maps: np.ndarray, 
                     sample_values: PreGeneratedSampleValues,
                     samples_values: PreGeneratedValuesArray) -> None:
        '''Save RGB, seg maps (disk), and the values to the array (RAM).'''

        if rgb_img is not None:
            img = Image.fromarray(rgb_img)
            img_dir = os.path.join(dataset_dir, self.IMG_DIR)
            img_path = os.path.join(img_dir, self.IMG_NAME_TEMPLATE.format(sample_idx))
            img.save(img_path)
            print(f'Saved image: {img_path}')

        seg_dir = os.path.join(dataset_dir, self.SEG_MAPS_DIR)
        seg_path = os.path.join(
            seg_dir, self.SEG_MAPS_NAME_TEMPLATE.format(sample_idx))
        np.save(seg_path, seg_maps)
        print(f'Saved segmentation maps: {seg_path}')

        samples_values.append(sample_values)

    def _save_values(self, 
                     samples_values: PreGeneratedValuesArray, 
                     dataset_dir: str) -> None:
        '''Save all sample values as a dictionary of numpy arrays.'''

        values_path = os.path.join(dataset_dir, self.VALUES_FNAME)
        np.save(values_path, samples_values.get())
        print(f'Saved samples values to {values_path}!')

    def generate(self) -> None:
        '''(Pre-)generate the whole dataset.'''

        for gender in ['male', 'female']:
            for upper_class in GarmentClasses.UPPER_GARMENT_CLASSES:
                for lower_class in GarmentClasses.LOWER_GARMENT_CLASSES:
                    garment_classes = GarmentClasses(upper_class, lower_class)
                    parametric_model = ParametricModel(
                        gender=gender, 
                        garment_classes=garment_classes
                    )
                    samples_values = PreGeneratedValuesArray()

                    dataset_dir = self.dataset_path_template.format(
                        dataset_name=self.DATASET_NAME,
                        gender=gender,
                        upper_garment_class=upper_class,
                        lower_garment_class=lower_class
                    )
                    num_samples = self.poses.shape[0]
                    subset_str = f'{gender}-{upper_class}-{lower_class}'
                    print(f'Generating {num_samples} samples for {subset_str}...')

                    for pose_idx in range(num_samples):
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
                    self._save_values(samples_values, dataset_dir)


if __name__ == '__main__':
    surreal_pregenerator = SurrealDataPreGenerator()
    surreal_pregenerator.generate()
