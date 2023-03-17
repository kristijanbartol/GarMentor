from typing import List, Tuple, Optional
from dataclasses import dataclass, fields
import numpy as np
import os
from random import randrange

from configs import paths
from configs.poseMF_shapeGaussian_net_config import get_cfg_defaults
from utils.augmentation.cam_augmentation import augment_cam_t_numpy
from utils.augmentation.smpl_augmentation import (
    normal_sample_shape_numpy,
    normal_sample_style_numpy
)
from utils.garment_classes import GarmentClasses


@dataclass
class PreparedSampleValues:

    """
    A standard dataclass used to handle prepared sample values.

    Note that cam_t was supposed to be deprecated, but it is actually
    required by the SURREAL-based datasets for the keypoint strategy
    which takes ground truth joints and projects them orthographically.
    Then the cam_t is applied on top to properly place 2D keypoints.
    For AGORA-like dataset it's a dummy value and it won't be used in
    the training loop with the AGORA-like data. In case I decide that
    the keypoint strategy (ground truth 3D -> projection -> cam_t) is
    necessarily inferior, I will remove cam_t (kbartol).

    Joints 2D data is common for all the datasets, but it differs
    between SURREAL-like and AGORA-like dataset in a way that for 
    SURREAL the 2D joints are used only if the keypoints are pre-
    extracted using the 2D keypoint detector. In case of AGORA, 2D
    joints are mandatory as they can't be projected afterwards (no
    X, Y, Z camera locations in the training time). However, AGORA
    can also create 2D joints by using 2D keypoint detector.

    Bounding box information, on the other hand, is specific to AGORA-
    like data.
    """

    pose: np.ndarray                    # (72,)
    shape: np.ndarray                   # (10,)
    style_vector: np.ndarray            # (4, 10)
    garment_labels: np.ndarray          # (4,)
    cam_t: np.ndarray                   # (3,)
    joints_3d: np.ndarray               # (17, 3)
    joints_2d: np.ndarray               # (17, 2)
    bbox: np.ndarray                    # (2, 2)

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


class PreparedValuesArray():

    """
    A class used to keep track of an array of prepared sampled values.
    """

    def __init__(self, samples_dict: dict = None):
        if samples_dict is not None:
            self._samples_dict = {k: list(v) for k, v in samples_dict.items()}
            self.keys = samples_dict.keys()
        else:
            self._samples_dict = {}
            self.keys = []

    def _set_dict_keys(self, sample_dict_keys: List[str]) -> None:
        """
        Add 's' to key name specify plural.
        """
        self.keys = [x + 's' for x in sample_dict_keys]

    def append(self, values: PreparedSampleValues) -> None:
        """
        Mimics adding to list of values to an array. Saves to latent dict.
        """
        if not self._samples_dict:
            self._set_dict_keys(values.keys())
            for k in self.keys:
                self._samples_dict[k] = []
        for ks, k in zip(self.keys, values.keys()):
            self._samples_dict[ks].append(values[k])

    def get(self) -> dict:
        """
        Get the dictionary with all np.ndarray items.
        """
        return_dict = {k: None for k, _ in self._samples_dict.items()}
        for ks in self.keys:
            return_dict[ks] = np.array(self._samples_dict[ks])
        return return_dict


class DataGenerator(object):

    """
    An abstract class which contains useful objects for data generation.

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
    """

    # TODO: Move these paths to a common path definitions file.
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
        """
        Load poses. Adapted from the original HierProb3D code.
        """
        data = np.load(paths.TRAIN_POSES_PATH)
        fnames = data['fnames']
        poses = data['poses']
        indices = [i for i, x in enumerate(fnames)
                    if (x.startswith('h36m') or x.startswith('up3d') or x.startswith('3dpw'))]
        return np.stack([poses[i] for i in indices], axis=0)
    
    def _init_useful_arrays(self) -> None:
        """
        These useful arrays are used to randomly sample data and transform points.
        """
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
        
    def _create_values_array(
            self, 
            dataset_dir: str
        ) -> Tuple[PreparedValuesArray, int]:
        """
        Create an array of prepared values, which consists of:
          - pose parameters
          - shape parameters
          - style vector
          - garment labels
          - camera translations (needed for target 2D kpt projections)
          - 3D joints

        The array is of type PreparedValuesArray and it contains
        PreparedSampleValues. If the values stored in a file are
        empty, then a new PreparedValuesArray container is created.
        Instead, the PreparedValuesArray starts with the values that
        are already saved into a dataset file.
        """
        values_fpath = os.path.join(dataset_dir, self.VALUES_FNAME)
        if os.path.exists(values_fpath):
            samples_dict = np.load(values_fpath, allow_pickle=True).item()
            num_generated = samples_dict['poses'].shape[0]
            samples_values = PreparedValuesArray(
                samples_dict=samples_dict
            )
        else:
            samples_values = PreparedValuesArray()
            num_generated = 0
        return samples_values, num_generated
    
    def _save_values(
            self, 
            samples_values: PreparedValuesArray, 
            dataset_dir: str
        ) -> None:
        """
        Save all sample values as a dictionary of numpy arrays.
        """
        values_path = os.path.join(dataset_dir, self.VALUES_FNAME)
        np.save(values_path, samples_values.get())
        print(f'Saved samples values to {values_path}!')
        
    def generate_random_params(
            self, 
            idx: Optional[int] = None
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate random pose, shape, camera T, and style vector.
        """
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
