from typing import Optional
from dataclasses import dataclass, fields
import numpy as np
from models.parametric_model import ParametricModel
import torch
import os
import cv2
import random
from PIL import Image

from configs import paths
from configs.poseMF_shapeGaussian_net_config import get_poseMF_shapeGaussian_cfg_defaults
from utils.augmentation.smpl_augmentation import (
    normal_sample_params_numpy,
    normal_sample_style_numpy
)
from utils.augmentation.cam_augmentation import augment_cam_t_numpy
from renderers.non_textured_renderer import NonTexturedRenderer
from utils.garment_classes import GarmentClasses

from tailornet_for_garmentor.models.tailornet_model import get_best_runner as get_tn_runner
from tailornet_for_garmentor.models.smpl4garment import SMPL4Garment
from tailornet_for_garmentor.utils.rotation import normalize_y_rotation
from tailornet_for_garmentor.utils.interpenetration import remove_interpenetration_fast


@dataclass
class PreGeneratedSampleValues:

    pose: np.ndarray
    shape: np.ndarray
    style_vector: np.ndarray
    cam_t: np.ndarray
    joints: np.ndarray
    garment_labels_vector: np.ndarray

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

    def __init__(self):
        self.empty()
        self.keys = []
        self.numpied = False    # to avoid errors if requesting values many times

    def empty(self):
        self._samples_dict = {}

    def _set_dict_keys(self, sample_dict_keys):
        # Add 's' to key name specify plural.
        self.keys = [x + 's' for x in sample_dict_keys]

    def append(self, values):
        # NOTE (kbartol): This is inefficient, but at least it encapsulates saving.
        if not self._samples_dict:
            self._set_dict_keys(values.keys())
            for k in self.keys:
                self._samples_dict[k] = []
        for ks, k in zip(self.keys, values.keys()):
            self._samples_dict[ks].append(values[k])

    def get(self):
        if not self.numpied:
            for ks in self.keys:
                self._samples_dict[ks] = np.array(self._samples_dict[ks])
        self.numpied = True
        return self._samples_dict


class DataPreGenerator(object):

    # The paths and template to files and folders where data will be stored.
    # /data/garmentor/
    #       {dataset_name}/
    #           {gender}/
    #               values.npy
    #               rgb/
    #                   {idx:5d}.png
    #               segmentations/
    #                   {idx:5d}_{garment_class}.png

    # Directory path templates.
    DATA_ROOT_DIR = '/data/garmentor/'
    IMG_DIR = 'rgb/'
    SEG_MAPS_DIR = 'segmentations/'

    # Filename templates.
    IMG_NAME_TEMPLATE = '{idx:5d}.png'
    SEG_MAPS_NAME_TEMPLATE = '{idx:5d}.npy'
    VALUES_FNAME = 'values.npy'

    def __init__(self):
        self.dataset_path_template = os.path.join(
            self.DATA_ROOT_DIR,
            '{dataset_name}',
            '{gender}'
        )

        # Parametric models gathers TailorNet and SMPL4Garment functionalities.
        self.parametric_model = ParametricModel()

        # Abstract renderer is initialized to None.
        self.renderer = None

        # Initialize lists to store relatively small values (e.g. not images and maps).
        self.samples_values = PreGeneratedValuesArray()


class SurrealDataPreGenerator(DataPreGenerator):

    DATASET_NAME = 'surreal'

    def __init__(self):
        super().__init__()

        # Load configuration.
        self.pose_shape_cfg = get_poseMF_shapeGaussian_cfg_defaults()

        # Initialize useful, predefined arrays.
        self._init_useful_arrays()

        # Initialize SURREAL-specific renderer.
        # TODO: Update this renderer to support proper RGB rendering (issue #20).
        self.renderer = NonTexturedRenderer()

        # Load SMPL poses.
        data = np.load(paths.TRAIN_POSES_PATH)
        fnames = data['fnames']
        poses = data['poses']
        
        # Not AMASS poses (kbartol: That's just taken from the original data generation code).
        indices = [i for i, x in enumerate(fnames)
                    if (x.startswith('h36m') or x.startswith('up3d') or x.startswith('3dpw'))]
        self.poses = np.stack([poses[i] for i in indices], axis=0)
        self.num_samples = poses.shape[0]

    def _init_useful_arrays(self):
        self.x_axis = np.array([1., 0., 0.], dtype=np.float32)
        self.delta_betas_std_vector = np.ones(
            self.pose_shape_cfg.MODEL.NUM_SMPL_BETAS, 
            dtype=np.float32) * \
                self.pose_shape_cfg.TRAIN.SYNTH_DATA.AUGMENT.SMPL.SHAPE_STD
        self.mean_shape = np.zeros(
            self.pose_shape_cfg.MODEL.NUM_SMPL_BETAS, 
            dtype=np.float32)
        self.delta_style_std_vector = np.ones(
            self.pose_shape_cfg.MODEL.NUM_STYLE_PARAMS, 
            dtype=np.float32) * \
                self.pose_shape_cfg.TRAIN.SYNTH_DATA.AUGMENT.GARMENTOR.STYLE_STD
        self.mean_style = np.zeros(
            self.pose_shape_cfg.MODEL.NUM_STYLE_PARAMS, 
            dtype=np.float32)
        self.mean_cam_t = np.array(
            self.pose_shape_cfg.TRAIN.SYNTH_DATA.MEAN_CAM_T, 
            dtype=np.float32)
        self.mean_cam_t = np.broadcast_to(
            self.mean_cam_t[None, :], 
            (self.pose_shape_cfg.TRAIN.BATCH_SIZE, -1))

    def generate_sample(self, idx, gender):
        pose = self.poses[idx]

        # Randomly sample body shape.
        shape = normal_sample_params_numpy(
            mean_params=self.mean_shape,
            std_vector=self.delta_betas_std_vector)

        # Random sample camera translation.
        cam_t = augment_cam_t_numpy(
            self.mean_cam_t,
            xy_std=self.pose_shape_cfg.TRAIN.SYNTH_DATA.AUGMENT.CAM.XY_STD,
            delta_z_range=self.pose_shape_cfg.TRAIN.SYNTH_DATA.AUGMENT.CAM.DELTA_Z_RANGE)

        # Randomly sample garment classes (upper and lower garment class).
        garment_class_pair = self.get_random_garment_classes()

        # Randomly sample garment parameters.
        style_vector = normal_sample_style_numpy(
            num_garment_classes=GarmentClasses.NUM_CLASSES,
            mean_params=self.mean_style,
            std_vector=self.delta_style_std_vector)

        # Call the parametric model that encapsulates TN, SMPL, and interpenetration resolution.
        upper_smpl_output, lower_smpl_output = self.parametric_model.run(
            gender=gender,
            garment_class_pair=garment_class_pair,
            pose=pose,
            shape=shape,
            style_vector=style_vector
        )

        # Get the joints from either upper or lower SMPL output, it's the same.
        joints = upper_smpl_output.joints

        # Render. Get only cloth segmentation maps, for now.
        seg_maps = self.renderer(
            body_verts=upper_smpl_output.body_verts,
            body_faces=upper_smpl_output.body_faces,
            upper_garment_verts=upper_smpl_output.garment_verts,
            upper_garment_faces=upper_smpl_output.garment_faces,
            lower_garment_verts=lower_smpl_output.garment_verts,
            lower_garment_faces=lower_smpl_output.garment_faces,
            cam_t=cam_t
        )

        # TODO: Render RGB image here, after issue #20 is solved.
        # rgb_img = render_output['rgb_img']
        rgb_img = None

        # For convenience, gather all the parameters and values in a single output.
        sample_values = PreGeneratedSampleValues(
            pose=pose,
            shape=shape,
            style_vector=style_vector,
            cam_t=cam_t,
            joints=joints
        )

        return rgb_img, seg_maps, sample_values

    def _save_sample(self, dataset_dir, sample_idx, rgb_img, seg_maps, sample_values):
        if rgb_img is not None:
            img = Image.fromarray(rgb_img)
            img_dir = os.path.join(dataset_dir, self.IMG_DIR)
            img_path = os.path.join(img_dir, self.IMG_NAME_TEMPLATE.format(sample_idx))
            img.save(img_path)

        seg_dir = os.path.join(dataset_dir, self.SEG_MAPS_DIR)
        seg_path = os.path.join(
            seg_dir, self.SEG_MAPS_NAME_TEMPLATE.format(sample_idx))
        np.save(seg_path, seg_maps)

        self.samples_values.append(sample_values)

    def _save_params(self, dataset_dir):
        params_path = os.path.join(dataset_dir, self.VALUES_FNAME)
        np.save(params_path, self.samples_values.get())

    def generate(self):
        for gender in ['male', 'female']:
            self.samples_values.empty()
            dataset_dir = self.dataset_path_template.format(
                dataset=self.DATASET_NAME,
                gender=gender
            )

            for pose_idx in range(self.poses.shape[0]):
                rgb_img, seg_maps, sample_values = self.generate_sample(
                    pose_idx, gender)
                self._save_sample(
                    dataset_dir=dataset_dir, 
                    sample_idx=pose_idx, 
                    rgb_img=rgb_img, 
                    seg_maps=seg_maps, 
                    sample_values=sample_values
                )
            self._save_params(dataset_dir)


if __name__ == '__main__':
    # Use for testing the module.
    pass
