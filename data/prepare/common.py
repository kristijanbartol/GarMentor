from typing import Any, Iterator, Tuple, Optional, Dict, Union
from dataclasses import dataclass, fields
import numpy as np
import os
import torch
import utils.sampling_utils

from configs import paths
from configs.poseMF_shapeGaussian_net_config import get_cfg_defaults
from models.pose2D_hrnet import get_pretrained_detector
from predict.predict_hrnet import predict_hrnet
from vis.visualizers.keypoints import KeypointsVisualizer


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

    pose: np.ndarray                        # (72,)
    shape: np.ndarray                       # (10,)
    style_vector: np.ndarray                # (4, 10)
    garment_labels: np.ndarray              # (4,)
    joints_3d: np.ndarray                   # (17, 3)
    joints_conf: np.ndarray                 # (17,)
    joints_2d: Optional[np.ndarray] = None  # (17, 2)
    cam_t: Optional[np.ndarray] = None      # (3,)
    bbox: Optional[np.ndarray] = None       # (2, 2)

    def __getitem__(
            self, 
            key: str
        ) -> np.ndarray:
        return getattr(self, key)

    def get(
            self, 
            key, 
            default=None
        ) -> Union[Any, None]:
        return getattr(self, key, default)

    def __iter__(self) -> Iterator[str]:
        return self.keys()

    def keys(self) -> Iterator[str]:
        keys = [t.name for t in fields(self)]
        return iter(keys)

    def values(self) -> Iterator[Any]:
        values = [getattr(self, t.name) for t in fields(self)]
        return iter(values)

    def items(self) -> Iterator[Tuple[str, Any]]:
        data = [(t.name, getattr(self, t.name)) for t in fields(self)]
        return iter(data)


class PreparedValuesArray():

    """
    A class used to keep track of an array of prepared sampled values.
    """

    def __init__(
            self, 
            samples_dict: Optional[Dict[str, np.ndarray]] = None
        ) -> None:
        if samples_dict is not None:
            self._samples_dict = {k: list(v) for k, v in samples_dict.items()}
            self.keys = samples_dict.keys()
        else:
            self._samples_dict = {}
            self.keys = []

    def _set_dict_keys(
            self, 
            sample_dict_keys: Iterator[str]
        ) -> None:
        """
        Add 's' to key name specify plural.
        """
        self.keys = [x + 's' for x in sample_dict_keys]

    def append(
            self, 
            values: PreparedSampleValues
        ) -> None:
        """
        Mimics adding to list of values to an array. Saves to latent dict.
        """
        if not self._samples_dict:
            self._set_dict_keys(values.keys())
            for k in self.keys:
                self._samples_dict[k] = []
        for ks, k in zip(self.keys, values.keys()):
            self._samples_dict[ks].append(values[k])

    def get(self) -> Dict[str, np.ndarray]:
        """
        Get the dictionary with all np.ndarray items.
        """
        return_dict = {k: np.empty(0,) for k, _ in self._samples_dict.items()}
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

    def __init__(
            self,
            data_split,
            preextract_kpt=False,
        ):
        self.dataset_path_template = os.path.join(
            paths.DATA_ROOT_DIR,
            '{dataset_name}',
            '{gender}',
            '{upper_garment_class}+{lower_garment_class}'
        )
        self.cfg = get_cfg_defaults()
        self.sampling_cfg = self.cfg.TRAIN.SYNTH_DATA.SAMPLING
        self._init_sampling_methods()
        self._init_useful_arrays()
        self.data_split = data_split
        self.preextract_kpt = preextract_kpt
        if preextract_kpt:
            self.kpt_model, self.kpt_cfg = get_pretrained_detector()
        self.keypoints_visualizer = KeypointsVisualizer()
        self.poses, self.global_orients, self.shapes, self.styles = [np.empty(0,)] * 4
    
    def _init_useful_arrays(self) -> None:
        """
        These useful arrays are used to randomly sample data and transform points.
        """
        # TODO: Put some of these configurations into the constant list.
        self.delta_betas_std_vector = np.ones(
            self.cfg.MODEL.NUM_SMPL_BETAS, 
            dtype=np.float32) * \
                self.cfg.TRAIN.SYNTH_DATA.AUGMENT.SMPL.SHAPE_STD
        self.mean_shape = np.zeros(
            self.cfg.MODEL.NUM_SMPL_BETAS, 
            dtype=np.float32)
        self.shape_min: Optional[float] = self.cfg.TRAIN.SYNTH_DATA.AUGMENT.SMPL.SHAPE_MIN
        self.shape_max: Optional[float] = self.cfg.TRAIN.SYNTH_DATA.AUGMENT.SMPL.SHAPE_MAX
        self.num_garment_classes = self.cfg.TRAIN.SYNTH_DATA.AUGMENT.GARMENTOR.NUM_GARMENT_CLASSES
        self.delta_style_std_vector = np.ones(
            self.cfg.MODEL.NUM_STYLE_PARAMS, 
            dtype=np.float32) * \
                self.cfg.TRAIN.SYNTH_DATA.AUGMENT.GARMENTOR.STYLE_STD
        self.mean_style = np.zeros(
            self.cfg.MODEL.NUM_STYLE_PARAMS, 
            dtype=np.float32)
        self.style_min: Optional[float] = self.cfg.TRAIN.SYNTH_DATA.AUGMENT.GARMENTOR.STYLE_MIN
        self.style_max: Optional[float] = self.cfg.TRAIN.SYNTH_DATA.AUGMENT.GARMENTOR.STYLE_MAX
        self.mean_cam_t = np.array(
            self.cfg.TRAIN.SYNTH_DATA.MEAN_CAM_T, 
            dtype=np.float32)
        
    def _init_sampling_methods(self):
        self.samplers = {
            'pose': getattr(
                utils.sampling_utils, 
                f'sample_{self.sampling_cfg.STRATEGY.POSE}_pose'
            ),
            'global_orient': getattr(
                utils.sampling_utils, 
                f'sample_{self.sampling_cfg.STRATEGY.GLOBAL_ORIENT}_global_orient'
            ),
            'shape': getattr(
                utils.sampling_utils, 
                f'sample_{self.sampling_cfg.STRATEGY.SHAPE}_shape'
            ),
            'style': getattr(
                utils.sampling_utils, 
                f'sample_{self.sampling_cfg.STRATEGY.STYLE}_style'
            )
        }
        
    @staticmethod
    def _create_values_array(
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
        values_fpath = os.path.join(dataset_dir, paths.VALUES_FNAME)
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

    def _predict_joints(
            self,
            rgb_tensor: torch.Tensor
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.preextract_kpt:
            print('Running pose detection...')
            hrnet_output = predict_hrnet(
                hrnet_model=self.kpt_model,
                hrnet_config=self.kpt_cfg,
                image=rgb_tensor
            )
            joints_2d = hrnet_output['joints2D'].detach().cpu().numpy()[:, ::-1]
            joints_conf = hrnet_output['joints2Dconfs'].detach().cpu().numpy()
            bbox = hrnet_output['bbox']
        else:
            joints_2d = None
            joints_conf = np.ones(17,)
            bbox = None
        return joints_2d, joints_conf, bbox # type:ignore
    
    @staticmethod
    def _save_values(
            samples_values: PreparedValuesArray, 
            dataset_dir: str,
            sample_idx: int
        ) -> None:
        """
        Save all sample values as a dictionary of numpy arrays.
        """
        print(f'Saving values on checkpoint #{sample_idx}')
        values_path = os.path.join(dataset_dir, paths.VALUES_FNAME)
        np.save(values_path, samples_values.get())
        print(f'Saved samples values to {values_path}!')
        
    def sample_all_params(
            self, 
            num_train: int,
            num_valid: int
        ) -> None:
        """
        Generate random pose, shape, camera T, and style vector.
        """
        self.poses = self.samplers['pose'](
            num_train=num_train,
            num_valid=num_valid
        )
        self.global_orients = self.samplers['global_orient'](
            num_train=num_train,
            num_valid=num_valid
        )
        self.shapes = self.samplers['shape'](
            mean_params=self.mean_shape,
            std_vector=self.delta_betas_std_vector,
            num_train=num_train,
            num_valid=num_valid,
            intervals_type=self.sampling_cfg.GENERALIZATION.SHAPE,
            clip_min=self.sampling_cfg.CLIP_MIN.SHAPE,
            clip_max=self.sampling_cfg.CLIP_MAX.SHAPE
        )
        self.styles = self.samplers['style'](
            mean_params=self.mean_style,
            std_vector=self.delta_style_std_vector,
            num_train=num_train,
            num_valid=num_valid,
            intervals_type=self.sampling_cfg.GENERALIZATION.STYLE,
            clip_min=self.sampling_cfg.CLIP_MIN.STYLE,
            clip_max=self.sampling_cfg.CLIP_MAX.STYLE
        )

    def get_params(
            self,
            idx
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (
            self.poses[idx], 
            self.global_orients[idx], 
            self.shapes[idx], 
            self.styles[idx]
        )
