from typing import Tuple, Union, Iterator, Any, Optional, Dict
from dataclasses import dataclass, fields
import numpy as np
import torch
import torch.cuda
import os
import sys
import argparse

sys.path.append('/garmentor')

from configs.const import SURREAL_DATASET_NAME
import configs.paths as paths
from configs.poseMF_shapeGaussian_net_config import get_cfg_defaults
from data.cat.parameters import Parameters
from data.cat.common import get_dataset_dirs
from models.pose2D_hrnet import get_pretrained_detector
from predict.predict_hrnet import predict_hrnet
from utils.garment_classes import GarmentClasses
from utils.image_utils import normalize_features
from vis.visualizers.clothed import ClothedVisualizer
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



class DataGenerator():

    """
    A data generation class specific to SURREAL dataset.
    """

    DATASET_NAME = SURREAL_DATASET_NAME
    CHECKPOINT_COUNT = 100

    def __init__(
            self,
            preextract_kpt=False
        ) -> None:
        """
        Initialize superclass and create clothed renderer.
        """
        self.preextract_kpt = preextract_kpt
        self.device = 'cuda:0'
        if preextract_kpt:
            self.kpt_model, self.kpt_cfg = get_pretrained_detector()
        self.keypoints_visualizer = KeypointsVisualizer()

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

    def generate_sample(
            self, 
            parameters: Parameters,
            data_split: str,
            idx: int,
            gender: str,
            clothed_visualizer: ClothedVisualizer
        ) -> Tuple[np.ndarray, np.ndarray, PreparedSampleValues]:
        """
        Generate a single training sample.
        """
        params_dict = parameters.get(
            data_split=data_split, 
            idx=idx
        )
        rgb_img, seg_maps, joints_3d = clothed_visualizer.vis_from_params(
            pose=params_dict['pose'],
            shape=params_dict['shape'],
            style_vector=params_dict['style']
        )
        joints_2d, joints_conf, bbox = self._predict_joints(
            rgb_tensor=torch.swapaxes(rgb_img, 0, 2),
        )
        rgb_img = rgb_img.cpu().numpy()
        rgb_img, seg_maps, joints_2d = normalize_features(
            rgb_img,
            seg_maps,
            joints_2d
        )
        sample_values = PreparedSampleValues(
            pose=params_dict['pose'],
            shape=params_dict['shape'],
            style_vector=params_dict['style'],
            garment_labels=clothed_visualizer.garment_classes.labels_vector,
            joints_3d=joints_3d,
            joints_conf=joints_conf,
            joints_2d=joints_2d,
            bbox=bbox       # TODO: Remove as it is unrealiable at this stage.
        )
        return (
            rgb_img,
            seg_maps, 
            sample_values
        )
    
    @staticmethod
    def _create_dirs(
            dataset_dir: str, 
            img_dirname: str, 
            seg_dirname: str,
            verification_dirname: str
        ) -> None:
        """
        Create image and segmentation mask directories.
        """
        for dirname in [img_dirname, seg_dirname, verification_dirname]:
            dirpath = os.path.join(dataset_dir, dirname)
            if not os.path.exists(dirpath):
                os.makedirs(dirpath)

    def _save_verification(
            self,
            dataset_dir: str,
            sample_idx: int,
            rgb_img: np.ndarray,
            joints_2d: Union[np.ndarray, None],
            seg_maps: np.ndarray,
            clothed_visualizer: ClothedVisualizer
    ) -> None:
        """
        Save RGB (+2D joints) and seg maps as images for verification.
        """
        print(f'Saving vefification images and seg maps (#{sample_idx})...')
        verify_dir_path = os.path.join(
            dataset_dir,
            paths.VERIFY_DIR
        )
        verify_rgb_path = os.path.join(
            verify_dir_path,
            paths.IMG_NAME_TEMPLATE.format(sample_idx=sample_idx)
        )
        seg_maps_paths = [os.path.join(
            verify_dir_path,
            paths.SEG_IMGS_NAME_TEMPLATE.format(
                sample_idx=sample_idx,
                idx=x
            )) for x in range(5)
        ]
        if joints_2d is not None:
            rgb_img = self.keypoints_visualizer.vis_keypoints(
                kpts=joints_2d,
                back_img=rgb_img
            )
        self.keypoints_visualizer.save_vis(
            img=rgb_img,
            save_path=verify_rgb_path
        )
        clothed_visualizer.save_masks_as_images(
            seg_masks=seg_maps,
            save_paths=seg_maps_paths
        )

    def _save_sample(
            self, 
            dataset_dir: str, 
            sample_idx: int, 
            rgb_img: np.ndarray, 
            seg_maps: np.ndarray, 
            sample_values: PreparedSampleValues,
            values_array: PreparedValuesArray,
            clothed_visualizer: ClothedVisualizer
        ) -> None:
        """
        Save RGB and seg maps to disk, and update the values in the array (RAM).
        """
        self._create_dirs(
            dataset_dir=dataset_dir,
            img_dirname=paths.RGB_DIR,
            seg_dirname=paths.SEG_MAPS_DIR,
            verification_dirname=paths.VERIFY_DIR
        )
        if rgb_img is not None:
            img_dir = os.path.join(
                dataset_dir, 
                paths.RGB_DIR
            )
            img_path = os.path.join(
                img_dir, 
                paths.IMG_NAME_TEMPLATE.format(sample_idx=sample_idx)
            )
            clothed_visualizer.save_vis(
                rgb_img=rgb_img,
                save_path=img_path
            )
        seg_dir = os.path.join(dataset_dir, paths.SEG_MAPS_DIR)
        seg_path = os.path.join(
            seg_dir, paths.SEG_MAPS_NAME_TEMPLATE.format(
                sample_idx=sample_idx)
            )
        clothed_visualizer.save_masks(
            seg_masks=seg_maps,
            save_path=seg_path
        )

        values_array.append(sample_values)
        if sample_idx % self.CHECKPOINT_COUNT == 0 and sample_idx != 0:
            self._save_values(
                values_array=values_array, 
                dataset_dir=dataset_dir,
                sample_idx=sample_idx
            )
            self._save_verification(
                dataset_dir=dataset_dir,
                sample_idx=sample_idx,
                rgb_img=rgb_img, 
                joints_2d=sample_values.joints_2d, 
                seg_maps=seg_maps,
                clothed_visualizer=clothed_visualizer
            )

    @staticmethod
    def _create_values_splits(
            dataset_dirs: Dict[str, str]
        ) -> Tuple[Dict[str, PreparedValuesArray], Dict[str, int]]:
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
        values_arrays = {}
        nums_generated = {}
        for data_split in dataset_dirs:
            values_fpath = os.path.join(dataset_dirs[data_split], paths.VALUES_FNAME)
            if os.path.exists(values_fpath):
                samples_dict = np.load(values_fpath, allow_pickle=True).item()
                num_generated = samples_dict['poses'].shape[0]
                values_arrays[data_split] = PreparedValuesArray(samples_dict=samples_dict)
                nums_generated[data_split] = num_generated
            else:
                values_arrays[data_split] = PreparedValuesArray()
                nums_generated[data_split] = 0
        return values_arrays, nums_generated
    
    @staticmethod
    def _save_values(
            values_array: PreparedValuesArray, 
            dataset_dir: str,
            sample_idx: int
        ) -> None:
        """
        Save all sample values as a dictionary of numpy arrays.
        """
        print(f'Saving values on checkpoint #{sample_idx}')
        values_path = os.path.join(dataset_dir, paths.VALUES_FNAME)
        np.save(values_path, values_array.get())
        print(f'Saved samples values to {values_path}!')

    def generate(
            self, 
            param_cfg: Dict,
            upper_class: str,
            lower_class: str,
            gender: str,
            img_wh: int,
            num_samples_to_reach: int
        ) -> None:
        """
        (Pre-)generate the dataset for particular upper+lower garment class.

        Create GarmentClasses and ClothedVisualizer classes. They contain
        classes such as ParametricModel and are therefore able to create
        bodies based on body models and other tasks hidden from the user
        of this method.

        The generated samples are stored in files (RGB images, seg maps),
        i.e., update in the corresponding sample values' arrays. The sample
        arrays are frequently updated on the disk in case the failure
        happens along the way.
        """
        parameters = Parameters(param_cfg=param_cfg)
        garment_classes = GarmentClasses(upper_class, lower_class)
        clothed_visualizer = ClothedVisualizer(
            device=self.device,
            gender=gender,
            garment_classes=garment_classes,
            img_wh=img_wh
        )
        dataset_dirs = get_dataset_dirs(
            param_cfg=param_cfg,
            upper_class=upper_class,
            lower_class=lower_class,
            gender=gender,
            img_wh=img_wh
        )
        values_splits, nums_generated = self._create_values_splits(
            dataset_dirs=dataset_dirs
        )
        total_nums_samples = {
            'train': num_samples_to_reach, 
            'valid': num_samples_to_reach // 5
        }
        for data_split in ['train', 'valid']:
            for pose_idx in range(nums_generated[data_split], total_nums_samples[data_split]):
                rgb_img, seg_maps, sample_values = self.generate_sample(
                    parameters=parameters,
                    data_split=data_split,
                    idx=pose_idx, 
                    gender=gender, 
                    clothed_visualizer=clothed_visualizer
                )
                self._save_sample(
                    dataset_dir=dataset_dirs[data_split], 
                    sample_idx=pose_idx, 
                    rgb_img=rgb_img, 
                    seg_maps=seg_maps, 
                    sample_values=sample_values,
                    values_array=values_splits[data_split],
                    clothed_visualizer=clothed_visualizer
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
    parser.add_argument('--preextract', dest='preextract', action='store_true', 
                        help='Whether to pre-extract 2D joint using HRNet pose detector.')
    parser.add_argument('--img_wh', '-I', type=int, choices=[256, 512, 1024, 2048], default=256,
                        help='The size of weight, i.e., height of the images.')
    args = parser.parse_args()
    
    data_generator = DataGenerator(
        preextract_kpt=args.preextract
    )
    param_cfg = {
        'pose': {
            'strategy': 'zero',
            'interval': 'intra'
        },
        'global_orient': {
            'strategy': 'zero',
            'interval': 'extra'
        },
        'shape': {
            'strategy': 'normal',
            'interval': 'extra'
        },
        'style': {
            'strategy': 'normal',
            'interval': 'extra'
        }
    }
    data_generator.generate(
        param_cfg=param_cfg,
        gender=args.gender,
        upper_class=args.upper_class,
        lower_class=args.lower_class,
        img_wh=args.img_wh,
        num_samples_to_reach=args.num_samples
    )
