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
from models.pose2D_hrnet import get_pretrained_detector
from predict.predict_hrnet import predict_hrnet
from utils.garment_classes import GarmentClasses
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
            idx: int,
            gender: str,
            clothed_visualizer: ClothedVisualizer
        ) -> Tuple[np.ndarray, np.ndarray, PreparedSampleValues]:
        """
        Generate a single training sample.
        """
        pose, shape, style_vector = self.generate_random_params(idx)

        print(f'Sample #{idx} ({gender}):')
        print(f'\tPose: {pose}')
        print(f'\tShape: {shape}')
        print(f'\tStyle: {style_vector}')

        rgb_img, seg_maps, joints_3d = clothed_visualizer.vis_from_params(
            pose=pose,
            shape=shape,
            style_vector=style_vector
        )
        joints_2d, joints_conf, bbox = self._predict_joints(
            rgb_tensor=torch.swapaxes(rgb_img, 0, 2),
        )
        rgb_img = rgb_img.cpu().numpy()
        
        sample_values = PreparedSampleValues(
            pose=pose,
            shape=shape,
            style_vector=style_vector,
            garment_labels=clothed_visualizer.garment_classes.labels_vector,
            joints_3d=joints_3d,
            joints_conf=joints_conf,
            joints_2d=joints_2d,
            bbox=bbox
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
            samples_values: PreparedValuesArray,
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

        samples_values.append(sample_values)
        if sample_idx % self.CHECKPOINT_COUNT == 0 and sample_idx != 0:
            self._save_values(
                samples_values=samples_values, 
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

        garment_classes = GarmentClasses(upper_class, lower_class)
        clothed_visualizer = ClothedVisualizer(
            device=self.device,
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
                clothed_visualizer=clothed_visualizer
            )
            self._save_sample(
                dataset_dir=dataset_dir, 
                sample_idx=pose_idx, 
                rgb_img=rgb_img, 
                seg_maps=seg_maps, 
                sample_values=sample_values,
                samples_values=samples_values,
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
    args = parser.parse_args()
    
    surreal_generator = DataGenerator(
        preextract_kpt=args.preextract
    )
    #1 - sample set(s) of params

    #2 - generate dataset(s)
    '''
    surreal_generator.generate(
        gender=args.gender,
        upper_class=args.upper_class,
        lower_class=args.lower_class,
        num_samples_per_class=args.num_samples
    )
    '''
