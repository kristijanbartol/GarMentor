from typing import Tuple
import os
import numpy as np
import torch

from configs.const import DIG_SURREAL_DATASET_NAME
from data.prepare.common import DataGenerator

import configs.paths as paths
from data.prepare.common import (
    PreparedSampleValues,
    PreparedValuesArray
)
from predict.predict_hrnet import predict_hrnet
from utils.garment_classes import GarmentClasses
from vis.visualizers.tn_clothed import TnClothedVisualizer
from vis.visualizers.keypoints import KeypointsVisualizer

from DIG_for_garmentor.networks import IGR, lbs_mlp, learnt_representations
from DIG_for_garmentor.smpl_pytorch.smpl_server import SMPLServer
from DIG_for_garmentor.utils.deform import rotate_root_pose_x, infer



class DigDataGenerator(DataGenerator):

    """
    A data generation class specific to DIG-SURREAL dataset.
    """

    DATASET_NAME = DIG_SURREAL_DATASET_NAME
    CHECKPOINT_COUNT = 25

    def __init__(
            self,
            preextract_kpt=False
        ) -> None:
        """
        Initialize superclass and create clothed renderer.
        """
        super().__init__(preextract_kpt=preextract_kpt)
        self.device = 'cuda:0'

    def generate_sample(
            self, 
            idx: int,
            gender: str,
            clothed_visualizer: TnClothedVisualizer
        ) -> Tuple[np.ndarray, np.ndarray, PreparedSampleValues]:
        """
        Generate a single training sample.
        """
        pose, shape, style_vector, cam_t = self.generate_random_params(idx)

        print(f'Sample #{idx} ({gender}):')
        print(f'\tPose: {pose}')
        print(f'\tShape: {shape}')
        print(f'\tCam T: {cam_t}')
        print(f'\tStyle: {style_vector}')

        rgb_img, seg_maps, joints_3d = clothed_visualizer.vis_from_params(
            pose=pose,
            shape=shape,
            style_vector=style_vector,
            cam_t=cam_t,     # TODO: Might remove cam_t as a parameter here.
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
            cam_t=cam_t,
            bbox=bbox
        )
        return (
            rgb_img,
            seg_maps, 
            sample_values
        )
    
    def _save_sample(
            self, 
            dataset_dir: str, 
            sample_idx: int, 
            rgb_img: np.ndarray, 
            seg_maps: np.ndarray, 
            sample_values: PreparedSampleValues,
            samples_values: PreparedValuesArray,
            clothed_visualizer: TnClothedVisualizer
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
    
    def _create_values_array(self, dataset_dir: str
                             ) -> Tuple[PreparedValuesArray, int]:
        """
        Create an array of prepared values, which consists of:
          - pose parameters
          - shape parameters
          - style vector
          - garment labels
          - camera translations (deprecated - will remove)
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
    
    def _log_class(self, 
                   gender: str, 
                   num_samples: int,
                   num_generated: int):
        total_num_samples = num_samples
        subset_str = f'{gender}-{upper_class}-{lower_class}'
        num_samples_to_generate = total_num_samples - num_generated
        print(f'Generating {total_num_samples}-{num_generated}='
              f'{num_samples_to_generate} samples for {subset_str}...')
        return total_num_samples
    
    def generate(self, 
                 gender: str,
                 num_samples: int
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
        clothed_visualizer = TnClothedVisualizer(
            device=self.device,
            gender=gender
        )
        dataset_dir = self.dataset_path_template.format(
            dataset_name=self.DATASET_NAME,
            gender=gender
        )  
        samples_values, num_generated = self._create_values_array(
            dataset_dir=dataset_dir
        )
        total_num_samples = self._log_class(
            gender=gender, 
            num_samples=num_samples,
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
