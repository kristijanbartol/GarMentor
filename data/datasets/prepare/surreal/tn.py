from typing import Tuple
import numpy as np
import torch
import os
import sys
import argparse

_module_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(_module_dir))

from data.datasets.prepare.common import (
    PreparedSampleValues,
    PreparedValuesArray
)
from data.datasets.prepare.common import DataGenerator
from utils.garment_classes import GarmentClasses
from vis.visualizers.clothed import ClothedVisualizer


class SurrealDataGenerator(DataGenerator):

    """
    A data generation class specific to SURREAL dataset.
    """

    DATASET_NAME = 'surreal'
    CHECKPOINT_COUNT = 100

    def __init__(self):
        """
        Initialize superclass and create clothed renderer.
        """
        super().__init__()

    def generate_sample(self, 
                        idx: int,
                        gender: str,
                        clothed_visualizer: ClothedVisualizer
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
            cam_t=cam_t     # TODO: Might remove cam_t as a parameter here.
        )
        sample_values = PreparedSampleValues(
            pose=pose,
            shape=shape,
            style_vector=style_vector,
            garment_labels=clothed_visualizer.garment_classes.labels_vector,
            cam_t=cam_t,
            joints=joints_3d
        )
        return rgb_img, seg_maps, sample_values
    
    @staticmethod
    def _create_dirs(
            dataset_dir: str, 
            img_dirname: str, 
            seg_dirname: str
        ) -> None:
        """
        Create image and segmentation mask directories.
        """
        img_dir = os.path.join(dataset_dir, img_dirname)
        seg_dir = os.path.join(dataset_dir, seg_dirname)
        
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        if not os.path.exists(seg_dir):
            os.makedirs(seg_dir)

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
            img_dirname=self.IMG_DIR,
            seg_dirname=self.SEG_MAPS_DIR
        )
        if rgb_img is not None:
            img_dir = os.path.join(
                dataset_dir, 
                self.IMG_DIR
            )
            img_path = os.path.join(
                img_dir, 
                self.IMG_NAME_TEMPLATE.format(idx=sample_idx)
            )
            clothed_visualizer.save_vis(
                img=rgb_img,
                save_path=img_path
            )
        seg_dir = os.path.join(dataset_dir, self.SEG_MAPS_DIR)
        seg_path = os.path.join(
            seg_dir, self.SEG_MAPS_NAME_TEMPLATE.format(idx=sample_idx))
        clothed_visualizer.save_masks(
            seg_masks=seg_maps,
            save_path=seg_path
        )

        samples_values.append(sample_values)
        if sample_idx % self.CHECKPOINT_COUNT == 0 and sample_idx != 0:
            print(f'Saving values on checkpoint #{sample_idx}')
            self._save_values(samples_values, dataset_dir)
            
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
                clothed_visualizer=clothed_visualizer)
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
    
    surreal_generator = SurrealDataGenerator()
    surreal_generator.generate(
        gender=args.gender,
        upper_class=args.upper_class,
        lower_class=args.lower_class,
        num_samples_per_class=args.num_samples
    )
