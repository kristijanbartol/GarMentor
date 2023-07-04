# NOTE (kbartol): Adapted from MPI (original copyright and license below)
# NOTE (kbartol): I should not publish this based on a more detailed AGORA license.

#------------------------------------------------------------------------------
# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2021 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de
#------------------------------------------------------------------------------
from typing import (
    Any,
    Dict,
    Tuple, 
    Optional, 
    Union, 
    Iterator
)
from pandas.core.series import Series
import logging
import os
import pandas
from tqdm import tqdm
import numpy as np
import pickle
import torch
import torch.cuda
import io
import cv2
from scipy.spatial.transform import Rotation as R
import sys
import argparse

sys.path.append('/garmentor')

from configs.const import (
    to_resolution_str,
    RESOLUTION,
    PREP_CROP_SIZE
)
import configs.paths as paths
from data.prepare.common import (
    PreparedSampleValues,
    PreparedValuesArray
)
from data.prepare.common import DataGenerator
from models.smpl_official import (
    easy_create_smpl_model,
    SMPL
)
from utils.garment_classes import GarmentClasses
from utils.joints2d_utils import extract_crop_coordinates


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)


@dataclass
class AgoraCoordinates:

    """
    An AGORA camera class (structure).
    """

    cam_x: float                      
    cam_y: float
    cam_z: float
    cam_yaw: float
    trans_x: float
    trans_y: float
    trans_z: float
    yaw: float
    height: int
    width: int

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


def focalLength_mm2px(focalLength, dslr_sens, focalPoint):
    focal_pixel = (focalLength / dslr_sens) * focalPoint * 2
    return focal_pixel


def unreal2cv2(points):
    # x --> y, y --> z, z --> x
    points = np.roll(points, 2, 1)
    # change direction of y
    points = points * np.array([1, -1, 1])
    return points


def smpl2opencv(joints_3d: np.ndarray) -> np.ndarray:
    # change sign of axis 1 and axis 2
    joints_3d = joints_3d * np.array([1, -1, -1])
    return joints_3d


def project_point(joint, RT, KKK):
    P = np.dot(KKK, RT)
    joints_2d = np.dot(P, joint)
    joints_2d = joints_2d[0:2] / joints_2d[2]

    return joints_2d


def project_2d(
        img_path: str,
        joints_3d: np.ndarray,
        agora_coords: AgoraCoordinates
    ) -> Tuple[np.ndarray, bool, np.ndarray, np.ndarray]:

    dslr_sens_width = 36
    dslr_sens_height = 20.25

    if 'hdri' in img_path:
        ground_plane = [0, 0, 0]
        scene_3d = False
        focal_length = 50
        cam_pos_world = [0, 0, 170]
        cam_yaw = 0
        cam_pitch = 0
    elif 'cam00' in img_path:
        ground_plane = [0, 0, 0]
        scene_3d = True
        focal_length = 18
        cam_pos_world = [400, -275, 265]
        cam_yaw = 135
        cam_pitch = 30
    elif 'cam01' in img_path:
        ground_plane = [0, 0, 0]
        scene_3d = True
        focal_length = 18
        cam_pos_world = [400, 225, 265]
        cam_yaw = -135
        cam_pitch = 30
    elif 'cam02' in img_path:
        ground_plane = [0, 0, 0]
        scene_3d = True
        focal_length = 18
        cam_pos_world = [-490, 170, 265]
        cam_yaw = -45
        cam_pitch = 30
    elif 'cam03' in img_path:
        ground_plane = [0, 0, 0]
        scene_3d = True
        focal_length = 18
        cam_pos_world = [-490, -275, 265]
        cam_yaw = 45
        cam_pitch = 30
    elif 'ag2' in img_path:
        ground_plane = [0, 0, 0]
        scene_3d = False
        focal_length = 28
        cam_pos_world = [0, 0, 170]
        cam_yaw = 0
        cam_pitch = 15
    else:
        ground_plane = [0, -1.7, 0]
        scene_3d = True
        focal_length = 28
        cam_pos_world = [
            agora_coords.cam_x,
            agora_coords.cam_y,
            agora_coords.cam_z
        ]
        cam_yaw = agora_coords.cam_yaw
        cam_pitch = 0

    trans_3d = [
        agora_coords.trans_x,
        agora_coords.trans_y,
        agora_coords.trans_z
    ]

    cx = agora_coords.width / 2
    cy = agora_coords.height / 2

    focalLength_x = focalLength_mm2px(
        focal_length, 
        dslr_sens_width, 
        agora_coords.width / 2
    )
    focalLength_y = focalLength_mm2px(
        focal_length, 
        dslr_sens_height, 
        agora_coords.height / 2
    )

    cam_mat = np.array([[focalLength_x, 0, cx],
                       [0, focalLength_y, cy],
                       [0, 0, 1]])

    # cam_pos_world and trans3d are in cm. Transform to meter
    trans_3d = np.array(trans_3d) / 100
    trans_3d = unreal2cv2(np.reshape(trans_3d, (1, 3)))
    cam_pos_world = np.array(cam_pos_world) / 100
    if scene_3d:
        cam_pos_world = unreal2cv2(
            np.reshape(
                cam_pos_world, (1, 3))) + np.array(ground_plane)
    else:
        cam_pos_world = unreal2cv2(np.reshape(cam_pos_world, (1, 3)))

    # get points in camera coordinate system
    joints_3d = smpl2opencv(joints_3d)

    # scans have a 90deg rotation, but for mean pose from vposer there is no
    # such rotation
    rotMat, _ = cv2.Rodrigues(
        np.array([[0, ((agora_coords.yaw - 90) / 180) * np.pi, 0]], dtype=float))

    joints_3d = np.matmul(rotMat, joints_3d.T).T
    joints_3d = joints_3d + trans_3d

    camera_rotationMatrix, _ = cv2.Rodrigues(
        np.array([0, ((-cam_yaw) / 180) * np.pi, 0]).reshape(3, 1)) #type:ignore
    camera_rotationMatrix2, _ = cv2.Rodrigues(
        np.array([cam_pitch / 180 * np.pi, 0, 0]).reshape(3, 1))

    joints_3d = np.matmul(camera_rotationMatrix, joints_3d.T - cam_pos_world.T).T #type:ignore
    joints_3d = np.matmul(camera_rotationMatrix2, joints_3d.T).T

    RT = np.concatenate((np.diag([1., 1., 1.]), np.zeros((3, 1))), axis=1)
    joints_2d = np.zeros((joints_3d.shape[0], 2))
    for i in range(joints_3d.shape[0]):
        joints_2d[i, :] = project_point(np.concatenate(
            [joints_3d[i, :], np.array([1])]), RT, cam_mat)
        
    crop_coords, crop_success = extract_crop_coordinates(
        joints_2d=joints_2d,
        h=agora_coords.height,
        w=agora_coords.width
    )
    return crop_coords, crop_success, joints_2d, joints_3d
    

@dataclass
class DfData:

    """
    A class collecting all the dataframe data.
    """

    img_path: str
    gt_3d_fit_path: str
    pose: np.ndarray                      
    shape: np.ndarray
    garment_classes: GarmentClasses
    style_vector: np.ndarray
    coordinates: AgoraCoordinates

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


class Cat3DPreparator(DataGenerator):

    """
    A data preparation class for CAT-3D<TN>.
    """

    DATASET_NAME = 'agora'
    CHECKPOINT_COUNT = 100

    def __init__(
            self,
            device='cuda:0',
            preextract_kpt=False
        ) -> None:
        """
        Initialize superclass and create clothed renderer.
        """
        super().__init__(preextract_kpt=preextract_kpt)
        self.device = device
        self.smpl_models = {
            'male': easy_create_smpl_model(
                gender='male',
                device=device
            ),
            'female': easy_create_smpl_model(
                gender='female',
                device=device
            )
        }

    @staticmethod
    def _collect_df_data(
            df_row: Series, 
            jdx: int,
            scene_name: str,
            resolution_label: str
        ) -> DfData:
        """
        Read dataframe data prepared for TN-AGORA and store into a structure.
        """
        gt_3d_fit_path = os.path.join(
            paths.AGORA_FITS_DIR,
            df_row.at['gt_path_smpl'][jdx].replace('.obj', '.pkl')
        )
        with open(gt_3d_fit_path, 'rb') as pkl_f:
            gt_3d_fit = pickle.load(pkl_f)

        garment_combination = df_row.at['garment_combinations'][jdx]
        garment_styles = df_row.at['garment_styles'][jdx]
        garment_classes = GarmentClasses(
            upper_class=garment_combination['upper'],
            lower_class=garment_combination['lower']
        )
        style_vector = garment_classes.to_style_vector(
            upper_style=garment_styles['upper'],
            lower_style=garment_styles['lower']
        )
        return DfData(
            img_path=os.path.join(
                paths.AGORA_FULL_IMG_DIR_TEMPLATE.format(
                    scene_name=scene_name,
                    resolution=to_resolution_str(resolution_label)
                ), 
                df_row['imgPath']
            ),
            gt_3d_fit_path=gt_3d_fit_path,
            pose=R.from_matrix(
                gt_3d_fit['full_pose'][0].cpu().detach().numpy()
            ).as_rotvec(),
            shape=gt_3d_fit['betas'][0].cpu().detach().numpy(),
            garment_classes=GarmentClasses(
                upper_class=garment_combination['upper'],
                lower_class=garment_combination['lower']
            ),
            style_vector=style_vector,
            coordinates=AgoraCoordinates(
                cam_x=df_row['camX'],
                cam_y=df_row['camY'],
                cam_z=df_row['camZ'],
                cam_yaw=df_row['camYaw'],
                trans_x=df_row['X'][jdx],
                trans_y=df_row['Y'][jdx],
                trans_z=df_row['Z'][jdx],
                yaw=df_row['Yaw'][jdx],
                height=RESOLUTION[resolution_label][0],
                width=RESOLUTION[resolution_label][1]
            )
        )

    @staticmethod
    def _crop_image(
            img_path: str,
            crop_coords: np.ndarray,
            resolution_label: str
    ) -> np.ndarray:
        img = cv2.imread(img_path)
        cropped_img = img[crop_coords[0][0]:crop_coords[0][1],
                          crop_coords[1][0]:crop_coords[1][1]]
        resized_img = cv2.resize(
            src=cropped_img, 
            dsize=tuple(PREP_CROP_SIZE[resolution_label]),
            interpolation=cv2.INTER_AREA
        )
        return resized_img

    def _save_sample(
            self, 
            sample_idx: int, 
            rgb_img: np.ndarray, 
            gender: str,
            sample_values: PreparedSampleValues,
            samples_values: PreparedValuesArray,
            seg_maps: Optional[np.ndarray] = None   # NOTE: Might experiment later.
        ) -> None:
        """
        Save RGB and seg maps to disk, and update the values in the array (RAM).
        """
        img_dir = os.path.join(
            paths.AGORA_PREPARED_DIR, 
            gender, 
            paths.RGB_DIR
        )
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        img_path = os.path.join(
            img_dir, 
            paths.IMG_NAME_TEMPLATE.format(sample_idx=sample_idx)
        )
        cv2.imwrite(img_path, rgb_img)

        samples_values.append(sample_values)
        if sample_idx % self.CHECKPOINT_COUNT == 0 and sample_idx != 0:
            print(f'Saving values on checkpoint #{sample_idx}')
            DataGenerator._save_values(
                samples_values=samples_values,
                dataset_dir=os.path.join(
                    paths.AGORA_PREPARED_DIR, 
                    gender
                ),
                sample_idx=sample_idx
            )

    def _get_3d_joints(
            self,
            gt_3d_fit: Dict[str, torch.Tensor],
            smpl_model: SMPL,
            pose2rot: bool = True
        ) -> np.ndarray:
        # Since SMPLX to SMPL conversion tool store root_pose and translation as
        # keys
        if 'root_pose' in gt_3d_fit.keys():
            gt_3d_fit['global_orient'] = gt_3d_fit.pop('root_pose')
        if 'translation' in gt_3d_fit.keys():
            gt_3d_fit['transl'] = gt_3d_fit.pop('translation')
        for k, v in gt_3d_fit.items():
            if torch.is_tensor(v):
                gt_3d_fit[k] = v.detach().cpu().numpy()

        model_output = smpl_model(
            betas=torch.Tensor(gt_3d_fit['betas'][:, :10]).float().to(self.device), #type:ignore
            global_orient=torch.Tensor(gt_3d_fit['global_orient']).float().to(self.device), #type:ignore
            body_pose=torch.Tensor(gt_3d_fit['body_pose']).float().to(self.device), #type:ignore
            transl=torch.Tensor(gt_3d_fit['transl']).float().to(self.device), #type:ignore
            pose2rot=pose2rot
        )
        return model_output.joints.detach().cpu().numpy().squeeze()

    def _get_projected_joints(
            self,
            gender: str,
            img_path: str,
            gt_3d_fit_path: str,
            agora_coords: AgoraCoordinates
        ) -> Tuple[np.ndarray, bool, np.ndarray, np.ndarray]:
        if self.device == 'cpu':
            gt_3d_fit = CPU_Unpickler(open(gt_3d_fit_path, 'rb')).load()
        else:
            gt_3d_fit = pickle.load(open(gt_3d_fit_path, 'rb'))

        joints_3d = self._get_3d_joints(
            gt_3d_fit=gt_3d_fit, 
            smpl_model=self.smpl_models[gender]
        )
        crop_coords, crop_success, gt_joints_cam_2d, gt_joints_cam_3d = project_2d(
            img_path=img_path,
            joints_3d=joints_3d,
            agora_coords=agora_coords
        )
        return crop_coords, crop_success, gt_joints_cam_2d, gt_joints_cam_3d

    def prepare_sample(
            self,
            gender: str,
            resolution_label: str,
            scene_name: str,
            df_row: Series, 
            jdx: int
        ) -> Tuple[Union[np.ndarray, None], Union[PreparedSampleValues, None]]:
        """
        Create an array of prepared values, which consists of:
        - pose parameters
        - shape parameters
        - style vector
        - garment labels
        - camera translations (dummy value for AGORA-like data)
        - 3D joints
        """
        df_data = self._collect_df_data(
            df_row=df_row, 
            jdx=jdx,
            scene_name=scene_name,
            resolution_label=resolution_label
        )
        crop_coords, crop_success, joints_2d, joints_3d = self._get_projected_joints(
            gender=gender,
            img_path=df_data.img_path,
            gt_3d_fit_path=df_data.gt_3d_fit_path,
            agora_coords=df_data.coordinates
        )
        if not crop_success:
            return None, None
        cam_t = np.zeros(shape=(3,), dtype=np.float32)
        img_crop = self._crop_image(
            img_path=df_data.img_path,
            crop_coords=crop_coords,
            resolution_label=resolution_label
        )
        bbox = None
        joints_conf = np.ones(joints_2d.shape[0]) #type:ignore
        if self.preextract_kpt:
            img_crop = torch.from_numpy(img_crop).float().to(self.device)
            joints_2d, joints_conf, bbox = self._predict_joints(
                rgb_tensor=img_crop
            )
            img_crop = img_crop[0].cpu().numpy()

        sample_values = PreparedSampleValues(
            pose=df_data.pose,
            shape=df_data.shape,
            style_vector=df_data.style_vector,
            garment_labels=df_data.garment_classes.labels_vector,
            cam_t=cam_t,
            joints_3d=joints_3d, #type:ignore
            joints_conf=joints_conf, #type:ignore
            joints_2d=joints_2d, #type:ignore
            bbox=bbox #type:ignore
        )
        return img_crop, sample_values

    def prepare(
            self,
            gender: str,
            resolution_label: str     # 'normal' or 'high'
        ) -> None:
        """
        (Pre-)generate the dataset for particular upper+lower garment class.

        The generated samples are stored in files (RGB images, seg maps),
        i.e., update in the corresponding sample values' arrays. The sample
        arrays are frequently updated on the disk in case the failure
        happens along the way.
        """
        samples_values, num_generated = DataGenerator._create_values_array(
            dataset_dir=os.path.join(
                paths.AGORA_PREPARED_DIR,
                gender
            )
        )
        scene_pkl_file_names = os.listdir(paths.AGORA_GT_DIR)
        samples_counter = 0
        for df_idx, df_file_name in tqdm(enumerate(scene_pkl_file_names)):
            logging.info(f'Preparing {df_idx}/{len(scene_pkl_file_names)} df...')
            df = pandas.read_pickle(os.path.join(
                paths.AGORA_GT_DIR,
                df_file_name)
            )
            scene_name = df_file_name.split('.')[0]
            for idx in tqdm(range(len(df))):
                for jdx, _gender in enumerate(df.iloc[idx]['gender']):
                    if _gender != gender or df.iloc[idx]['kid'][jdx] == True:
                        continue
                    if samples_counter < num_generated: # NOTE: To avoid regeneration.
                        samples_counter += 1

                    img_crop, sample_values = self.prepare_sample(
                        gender=gender,
                        scene_name=scene_name,
                        resolution_label=resolution_label,
                        df_row=df.iloc[idx], 
                        jdx=jdx
                    )
                    if img_crop is None and sample_values is None:
                        continue
                    self._save_sample(
                        sample_idx=samples_counter, 
                        rgb_img=img_crop, #type:ignore
                        seg_maps=None,  # NOTE: Might do with pretrained cloth seg.
                        gender=gender,
                        sample_values=sample_values, #type:ignore
                        samples_values=samples_values
                    )
                    samples_counter += 1


class Cat3DTestPreparator(Cat3DPreparator):

    def __init__(self):
        super().__init__(preextract_kpt=False)

    def prepare(
            self,
            gender: str,
            resolution_label: str = 'high'     # 'normal' or 'high'
        ) -> None:
        """
        (Pre-)generate the dataset for particular upper+lower garment class.

        The generated samples are stored in files (RGB images, seg maps),
        i.e., update in the corresponding sample values' arrays. The sample
        arrays are frequently updated on the disk in case the failure
        happens along the way.
        """
        samples_values, num_generated = DataGenerator._create_values_array(
            dataset_dir=os.path.join(
                paths.AGORA_PREPARED_DIR,
                gender
            )
        )
        scene_pkl_file_names = os.listdir(paths.AGORA_GT_DIR)
        samples_counter = 0
        for df_idx, df_file_name in tqdm(enumerate(scene_pkl_file_names)):
            logging.info(f'Preparing {df_idx}/{len(scene_pkl_file_names)} df...')
            df = pandas.read_pickle(os.path.join(
                paths.AGORA_GT_DIR,
                df_file_name)
            )
            scene_name = df_file_name.split('.')[0]
            for idx in tqdm(range(len(df))):
                for jdx, _gender in enumerate(df.iloc[idx]['gender']):
                    if _gender != gender or df.iloc[idx]['kid'][jdx] == True:
                        continue
                    if samples_counter < num_generated: # NOTE: To avoid regeneration.
                        samples_counter += 1

                    img_crop, sample_values = self.prepare_sample(
                        gender=gender,
                        scene_name=scene_name,
                        resolution_label=resolution_label,
                        df_row=df.iloc[idx], 
                        jdx=jdx
                    )
                    if img_crop is None and sample_values is None:
                        continue
                    self._save_sample(
                        sample_idx=samples_counter, 
                        rgb_img=img_crop, #type:ignore
                        seg_maps=None,  # NOTE: Might do with pretrained cloth seg.
                        gender=gender,
                        sample_values=sample_values, #type:ignore
                        samples_values=samples_values
                    )
                    samples_counter += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_split', '-D', type=str, choices=['train', 'test'],
                        help='data split for which to prepare data')
    args = parser.parse_args()

    if args.data_split == 'train':
        agora_preparator = Cat3DPreparator(
            preextract_kpt=False
        )
    else:
        agora_preparator = Cat3DTestPreparator()
        
    agora_preparator.prepare(
        gender='male',
        resolution_label='normal'
    )
