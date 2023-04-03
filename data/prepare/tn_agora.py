# NOTE (kbartol): Adapted from MPI (original copyright and license below):

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
    Tuple, 
    Optional, 
    Union, 
    Iterator, 
    Any
)
from dataclasses import dataclass, fields
from pandas.core.series import Series
import argparse
import logging
import os
import pandas
from glob import glob
from tqdm import tqdm
import numpy as np
import pickle
import torch
import torch.cuda
import io
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import smplx
import itertools
from scipy.spatial.transform import Rotation as R

from configs.const import (
    AGORA_HEIGHT,
    AGORA_WIDTH
)
import configs.paths as paths
from data.prepare.common import (
    PreparedSampleValues,
    PreparedValuesArray
)
from data.prepare.common import DataGenerator
from models.smpl_official import easy_create_smpl_model
from predict.predict_hrnet import predict_hrnet
from utils.garment_classes import GarmentClasses


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)


def load_model(model_folder: str):
    return smplx.create(
        model_path=model_folder, 
        model_type='smpl',
        gender='neutral',
        ext='npz'
    )


def focalLength_mm2px(focalLength, dslr_sens, focalPoint):
    focal_pixel = (focalLength / dslr_sens) * focalPoint * 2
    return focal_pixel


def unreal2cv2(points):
    # x --> y, y --> z, z --> x
    points = np.roll(points, 2, 1)
    # change direction of y
    points = points * np.array([1, -1, 1])
    return points


def smpl2opencv(j3d):
    # change sign of axis 1 and axis 2
    j3d = j3d * np.array([1, -1, -1])
    return j3d


def project_point(joint, RT, KKK):

    P = np.dot(KKK, RT)
    joints_2d = np.dot(P, joint)
    joints_2d = joints_2d[0:2] / joints_2d[2]

    return joints_2d


def project_2d(
        img_name,
        debug,
        df_row,
        jdx,
        joints3d
    ) -> Tuple[Union[np.ndarray, None], Union[np.ndarray, None], Union[np.ndarray, None]]:

    dslr_sens_width = 36
    dslr_sens_height = 20.25

    img_path = os.path.join(paths.AGORA_RGB_DIR_PATH, img_name)
    if 'hdri' in img_path:
        ground_plane = [0, 0, 0]
        scene3d = False
        focalLength = 50
        camPosWorld = [0, 0, 170]
        camYaw = 0
        camPitch = 0
    elif 'cam00' in img_path:
        ground_plane = [0, 0, 0]
        scene3d = True
        focalLength = 18
        camPosWorld = [400, -275, 265]
        camYaw = 135
        camPitch = 30
    elif 'cam01' in img_path:
        ground_plane = [0, 0, 0]
        scene3d = True
        focalLength = 18
        camPosWorld = [400, 225, 265]
        camYaw = -135
        camPitch = 30
    elif 'cam02' in img_path:
        ground_plane = [0, 0, 0]
        scene3d = True
        focalLength = 18
        camPosWorld = [-490, 170, 265]
        camYaw = -45
        camPitch = 30
    elif 'cam03' in img_path:
        ground_plane = [0, 0, 0]
        scene3d = True
        focalLength = 18
        camPosWorld = [-490, -275, 265]
        camYaw = 45
        camPitch = 30
    elif 'ag2' in img_path:
        ground_plane = [0, 0, 0]
        scene3d = False
        focalLength = 28
        camPosWorld = [0, 0, 170]
        camYaw = 0
        camPitch = 15
    else:
        ground_plane = [0, -1.7, 0]
        scene3d = True
        focalLength = 28
        camPosWorld = [
            df_row['camX'],
            df_row['camY'],
            df_row['camZ']]
        camYaw = df_row['camYaw']
        camPitch = 0

    yawSMPL = df_row['Yaw'][jdx]
    trans3d = [df_row['X'][jdx],
                df_row['Y'][jdx],
                df_row['Z'][jdx]]

    bounding_box, gt2d, gt3d_camCoord = project2d(
        joints3d=joints3d, 
        focalLength=focalLength, 
        scene3d=scene3d,
        trans3d=trans3d,
        dslr_sens_width=dslr_sens_width,
        dslr_sens_height=dslr_sens_height,
        camPosWorld=camPosWorld,
        cy=AGORA_HEIGHT / 2,
        cx=AGORA_WIDTH / 2,
        imgPath=img_path,
        yawSMPL=yawSMPL,
        ground_plane=ground_plane,
        debug=debug,
        jdx=jdx,
        camPitch=camPitch, 
        camYaw=camYaw
    )
    return bounding_box, gt2d, gt3d_camCoord


def extract_bounding_box(joints_2d: np.ndarray) -> np.ndarray:
    """
    Bounding box extraction algorithm from 2D joint coordinates.

    First, get the minimum and maximum x and y coordinates of 2D joints.
    Then, get the middle (center) point of the bounding box as an average
    of all joint coordinates. The bounding box height and width are the
    differences between the maximum and minimum x and y coordinates, i.e.,
    the differences multiple by some constant (x1.3) in order to cut the
    image around the subject and not just exactly the keypoints part.
    The bounding box coordinates are calculated as the offset from the 
    middle (center) point. The offset is half of the bounding box height 
    (for both x and y coordinates because we want the bounding box to 
    be quadratic).

    Note that the bounding box might seem a bit more over the 
    top of the head than beneath the feet because the mean point is
    a bit above the middle of the body due to more keypoints being in
    the upper part of the body. It doesn't matter much for the model
    because I anyways do not use the same orthographic projection as
    in ClothSURREAL and will not estimate the "camera parameters" for
    the ClothAGORA, or might even completely remove that from the
    model outputs.
    """
    _, y_min = joints_2d.min(axis=0)
    _, y_max = joints_2d.max(axis=0)
    c_x, c_y = joints_2d.mean(axis=0)
    d_y = y_max - y_min
    h_y = d_y * 1.3

    x1, y1 = (c_x - h_y / 2, c_y - h_y / 2)
    x2, y2 = (c_x + h_y / 2, c_y + h_y / 2)

    return np.array([[x1, x2], [y1, y2]], dtype=np.float32)


def project2d(
        joints3d,
        focalLength,
        scene3d,
        trans3d,
        dslr_sens_width,
        dslr_sens_height,
        camPosWorld,
        cy,
        cx,
        img_path,
        yawSMPL,
        ground_plane,
        debug=False,
        jdx=-1,
        camPitch=0,
        camYaw=0
    ):

    focalLength_x = focalLength_mm2px(focalLength, dslr_sens_width, cx)
    focalLength_y = focalLength_mm2px(focalLength, dslr_sens_height, cy)

    camMat = np.array([[focalLength_x, 0, cx],
                       [0, focalLength_y, cy],
                       [0, 0, 1]])

    # camPosWorld and trans3d are in cm. Transform to meter
    trans3d = np.array(trans3d) / 100
    trans3d = unreal2cv2(np.reshape(trans3d, (1, 3)))
    camPosWorld = np.array(camPosWorld) / 100
    if scene3d:
        camPosWorld = unreal2cv2(
            np.reshape(
                camPosWorld, (1, 3))) + np.array(ground_plane)
    else:
        camPosWorld = unreal2cv2(np.reshape(camPosWorld, (1, 3)))

    # get points in camera coordinate system
    j3d = smpl2opencv(joints3d)

    # scans have a 90deg rotation, but for mean pose from vposer there is no
    # such rotation
    rotMat, _ = cv2.Rodrigues(
        np.array([[0, ((yawSMPL - 90) / 180) * np.pi, 0]], dtype=float))

    j3d = np.matmul(rotMat, j3d.T).T
    j3d = j3d + trans3d

    camera_rotationMatrix, _ = cv2.Rodrigues(
        np.array([0, ((-camYaw) / 180) * np.pi, 0]).reshape(3, 1))
    camera_rotationMatrix2, _ = cv2.Rodrigues(
        np.array([camPitch / 180 * np.pi, 0, 0]).reshape(3, 1))

    j3d_new = np.matmul(camera_rotationMatrix, j3d.T - camPosWorld.T).T
    j3d_new = np.matmul(camera_rotationMatrix2, j3d_new.T).T

    RT = np.concatenate((np.diag([1., 1., 1.]), np.zeros((3, 1))), axis=1)
    j2d = np.zeros((j3d_new.shape[0], 2))
    for i in range(j3d_new.shape[0]):
        j2d[i, :] = project_point(np.concatenate(
            [j3d_new[i, :], np.array([1])]), RT, camMat)
        
    bounding_box = extract_bounding_box(j2d)

    if debug:
        import matplotlib.cm as cm

        if len(j2d) < 200:  # No rendering for verts
            if not (img_path is None):
                img = cv2.imread(img_path)
                img = img[:, :, ::-1]
                colors = cm.tab20c(np.linspace(0, 1, 25)) # type: ignore
                fig = plt.figure(dpi=300)
                ax = fig.add_subplot(111)
                if not (img_path is None):
                    ax.imshow(img)
                for i in range(22):
                    ax.scatter(j2d[i, 0], j2d[i, 1], c=colors[i], s=0.1)
                    rect = patches.Rectangle(
                        xy=(bounding_box[0][0], bounding_box[1][0]), 
                        width=bounding_box[0][1] - bounding_box[0][0], 
                        height=bounding_box[1][1] - bounding_box[1][0], 
                        linewidth=1, 
                        edgecolor='r', 
                        facecolor='none'
                    )
                    ax.add_patch(rect)

                if not (img_path is None):
                    savename = img_path.split('/')[-1]
                    savename = savename.replace('.pkl', '.jpg')
                    plt.savefig(
                        os.path.join(
                            paths.AGORA_DEBUG_DIR_PATH,
                            'image' +
                            str(jdx) +
                            savename))
                    plt.close('all')

    return bounding_box, j2d, j3d_new


def get_smpl_vertices(
        gt,
        smpl_neutral,
        pose2rot=True):
    # All neutral model are used
    model_gt = smpl_neutral
    num_betas = 10

    # Since SMPLX to SMPL conversion tool store root_pose and translation as
    # keys
    if 'root_pose' in gt.keys():
        gt['global_orient'] = gt.pop('root_pose')
    if 'translation' in gt.keys():
        gt['transl'] = gt.pop('translation')
    for k, v in gt.items():
        if torch.is_tensor(v):
            gt[k] = v.detach().cpu().numpy()

    smpl_gt = model_gt(
        betas=torch.Tensor(gt['betas'][:, :num_betas], dtype=torch.float), #type:ignore
        global_orient=torch.Tensor(gt['global_orient'], dtype=torch.float), #type:ignore
        body_pose=torch.Tensor(gt['body_pose'], dtype=torch.float), #type:ignore
        transl=torch.Tensor(gt['transl'], dtype=torch.float), pose2rot=pose2rot #type:ignore
    )

    return smpl_gt.joints.detach().cpu().numpy().squeeze()


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
    

@dataclass
class DfData:

    """
    A class collecting all the dataframe data.
    """

    img_name: str
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


class AGORAPreparator(DataGenerator):

    """
    A data preparation class for ClothAGORA.
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
        self.data_dir = os.path.join(
            paths.DATA_ROOT_DIR,
            self.DATASET_NAME
        )
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
        img_dir = os.path.join(self.data_dir, gender, paths.RGB_DIR)
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        img_path = os.path.join(
            img_dir, 
            paths.IMG_NAME_TEMPLATE.format(idx=sample_idx)
        )
        cv2.imwrite(img_path, rgb_img)

        samples_values.append(sample_values)
        if sample_idx % self.CHECKPOINT_COUNT == 0 and sample_idx != 0:
            print(f'Saving values on checkpoint #{sample_idx}')
            DataGenerator._save_values(
                samples_values=samples_values,
                dataset_dir=os.path.join(
                    self.data_dir, 
                    gender
                )
            )

    def _get_projected_joints(
            self,
            gender,
            gt_3d_fit_path: str,
            df_row: Series,
            jdx: int
        ) -> Tuple[Union[np.ndarray, None], Union[np.ndarray, None], Union[np.ndarray, None]]:
        gt_3d_fit_path = os.path.join(
            paths.AGORA_FITS_DIR,
            df_row.at['gt_path_smpl'][jdx].replace('.obj', '.pkl')
        )
        if self.device == 'cpu':
            gt = CPU_Unpickler(open(gt_3d_fit_path, 'rb')).load()
        else:
            gt = pickle.load(open(gt_3d_fit_path, 'rb'))
        gt_joints_local = get_smpl_vertices(gt, self.smpl_model[gender])

        bounding_box, gt_joints_cam_2d, gt_joints_cam_3d = project_2d(df_row, jdx, gt_joints_local) # type: ignore
        return bounding_box, gt_joints_cam_2d, gt_joints_cam_3d
    
    @staticmethod
    def _collect_df_data(
            df_row: Series, 
            jdx: int
        ) -> DfData:
        img_name = os.path.join(paths.AGORA_IMG_DIR, df_row['imgPath'])
        smpl_gt_path = df_row.at['gt_path_smpl'][jdx]
        with open(smpl_gt_path, 'rb') as pkl_f:
            smpl_gt = pickle.load(pkl_f)
        pose_rot = smpl_gt['full_pose'][0].cpu().detach().numpy().from_matrix()
        pose = R.as_rotvec(pose_rot)
        shape = smpl_gt['betas'][0].cpu().detach().numpy()

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
        coordinates = AgoraCoordinates(
            cam_x=df_row['camX'],
            cam_y=df_row['camY'],
            cam_z=df_row['camZ'],
            trans_x=df_row['X'],
            trans_y=df_row['Y'],
            trans_z=df_row['Z']
        )
        return DfData(
            img_name=os.path.join(paths.AGORA_IMG_DIR, df_row['imgPath']),
            pose=R.as_rotvec(pose_rot),
            shape=smpl_gt['betas'][0].cpu().detach().numpy(),
            garment_classes=garment_classes,
            style_vector=style_vector,
            coordinates=coordinates
        )

    def prepare_sample(
            self,
            df_row: Series, 
            jdx: int
        ) -> Tuple[np.ndarray, PreparedSampleValues]:
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
            jdx=jdx)
        crop_coord, joints_2d, joints_3d = self._get_projected_joints(
            df_row=df_row,
            jdx=jdx
        )
        cam_t = np.zeros(shape=(3,), dtype=np.float32)

        img = cv2.imread(img_path)
        img_crop = img[crop_coord[0][0]:crop_coord[0][1], # type: ignore
                       crop_coord[1][0]:crop_coord[1][1]] # type: ignore

        bbox = None
        joints_conf = np.ones(joints_2d.shape[0]) #type:ignore
        if self.preextract_kpt:
            img_crop = torch.from_numpy(img).float().to(self.device)
            joints_2d, joints_conf, bbox = self._predict_joints(
                rgb_tensor=img_crop
            )
            img_crop = img_crop[0].cpu().numpy()

        sample_values = PreparedSampleValues(
            pose=pose,
            shape=shape,
            style_vector=style_vector,
            garment_labels=garment_classes.labels_vector,
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
            df: pandas.DataFrame
        ) -> None:
        """
        (Pre-)generate the dataset for particular upper+lower garment class.

        The generated samples are stored in files (RGB images, seg maps),
        i.e., update in the corresponding sample values' arrays. The sample
        arrays are frequently updated on the disk in case the failure
        happens along the way.
        """
        samples_values, num_generated = DataGenerator._create_values_array(
            dataset_dir=self.data_dir
        )
        scene_pkl_files = os.listdir(paths.AGORA_PKL_DIR)
        samples_counter = 0
        for df_iter, df_path in tqdm(enumerate(scene_pkl_files)):
            logging.info(f'Preparing {df_iter}/{len(scene_pkl_files)} df...')
            df = pandas.read_pickle(df_path)
            for idx in tqdm(range(len(df))):
                for jdx, _gender in enumerate(df.iloc[idx]['gender']):
                    if _gender != gender:
                        continue
                    if samples_counter < num_generated: # NOTE: To avoid regeneration.
                        samples_counter += 1

                    img_crop, sample_values = self.prepare_sample(
                        df_row=df.iloc[idx], 
                        jdx=jdx
                    )
                    self._save_sample(
                        sample_idx=samples_counter, 
                        rgb_img=img_crop, 
                        seg_maps=None,  # NOTE: Might do with pretrained cloth seg.
                        gender=gender,
                        sample_values=sample_values,
                        samples_values=samples_values
                    )
                    samples_counter += 1


if __name__ == '__main__':
    agora_preparator = AGORAPreparator()
    agora_preparator.prepare(
        gender='male'
    )
