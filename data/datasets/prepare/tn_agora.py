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
from typing import List, Tuple
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
import io
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import smplx
import itertools
from pytorch3d.transforms import matrix_to_axis_angle

from configs.paths import (
    AGORA_IMG_DIR,
    AGORA_PKL_DIR,
    AGORA_RGB_DIR,
    AGORA_VALUES_PATH
)
from data.datasets.prepare.common import (
    PreparedSampleValues,
    PreparedValuesArray
)
from data.datasets.prepare.common import DataGenerator
from utils.garment_classes import GarmentClasses


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)


def load_model(args):

    if args.modeltype == 'SMPLX' and args.pose2rot:
        model_male = smplx.create(args.modelFolder, model_type='smplx',
                                  gender='male',
                                  ext='npz',
                                  num_betas=args.numBetas, use_pca=False)
        model_male_kid = smplx.create(args.modelFolder, model_type='smplx',
                                      gender='male',
                                      age='kid',
                                      kid_template_path=args.kid_template_path,
                                      ext='npz', use_pca=False)

        model_female = smplx.create(args.modelFolder, model_type='smplx',
                                    gender='female',
                                    ext='npz',
                                    num_betas=args.numBetas,
                                    use_pca=False)

        model_female_kid = smplx.create(
            args.modelFolder,
            model_type='smplx',
            gender='female',
            age='kid',
            kid_template_path=args.kid_template_path,
            ext='npz',
            use_pca=False)

        model_neutral = smplx.create(args.modelFolder, model_type='smplx',
                                     gender='neutral',
                                     ext='npz',
                                     num_betas=args.numBetas,
                                     use_pca=False)

        model_neutral_kid = smplx.create(
            args.modelFolder,
            model_type='smplx',
            gender='neutral',
            age='kid',
            kid_template_path=args.kid_template_path,
            ext='npz',
            use_pca=False)

    elif args.modeltype == 'SMPLX' and not args.pose2rot:
        # If params are in rotation matrix format then we need to use SMPLXLayer class
        model_male = smplx.build_layer(args.modelFolder, model_type='smplx',
                                  gender='male',
                                  ext='npz',
                                  num_betas=args.numBetas, use_pca=False)
        model_male_kid = smplx.build_layer(args.modelFolder, model_type='smplx',
                                      gender='male',
                                      age='kid',
                                      kid_template_path=args.kid_template_path,
                                      ext='npz', use_pca=False)

        model_female = smplx.build_layer(args.modelFolder, model_type='smplx',
                                    gender='female',
                                    ext='npz',
                                    num_betas=args.numBetas,
                                    use_pca=False)

        model_female_kid = smplx.build_layer(
            args.modelFolder,
            model_type='smplx',
            gender='female',
            age='kid',
            kid_template_path=args.kid_template_path,
            ext='npz',
            use_pca=False)

        model_neutral = smplx.build_layer(args.modelFolder, model_type='smplx',
                                     gender='neutral',
                                     ext='npz',
                                     num_betas=args.numBetas,
                                     use_pca=False)

        model_neutral_kid = smplx.build_layer(
            args.modelFolder,
            model_type='smplx',
            gender='neutral',
            age='kid',
            kid_template_path=args.kid_template_path,
            ext='npz',
            use_pca=False)

    elif args.modeltype == 'SMPL':
        model_male = smplx.create(args.modelFolder, model_type='smpl',
                                  gender='male',
                                  ext='npz')
        model_male_kid = smplx.create(args.modelFolder, model_type='smpl',
                                      gender='male', age='kid',
                                      kid_template_path=args.kid_template_path,
                                      ext='npz')
        model_female = smplx.create(args.modelFolder, model_type='smpl',
                                    gender='female',
                                    ext='npz')
        model_female_kid = smplx.create(
            args.modelFolder,
            model_type='smpl',
            gender='female',
            age='kid',
            kid_template_path=args.kid_template_path,
            ext='npz')
        model_neutral = smplx.create(args.modelFolder, model_type='smpl',
                                     gender='neutral',
                                     ext='npz')
        model_neutral_kid = smplx.create(
            args.modelFolder,
            model_type='smpl',
            gender='neutral',
            age='kid',
            kid_template_path=args.kid_template_path,
            ext='npz')
    else:
        raise ValueError('Provide correct modeltype smpl/smplx')
    return model_male, model_male_kid, model_female, model_female_kid, model_neutral, model_neutral_kid


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
        args,
        df_row,
        i,
        jdx,
        joints3d,
        meanPose=False):

    dslr_sens_width = 36
    dslr_sens_height = 20.25
    imgWidth = args.imgWidth
    imgHeight = args.imgHeight
    debug_path = args.debug_path
    imgBase = args.imgFolder
    imgName = df_row['imgPath']
    if imgWidth == 1280 and '_1280x720.png' not in imgName:
        #If 1280x720 images are used then image name needs to be updated
        imgName = imgName.replace('.png','_1280x720.png')
        df_row['imgPath']=imgName

    imgPath = os.path.join(imgBase, df_row['imgPath'])
    if not os.path.exists(imgPath):
        logging.error('Image not found: %s' % imgPath)
        return None, None
    if 'hdri' in imgPath:
        ground_plane = [0, 0, 0]
        scene3d = False
        focalLength = 50
        camPosWorld = [0, 0, 170]
        camYaw = 0
        camPitch = 0

    elif 'cam00' in imgPath:
        ground_plane = [0, 0, 0]
        scene3d = True
        focalLength = 18
        camPosWorld = [400, -275, 265]
        camYaw = 135
        camPitch = 30
    elif 'cam01' in imgPath:
        ground_plane = [0, 0, 0]
        scene3d = True
        focalLength = 18
        camPosWorld = [400, 225, 265]
        camYaw = -135
        camPitch = 30
    elif 'cam02' in imgPath:
        ground_plane = [0, 0, 0]
        scene3d = True
        focalLength = 18
        camPosWorld = [-490, 170, 265]
        camYaw = -45
        camPitch = 30
    elif 'cam03' in imgPath:
        ground_plane = [0, 0, 0]
        scene3d = True
        focalLength = 18
        camPosWorld = [-490, -275, 265]
        camYaw = 45
        camPitch = 30
    elif 'ag2' in imgPath:
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

    if meanPose:
        yawSMPL = 0
        trans3d = [0, 0, 0]
    else:
        yawSMPL = df_row['Yaw'][jdx]
        trans3d = [df_row['X'][jdx],
                   df_row['Y'][jdx],
                   df_row['Z'][jdx]]

    bounding_box, gt2d, gt3d_camCoord = project2d(joints3d, focalLength=focalLength, scene3d=scene3d,
                                    trans3d=trans3d,
                                    dslr_sens_width=dslr_sens_width,
                                    dslr_sens_height=dslr_sens_height,
                                    camPosWorld=camPosWorld,
                                    cy=imgHeight / 2,
                                    cx=imgWidth / 2,
                                    imgPath=imgPath,
                                    yawSMPL=yawSMPL,
                                    ground_plane=ground_plane,
                                    debug_path=debug_path,
                                    debug=args.debug,
                                    jdx=jdx,
                                    meanPose=meanPose, camPitch=camPitch, camYaw=camYaw)
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
        j3d,
        focalLength,
        scene3d,
        trans3d,
        dslr_sens_width,
        dslr_sens_height,
        camPosWorld,
        cy,
        cx,
        imgPath,
        yawSMPL,
        ground_plane,
        debug_path,
        debug=False,
        jdx=-1,
        meanPose=False,
        camPitch=0,
        camYaw=0):

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
    j3d = smpl2opencv(j3d)

    # scans have a 90deg rotation, but for mean pose from vposer there is no
    # such rotation
    if meanPose:
        rotMat, _ = cv2.Rodrigues(
            np.array([[0, (yawSMPL) / 180 * np.pi, 0]], dtype=float))
    else:
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
        if not os.path.exists(debug_path):
            os.makedirs(debug_path)

        if len(j2d) < 200:  # No rendering for verts
            if not (imgPath is None):
                img = cv2.imread(imgPath)
                img = img[:, :, ::-1]
                colors = cm.tab20c(np.linspace(0, 1, 25))
                fig = plt.figure(dpi=300)
                ax = fig.add_subplot(111)
                if not (imgPath is None):
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

                if not (imgPath is None):
                    savename = imgPath.split('/')[-1]
                    savename = savename.replace('.pkl', '.jpg')
                    plt.savefig(
                        os.path.join(
                            debug_path,
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

    smpl_gt = model_gt(betas=torch.tensor(gt['betas'][:, :num_betas], dtype=torch.float),
                        global_orient=torch.tensor(gt['global_orient'], dtype=torch.float),
                        body_pose=torch.tensor(gt['body_pose'], dtype=torch.float),
                        transl=torch.tensor(gt['transl'], dtype=torch.float), pose2rot=pose2rot)

    return smpl_gt.joints.detach().cpu().numpy().squeeze()


def get_projected_joints(
        args,
        df_row,
        jdx,
        model_male_kid_gt,
        model_female_kid,
        model_neutral_kid,
        model_male_gt,
        model_female_gt,
        model_neutral
    ):
    kid_flag = df_row.at['kid'][jdx]
    gender = df_row.at['gender'][jdx]

    smpl_path = os.path.join(
        args.gt_model_path,
        df_row.at['gt_path_smpl'][jdx]).replace(
        '.obj',
        '.pkl')
    if torch.cuda.is_available():
        gt = pickle.load(open(smpl_path, 'rb'))
    else:
        gt = CPU_Unpickler(open(smpl_path, 'rb')).load()
    gt_joints_local = get_smpl_vertices(
        kid_flag, gt, gender, model_neutral_kid, model_neutral)

    bounding_box, gt_joints_cam_2d, gt_joints_cam_3d = project_2d(args, df_row, jdx, gt_joints_local)
    return bounding_box, gt_joints_cam_2d, gt_joints_cam_3d


def get_projected_data(
        df_row: Series,
        jdx: int
    ):
    #gt_paths = 'data/Cam'
    gt_paths = ['data/Cam_julien/ground_truth.pkl']
    #img_folder = 'data/images_3840x2160/'
    img_folder = 'data/images_julien/'

    # Parse arguments to mimic the command line arguments.
    args = argparse.Namespace()
    args.modelFolder = 'data/model/'
    args.debug_path = 'data/debug_images/'
    args.numBetas = 10
    args.imgHeight = 2160
    args.imgWidth = 3840
    args.imgFolder = img_folder
    args.gt_paths = gt_paths
    args.modeltype = 'SMPL'
    args.kid_template_path = 'data/smpl_kid_template.npy'
    args.gt_model_path = 'data/GT_fits'
    args.debug = False
    args.regenerate = False

    model_male, model_male_kid, model_female, model_female_kid, model_neutral, model_neutral_kid = load_model(
        args)
    
    return get_projected_joints(
        args, 
        df_row,
        jdx,
        model_male_kid,
        model_female_kid,
        model_neutral_kid,
        model_male,
        model_female,
        model_neutral
    )


def prepare_sample(
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
    smpl_gt_path = df_row.at['gt_path_smpl'][jdx]
    with open(smpl_gt_path, 'rb') as pkl_f:
        smpl_gt = pickle.read(pkl_f)
        pose = matrix_to_axis_angle(smpl_gt['full_pose'])[0].cpu().detach().numpy()
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
    cam_t = np.zeros(shape=(3,), dtype=np.float32)
    bbox, joints_2d, joints_3d = get_projected_data(
        df_row=df_row,
        jdx=jdx
    )
    sample_values = PreparedSampleValues(
        pose=pose,
        shape=shape,
        style_vector=style_vector,
        garment_labels=garment_classes.labels_vector,
        cam_t=cam_t,
        joints_3d=joints_3d,
        joints_2d=joints_2d,
        bbox=bbox
    )
    img_path = os.path.join(AGORA_IMG_DIR, df_row['imgPath'])
    img = cv2.imread(img_path)
    img_crop = img[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1]]

    return img_crop, sample_values


class AGORADataGenerator(DataGenerator):

    """
    A data generation class specific to SURREAL dataset.
    """

    DATASET_NAME = 'agora'
    CHECKPOINT_COUNT = 100

    def __init__(self):
        """
        Initialize superclass and create clothed renderer.
        """
        super().__init__()

    def _save_sample(
            self, 
            dataset_dir: str, 
            sample_idx: int, 
            rgb_img: np.ndarray, 
            seg_maps: np.ndarray, 
            sample_values: PreparedSampleValues,
            samples_values: PreparedValuesArray
        ) -> None:
        """
        Save RGB and seg maps to disk, and update the values in the array (RAM).
        """
        img_dir = os.path.join(dataset_dir, self.IMG_DIR)
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        if rgb_img is not None:
            img_dir = os.path.join(
                dataset_dir, 
                self.IMG_DIR
            )
            img_path = os.path.join(
                img_dir, 
                self.IMG_NAME_TEMPLATE.format(idx=sample_idx)
            )
            cv2.imwrite(img_path, rgb_img)
        samples_values.append(sample_values)
        if sample_idx % self.CHECKPOINT_COUNT == 0 and sample_idx != 0:
            print(f'Saving values on checkpoint #{sample_idx}')
            self._save_values(samples_values, dataset_dir)

    def prepare(
            self,
            df: pandas.DataFrame
        ) -> None:
        """
        (Pre-)generate the dataset for particular upper+lower garment class.

        The generated samples are stored in files (RGB images, seg maps),
        i.e., update in the corresponding sample values' arrays. The sample
        arrays are frequently updated on the disk in case the failure
        happens along the way.
        """
        dataset_dir = os.path.join(
            self.DATA_ROOT_DIR,
            self.DATASET_NAME
        )
        samples_values, num_generated = self._create_values_array(
            dataset_dir=dataset_dir
        )
        agora_pkl_files = os.listdir(AGORA_PKL_DIR)
        samples_counter = 0
        for df_iter, df_path in tqdm(enumerate(agora_pkl_files)):
            logging.info(f'Preparing {df_iter}/{len(agora_pkl_files)} df...')
            df = pandas.read_pickle(df_path)
            for idx in tqdm(range(len(df))):
                for jdx in range(len(df.iloc[idx]['isValid'])):
                    if samples_counter < num_generated:
                        samples_counter += 1
                    img_crop, sample_values = prepare_sample(df.iloc[idx], jdx)

                    self._save_sample(
                        dataset_dir=dataset_dir, 
                        sample_idx=samples_counter, 
                        rgb_img=img_crop, 
                        seg_maps=None, 
                        sample_values=sample_values,
                        samples_values=samples_values
                    )
                    samples_counter += 1
