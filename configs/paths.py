import os

# ------------------- SMPL Files -------------------
SMPL = '/data/hierprob3d/smpl'
J_REGRESSOR_EXTRA = '/data/hierprob3d/J_regressor_extra.npy'
COCOPLUS_REGRESSOR= '/data/hierprob3d/cocoplus_regressor.npy'
H36M_REGRESSOR = '/data/hierprob3d/J_regressor_h36m.npy'

# ------------------- DensePose Files for Textured Rendering -------------------
DP_UV_PROCESSED_FILE = '/data/hierprob3d/UV_Processed.mat'

# ------------------------- Eval Datasets -------------------------
PW3D_PATH = '/data/3DPW/test'
SSP3D_PATH = '/data/SSP-3D'

# ------------------------- Train Datasets -------------------------
TRAIN_POSES_PATH = '/data/hierprob3d/training/smpl_train_poses.npz'
TRAIN_TEXTURES_PATH = '/data/hierprob3d/training/smpl_train_textures.npz'
TRAIN_BACKGROUNDS_PATH = '/data/hierprob3d/training/lsun_backgrounds/train'
VAL_POSES_PATH = '/data/hierprob3d/training/smpl_val_poses.npz'
VAL_TEXTURES_PATH = '/data/hierprob3d/training/smpl_val_textures.npz'
VAL_BACKGROUNDS_PATH = '/data/hierprob3d/training/lsun_backgrounds/val'

# 2D keypoint detector path
HRNET_PATH = '/data/hierprob3d/pose_hrnet_w48_384x288.pth'

# Data directories
""" 
Train dataset abstract class containing only common constants.

Folder hierarchy of the stored data.
<DATA_ROOT_DIR>
    {dataset_name}/
        {gender}/
            <PARAMS_FNAME>
            <IMG_DIR>/
                <IMG_NAME_1>
                ...
            <SEG_MAP_DIR>/
                <SEG_MAP_1_1>
                ...
"""
DATA_ROOT_DIR = '/data/garmentor/'
RGB_DIR = 'rgb/'
VERIFY_DIR = 'verify/'
SEG_MAPS_DIR = 'segmentations/'

IMG_NAME_TEMPLATE = '{sample_idx:05d}.png'
SEG_MAPS_NAME_TEMPLATE = '{sample_idx:05d}.npz'
SEG_IMGS_NAME_TEMPLATE = '{sample_idx:05d}-{idx}.png'
VALUES_FNAME = 'values.npy'

# AGORA paths
AGORA_ROOT_DIR = os.path.join(DATA_ROOT_DIR, 'agora/')
AGORA_FULL_IMG_DIR = os.path.join(AGORA_ROOT_DIR, 'full_img/')
AGORA_FITS_DIR = os.path.join(AGORA_ROOT_DIR, 'fits_3d/')

AGORA_PKL_DIR = os.path.join(AGORA_ROOT_DIR, 'pkl/')
AGORA_GT_DIR = os.path.join(AGORA_PKL_DIR, 'gt/')
AGORA_CAM_DIR = os.path.join(AGORA_PKL_DIR, 'camera/')

AGORA_PREPARED_DIR = os.path.join(AGORA_ROOT_DIR, 'prepared/')
AGORA_VALUES_PATH = os.path.join(AGORA_PREPARED_DIR, 'values.npy')
AGORA_RGB_DIR_PATH = os.path.join(AGORA_PREPARED_DIR, 'rgb/')
AGORA_DEBUG_DIR_PATH = os.path.join(AGORA_PREPARED_DIR, 'debug/')
