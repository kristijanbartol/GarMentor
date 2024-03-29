import os

# ------------------- SMPL Files -------------------
SMPL_DIR = '/data/hierprob3d/smpl'
SMPL_PATH_TEMPLATE = os.path.join(SMPL_DIR, 'SMPL_{gender}.pkl')
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
DATA_ROOT_DIR = '/data/cat/'
TN_DIR = 'tn/'
DN_DIR = 'dn/'
PARAMS_DIR = 'params/'
DATASETS_DIR = 'datasets/'
RGB_DIR = 'rgb/'
VERIFY_DIR = 'verify/'
SEG_MAPS_DIR = 'segmentations/'

IMG_NAME_TEMPLATE = '{sample_idx:05d}.png'
SEG_MAPS_NAME_TEMPLATE = '{sample_idx:05d}.npz'
SEG_IMGS_NAME_TEMPLATE = '{sample_idx:05d}-{idx}.png'
VALUES_FNAME = 'values.npy'

# AGORA paths
AGORA_ROOT_DIR = os.path.join(
    DATA_ROOT_DIR, 
    'agora/'
)
AGORA_FULL_IMG_ROOT_DIR = os.path.join(
    AGORA_ROOT_DIR, 
    'full_img/'
)
AGORA_FITS_DIR = os.path.join(
    AGORA_ROOT_DIR, 
    'fits_3d/'
)

AGORA_PKL_DIR = os.path.join(
    AGORA_ROOT_DIR, 
    'pkl/'
)
AGORA_GT_DIR = os.path.join(
    AGORA_PKL_DIR, 
    'gt/'
)
AGORA_CAM_DIR = os.path.join(
    AGORA_PKL_DIR, 
    'camera/'
)

AGORA_PREPARED_DIR = os.path.join(
    AGORA_ROOT_DIR, 
    'prepared/'
)
AGORA_DEBUG_DIR_PATH = os.path.join(
    AGORA_PREPARED_DIR, 
    'debug/'
)

RESOLUTION_TEMPLATE = '{resolution}'    # "H x W"
SCENE_NAME_TEMPLATE = '{scene_name}'
AGORA_FULL_IMG_DIR_TEMPLATE = os.path.join(
    AGORA_FULL_IMG_ROOT_DIR, 
    SCENE_NAME_TEMPLATE,
    RESOLUTION_TEMPLATE
)

############ DRAPENET PATHS ###################
DRAPENET_DIR = '/GarMentor/drapenet_for_garmentor/'
DRAPENET_CHECKPOINTS = os.path.join(DRAPENET_DIR, 'checkpoints/')
DRAPENET_EXTRADIR = os.path.join(DRAPENET_DIR, 'extra-data/')
DRAPENET_SMPLDIR = os.path.join(DRAPENET_DIR, 'smpl_pytorch/')

TOP_CODES_FNAME = 'top_codes.pt'
TOP_MODEL_FNAME = 'top_udf.pt'

BOTTOM_CODES_FNAME = 'bottom_codes.pt'
BOTTOM_MODEL_FNAME = 'bottom_udf.pt'
