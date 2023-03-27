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

# Data directories
DATA_ROOT_DIR = '/data/garmentor/'
IMG_DIR = 'rgb/'
SEG_MAPS_DIR = 'segmentations/'

IMG_NAME_TEMPLATE = '{idx:05d}.png'
SEG_MAPS_NAME_TEMPLATE = '{idx:05d}.npz'
VALUES_FNAME = 'values.npy'

# AGORA paths
AGORA_PKL_DIR = '/data/agora/pkl/'
AGORA_IMG_DIR = '/data/agora/full_img/'
AGORA_RGB_DIR = '/data/garmentor/agora/rgb/'
AGORA_VALUES_PATH = '/data/garmentor/agora/values.npy'
