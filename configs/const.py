# Rendering constants
CAM_DIST = 2.7
MEAN_CAM_Y_OFFSET = 0.2
# NOTE (cam_t): cam_t is currently relevant only for the augmentation purposes,
#               and not as a parameter for learning.
MEAN_CAM_T = [0.0, MEAN_CAM_Y_OFFSET, CAM_DIST]
FOCAL_LENGTH = 300.0
IMG_WH = 256
NUM_SMPL_BETAS = 10
ORTHOGRAPHIC_SCALE = 0.9
WP_CAM = [ORTHOGRAPHIC_SCALE, 0., 0.]

LIGHT_T = ((0.0, 0.0, 2.0),)
LIGHT_AMBIENT_COLOR = ((0.5, 0.5, 0.5),)
LIGHT_DIFFUSE_COLOR = ((0.3, 0.3, 0.3),)
LIGHT_SPECULAR_COLOR = ((0.2, 0.2, 0.2),)
BACKGROUND_COLOR = (0.0, 0.0, 0.0)

# Input data
BBOX_SCALE_FACTOR = 1.2
OBJECT_DETECT_THRESHOLD = 0.95

# Constant strings
TRAIN = 'train'
VALID = 'valid'

DIG_SURREAL_DATASET_NAME = 'dig-surreal'
TN_SURREAL_DATASET_NAME = 'tn-surreal'
TN_AGORA_DATASET_NAME = 'tn-agora'

# AGORA const
RESOLUTION = {
    'high': [2160, 3840],
    'normal': [720, 1280]
}
PREP_CROP_SIZE = {
    'high': [600, 600],
    'normal': [300, 300]
}
TRAIN_CROP_SIZE = {
    'high': [512, 512],
    'normal': [256, 256]
}
AGORA_BBOX_COEF = 1.4

def to_resolution_str(resolution_label: str) -> str:
    h, w = RESOLUTION[resolution_label]
    return f'{h}x{w}'
