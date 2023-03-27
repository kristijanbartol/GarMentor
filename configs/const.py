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

# Constant strings
TRAIN = 'train'
VALID = 'valid'
SURREAL_DATASET_NAME = 'surreal'
AGORA_DATASET_NAME = 'agora'
