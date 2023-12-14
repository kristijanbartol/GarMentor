import numpy as np

# Rendering constants
CAM_DIST = 2.7
MEAN_CAM_Y_OFFSET = .3
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
HEATMAP_GAUSSIAN_STD = 4.0

# Height constants (for the volume extraction)
GENDER_HEIGHT = {
    'male': 1.82,
    'female': 1.68,
    'neutral': 1.75
}

# Sampling settings
NUM_SHAPE_PARAMS = 10
SHAPE_MEANS = np.zeros(NUM_SHAPE_PARAMS, dtype=np.float32)
SHAPE_STDS = np.ones(NUM_SHAPE_PARAMS, dtype=np.float32) * 1.25
NUM_STYLE_PARAMS = 4
NUM_GARMENT_CLASSES = 4
STYLE_MEANS = np.zeros(NUM_STYLE_PARAMS, dtype=np.float32)
STYLE_STDS = np.ones(NUM_STYLE_PARAMS, dtype=np.float32) * 0.25

SAMPLING_STRATEGIES = {
    'pose': [
        'zero',
        'simple',
        'mocap'
    ],
    'global_orient': [
        'zero',
        'frontal',
        'diverse',
        'mocap'
    ],
    'shape': [
        'normal',
        #'uniform'
    ],
    'style': [
        'normal',
        'predefined'    # for DrapeNet, in particular
        #'uniform'
    ]
}

NUM_SAMPLES = {
    'zero_pose': {
        'train': 1,
        'valid': 1
    },
    'simple_pose': {
        'train': 20000,
        'valid': 4000
    },
    'mocap_pose': {
        'train': None,  # will be automatically determined
        'valid': None   # will be automatically determined
    },
    'zero_global_orient': {
        'train': 1,
        'valid': 1
    },
    'frontal_global_orient': {
        'train': 20000,
        'valid': 4000
    },
    'diverse_global_orient': {
        'train': 20000,
        'valid': 4000
    },
    'mocap_global_orient': {
        'train': None,  # will be automatically determined
        'valid': None   # will be automatically determined
    },
    'normal_shape': {
        'train': 100000,
        'valid': 20000
    },
    'uniform_shape': {
        'train': 100000,
        'valid': 20000
    },
    'normal_style': {
        'train': 100000,
        'valid': 20000
    },
    'normal_style': {
        'train': 100000,
        'valid': 20000
    },
}

FRONTAL_ORIENT_RANGES = np.array([[-0.5, 0.4], [-0.6, 0.6], [0., 0.]])
DIVERSE_ORIENT_RANGES = np.array([[-0.5, 0.4], [-3.0, 3.0], [-0.1, 0.1]])
SIMPLE_POSE_RANGES = {
    1 : np.array([[0, 0], [0, 0], [0, 0.5]]),     # left hip
    2 : np.array([[0, 0], [0, 0], [-0.5, 0]]),    # right hip
    7 : np.array([[0, 0], [0, 0], [-0.5, 0.2]]),  # left ankle
    8 : np.array([[0, 0], [0, 0], [-0.2, 0.5]]),  # right ankle
    16: np.array([[0, 0], [0, 0], [-1.3, 1]]),    # left shoulder
    17: np.array([[-1, 1.3], [0, 0], [0, 0]]),    # right shoulder
    18: np.array([[-1, 0.2], [0, 0], [0, 0]]),    # left elbow
    19: np.array([[-1, 0.2], [0, 0], [0, 0]]),    # right elbow
    20: np.array([(-0.5, 0.5), (0, 0), (0, 0)]),  # left wrist
    21: np.array([(-0.5, 0.5), (0, 0), (0, 0)])   # right wrist
}
INTRA_ORIENT_INTERVALS = {
    'frontal': [
        [None, [-0.5, -0.45], None],
        [None, [-0.3, -0.25], None],
        [None, [0.25, 0.3], None],
        [None, [0.45, 0.5], None]
    ],
    'diverse': [
        [None, [-2.4, -2.1], None],
        [None, [-1.2, -0.9], None],
        [None, [0.9, 1.2], None],
        [None, [2.1, 2.4], None]
    ]
}
EXTRA_ORIENT_INTERVALS = {
    'frontal': [
        [None, [0.4, 0.6], None]
    ],
    'diverse': [
        [None, [1.8, 3.0], None]
    ]
}
INTRA_SHAPE_INTERVALS = [
    [None, [-2.4, -1.5], [-2.4, -1.5]] + [None] * 7,
    [None, [-1.0, -0.4], [-1.0, -0.4]] + [None] * 7,
    [None, [0.4, 1.0], [0.4, 1.0]] + [None] * 7,
    [None, [1.5, 2.4], [1.5, 2.4]] + [None] * 7
]
EXTRA_SHAPE_INTERVALS = [
    [None, [1.5, 3.], [1.5, 3]] + [None] * 7
]
INTRA_STYLE_INTERVALS = [
    [None, [-0.6, -0.52], None, None],
    [None, [-0.3, -0.22], None, None],
    [None, [0.22, 0.3], None, None],
    [None, [0.52, 0.6], None, None]
]
EXTRA_STYLE_INTERVALS = [
    [None, [0.3, 0.6], None, None]
]

SHAPE_MIN = -3.0
SHAPE_MAX = 3.0
STYLE_MIN = -0.75
STYLE_MAX = 0.75

# Constant strings
TRAIN = 'train'
VALID = 'valid'
SURREAL_DATASET_NAME = 'surreal'
AGORA_DATASET_NAME = 'agora'

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
