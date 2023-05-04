from yacs.config import CfgNode

from configs.const import (
    MEAN_CAM_T,
    FOCAL_LENGTH,
    HEATMAP_GAUSSIAN_STD,
    IMG_WH,
    NUM_SMPL_BETAS,
    WP_CAM,
    BBOX_SCALE_FACTOR,
    OBJECT_DETECT_THRESHOLD
)

_C = CfgNode()

# Model
_C.MODEL = CfgNode()
#_C.MODEL.NUM_IN_CHANNELS = 18
#_C.MODEL.NUM_IN_CHANNELS = 17 + 5
#_C.MODEL.NUM_IN_CHANNELS = 20
_C.MODEL.NUM_IN_CHANNELS = 18 + 5
#_C.MODEL.NUM_RESNET_LAYERS = 18
_C.MODEL.NUM_RESNET_LAYERS = 50
#_C.MODEL.NUM_RESNET_LAYERS = 101
_C.MODEL.EMBED_DIM = 256
_C.MODEL.DELTA_I = True
_C.MODEL.DELTA_I_WEIGHT = 1.0
_C.MODEL.NUM_SMPL_BETAS = NUM_SMPL_BETAS
_C.MODEL.NUM_STYLE_PARAMS = 4
#_C.MODEL.NUM_GARMENT_CLASSES = 4
_C.MODEL.NUM_GARMENT_CLASSES = 2
_C.MODEL.USE_STYLE = True
#_C.MODEL.USE_STYLE = False
_C.MODEL.WP_CAM = WP_CAM
_C.MODEL.OUTPUT_SET = 'style'

# Input Data
_C.DATA = CfgNode()
_C.DATA.BBOX_THRESHOLD = OBJECT_DETECT_THRESHOLD
_C.DATA.BBOX_SCALE_FACTOR = BBOX_SCALE_FACTOR
_C.DATA.PROXY_REP_SIZE = IMG_WH
_C.DATA.HEATMAP_GAUSSIAN_STD = HEATMAP_GAUSSIAN_STD
_C.DATA.EDGE_NMS = True
_C.DATA.EDGE_THRESHOLD = 0.0
_C.DATA.EDGE_GAUSSIAN_STD = 1.0
_C.DATA.EDGE_GAUSSIAN_SIZE = 5

# Train
_C.TRAIN = CfgNode()
_C.TRAIN.NUM_EPOCHS = 300
#_C.TRAIN.BATCH_SIZE = 2
_C.TRAIN.BATCH_SIZE = 32
#_C.TRAIN.BATCH_SIZE = 4
_C.TRAIN.LR = 0.0001
_C.TRAIN.EPOCHS_PER_SAVE = 5
_C.TRAIN.PIN_MEMORY = True
_C.TRAIN.NUM_WORKERS = 2

# Train - Synthetic Data
_C.TRAIN.SYNTH_DATA = CfgNode()
_C.TRAIN.SYNTH_DATA.FOCAL_LENGTH = FOCAL_LENGTH
_C.TRAIN.SYNTH_DATA.MEAN_CAM_T = MEAN_CAM_T
_C.TRAIN.SYNTH_DATA.CROP_INPUT = True

# Train - Synthetic Data - Augmentation
_C.TRAIN.SYNTH_DATA.AUGMENT = CfgNode()

_C.TRAIN.SYNTH_DATA.AUGMENT.SMPL = CfgNode()
_C.TRAIN.SYNTH_DATA.AUGMENT.SMPL.SHAPE_STD = 1.25

_C.TRAIN.SYNTH_DATA.AUGMENT.GARMENTOR = CfgNode()
_C.TRAIN.SYNTH_DATA.AUGMENT.GARMENTOR.STYLE_STD = 0.5

_C.TRAIN.SYNTH_DATA.AUGMENT.CAM = CfgNode()
_C.TRAIN.SYNTH_DATA.AUGMENT.CAM.XY_STD = 0.05
_C.TRAIN.SYNTH_DATA.AUGMENT.CAM.DELTA_Z_RANGE = [-0.4, 0.1]

_C.TRAIN.SYNTH_DATA.AUGMENT.BBOX = CfgNode()
_C.TRAIN.SYNTH_DATA.AUGMENT.BBOX.DELTA_SCALE_RANGE = [-0.3, 0.2]
_C.TRAIN.SYNTH_DATA.AUGMENT.BBOX.DELTA_CENTRE_RANGE = [-5, 5]

_C.TRAIN.SYNTH_DATA.AUGMENT.PROXY_REP = CfgNode()
_C.TRAIN.SYNTH_DATA.AUGMENT.PROXY_REP.REMOVE_PARTS_CLASSES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]  # DensePose part classes
_C.TRAIN.SYNTH_DATA.AUGMENT.PROXY_REP.REMOVE_PARTS_PROBS = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1,
                                                            0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05]
_C.TRAIN.SYNTH_DATA.AUGMENT.PROXY_REP.REMOVE_APPENDAGE_JOINTS_PROB = 0.5
_C.TRAIN.SYNTH_DATA.AUGMENT.PROXY_REP.REMOVE_JOINTS_INDICES = [7, 8, 9, 10, 13, 14, 15, 16]  # COCO joint labels
_C.TRAIN.SYNTH_DATA.AUGMENT.PROXY_REP.REMOVE_JOINTS_PROB = 0.1
_C.TRAIN.SYNTH_DATA.AUGMENT.PROXY_REP.DELTA_J2D_DEV_RANGE = [-6, 6]
_C.TRAIN.SYNTH_DATA.AUGMENT.PROXY_REP.JOINTS_TO_SWAP = [[5, 6], [11, 12]]  # COCO joint labels
_C.TRAIN.SYNTH_DATA.AUGMENT.PROXY_REP.JOINTS_SWAP_PROB = 0.1
_C.TRAIN.SYNTH_DATA.AUGMENT.PROXY_REP.OCCLUDE_BOX_DIM = 48
_C.TRAIN.SYNTH_DATA.AUGMENT.PROXY_REP.OCCLUDE_BOX_PROB = 0.1
_C.TRAIN.SYNTH_DATA.AUGMENT.PROXY_REP.OCCLUDE_BOTTOM_PROB = 0.02
_C.TRAIN.SYNTH_DATA.AUGMENT.PROXY_REP.OCCLUDE_TOP_PROB = 0.005
_C.TRAIN.SYNTH_DATA.AUGMENT.PROXY_REP.OCCLUDE_VERTICAL_PROB = 0.05
_C.TRAIN.SYNTH_DATA.AUGMENT.PROXY_REP.EXTREME_CROP_PROB = 0.1

_C.TRAIN.SYNTH_DATA.AUGMENT.RGB = CfgNode()
_C.TRAIN.SYNTH_DATA.AUGMENT.RGB.LIGHT_LOC_RANGE = [0.05, 3.0]
_C.TRAIN.SYNTH_DATA.AUGMENT.RGB.LIGHT_AMBIENT_RANGE = [0.4, 0.8]
_C.TRAIN.SYNTH_DATA.AUGMENT.RGB.LIGHT_DIFFUSE_RANGE = [0.4, 0.8]
_C.TRAIN.SYNTH_DATA.AUGMENT.RGB.LIGHT_SPECULAR_RANGE = [0.0, 0.5]
_C.TRAIN.SYNTH_DATA.AUGMENT.RGB.OCCLUDE_BOTTOM_PROB = 0.02
_C.TRAIN.SYNTH_DATA.AUGMENT.RGB.OCCLUDE_TOP_PROB = 0.005
_C.TRAIN.SYNTH_DATA.AUGMENT.RGB.OCCLUDE_VERTICAL_PROB = 0.05
_C.TRAIN.SYNTH_DATA.AUGMENT.RGB.PIXEL_CHANNEL_NOISE = 0.2

# Loss
_C.LOSS = CfgNode()
_C.LOSS.SAMPLE_ON_CPU = True  # Rejection sampling from matrix-Fisher distribution may be faster on CPU than GPU
_C.LOSS.NUM_SAMPLES = 8
_C.LOSS.STAGE_CHANGE_EPOCH = 300

_C.LOSS.STAGE1 = CfgNode()
_C.LOSS.STAGE1.REDUCTION = 'mean'
_C.LOSS.STAGE1.MF_OVERREG = 1.005
_C.LOSS.STAGE1.J2D_LOSS_ON = 'means'
_C.LOSS.STAGE1.WEIGHTS = CfgNode()
_C.LOSS.STAGE1.WEIGHTS.POSE = 80.0
_C.LOSS.STAGE1.WEIGHTS.SHAPE = 80.0
_C.LOSS.STAGE1.WEIGHTS.STYLE = 160.0
_C.LOSS.STAGE1.WEIGHTS.JOINTS2D = 5000.0
_C.LOSS.STAGE1.WEIGHTS.GLOB_ROTMATS = 5000.0
_C.LOSS.STAGE1.WEIGHTS.VERTS3D = 0.0
_C.LOSS.STAGE1.WEIGHTS.JOINTS3D = 0.0

_C.LOSS.STAGE2 = CfgNode()
_C.LOSS.STAGE2.REDUCTION = 'mean'
_C.LOSS.STAGE2.MF_OVERREG = 1.005
_C.LOSS.STAGE2.J2D_LOSS_ON = 'means+samples'
_C.LOSS.STAGE2.WEIGHTS = CfgNode()
_C.LOSS.STAGE2.WEIGHTS.POSE = 10.0
_C.LOSS.STAGE2.WEIGHTS.SHAPE = 80.0
_C.LOSS.STAGE2.WEIGHTS.JOINTS2D = 30000.0
_C.LOSS.STAGE2.WEIGHTS.GLOB_ROTMATS = 5000.0
_C.LOSS.STAGE2.WEIGHTS.VERTS3D = 5000.0
_C.LOSS.STAGE2.WEIGHTS.JOINTS3D = 5000.0


def get_cfg_defaults():
    return _C.clone()
