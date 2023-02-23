import os
import numpy as np

#######################
### GARMENT CLASSES ###
#######################

MGN_CLASSES = ['Pants', 'ShortPants', 'ShirtNoCoat', 'TShirtNoCoat']
GARMENT_CLASSES = ['pant', 'short-pant', 'shirt', 't-shirt']

#############
### PATHS ###
#############

DATA_DIR = '/data/'

AGORA_DIR = os.path.join(DATA_DIR, 'agora/')
GARMENTOR_DIR = os.path.join(DATA_DIR, 'garmentor/')
MGN_DIR = os.path.join(DATA_DIR, 'mgn/')

SCANS_DIR = os.path.join(AGORA_DIR, 'smplx_gt/')
MGN_DATASET = os.path.join(MGN_DIR, 'Multi-Garment_dataset/')
UV_MAPS_PATH = os.path.join(MGN_DIR, 'uv_maps/')
SUBJECT_OBJ_SAVEDIR = os.path.join(GARMENTOR_DIR, 'agora/subjects/')
SCENE_OBJ_SAVEDIR = os.path.join(GARMENTOR_DIR, 'agora', 'scenes')
CAM_DIR = os.path.join(AGORA_DIR, 'camera/')

TRAIN_CAM_DIR = os.path.join(CAM_DIR, 'train/')
VAL_CAM_DIR = os.path.join(CAM_DIR, 'valid/')

BODY_FACES = np.load('/data/hierprob3d/smpl/smpl_faces.npy').astype(np.int32)
