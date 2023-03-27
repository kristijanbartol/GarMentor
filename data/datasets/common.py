from typing import List
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import os

import configs.paths as paths


def get_background_paths(
        backgrounds_dir_path: str, 
        num_backgrounds: int = -1
    ) -> List[str]:
    print('Loading background paths...')
    backgrounds_paths = []
    for f in tqdm(sorted(os.listdir(backgrounds_dir_path)[:num_backgrounds])): 
        if f.endswith('.jpg'):
            backgrounds_paths.append(
                os.path.join(backgrounds_dir_path, f)
            )
    return backgrounds_paths


def load_background(
        backgrounds_paths: List[str],
        img_wh: int,
        num_samples: int = 1
    ) -> np.ndarray:
    """
    Load random backgrounds. Adapted from the original HierProb3D code.
    """
    bg_samples = []
    for _ in range(num_samples):
        bg_idx = torch.randint(low=0, high=len(backgrounds_paths), # type: ignore
            size=(1,)).item()
        bg_path = backgrounds_paths[bg_idx]
        background = cv2.cvtColor(cv2.imread(bg_path), cv2.COLOR_BGR2RGB)
        background = cv2.resize(background, (img_wh, img_wh), 
            interpolation=cv2.INTER_LINEAR)
        background = background.transpose(2, 0, 1)
        bg_samples.append(background)
    bg_samples = np.stack(bg_samples, axis=0).squeeze()
    return torch.from_numpy(bg_samples / 255.).float() # type: ignore


class Values:
    
    def __init__(self):
        self.poses = np.empty(0)
        self.shapes = np.empty(0)
        self.style_vectors = np.empty(0)
        self.garment_labelss = np.empty(0)
        self.joints_3ds = np.empty(0)
        self.joints_2ds = np.empty(0)
        self.cam_ts = np.empty(0)
        self.bboxs = np.empty(0)
    
    def load(self, 
             np_path: str, 
             data_slice: slice = slice(None)
             ) -> None:
        data = np.load(np_path, allow_pickle=True).item()
        
        np.append(self.poses, data['poses'][data_slice])
        np.append(self.shapes, data['shapes'][data_slice])
        np.append(self.style_vectors, data['style_vectors'][data_slice])
        np.append(self.cam_ts, data['cam_ts'][data_slice])
        np.append(self.joints_2ds, data['joint_2ds'][data_slice])
        np.append(self.joints_3ds, data['joints_3ds'][data_slice])
        np.append(self.garment_labelss, data['garment_labelss'][data_slice])
        np.append(self.bboxs, data['bboxs'][data_slice])
        
    def to_numpy(self) -> None:
        self.poses = np.concatenate(self.poses, axis=0)
        self.shapes = np.concatenate(self.shapes, axis=0)
        self.style_vectors = np.concatenate(self.style_vectors, axis=0)
        self.cam_ts = np.concatenate(self.cam_ts, axis=0)
        self.joints_2ds = np.concatenate(self.joints_2ds, axis=0)
        self.joints_3ds = np.concatenate(self.joints_3ds, axis=0)
        self.garment_labelss = np.concatenate(self.garment_labelss, axis=0)
        self.bboxs = np.concatenate(self.bboxs, axis=0)
    
    def __len__(self):
        return self.poses.shape[0]
