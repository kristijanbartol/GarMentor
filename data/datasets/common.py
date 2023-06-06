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
        background = background.transpose(2, 0, 1)[::-1]
        bg_samples.append(background)
    bg_samples = np.stack(bg_samples, axis=0).squeeze()
    return torch.from_numpy(bg_samples / 255.).float() # type: ignore


class Values:
    
    def __init__(self):
        self._poses: List = []
        self._shapes = []
        self._style_vectors = []
        self._garment_labelss = []
        self._joints_3ds = []
        self._joints_2ds = []
        self._cam_ts = []
        self._bboxs = []
    
    def load(self, 
             np_path: str, 
             data_slice: slice = slice(None)
             ) -> None:
        data = np.load(np_path, allow_pickle=True).item()
        
        self._poses.append(data['poses'][data_slice])
        self._shapes.append(data['shapes'][data_slice])
        self._style_vectors.append(data['style_vectors'][data_slice])
        self._cam_ts.append(data['cam_ts'][data_slice])
        self._joints_2ds.append(data['joints_2ds'][data_slice])
        self._joints_3ds.append(data['joints_3ds'][data_slice])
        self._garment_labelss.append(data['garment_labelss'][data_slice])
        self._bboxs.append(data['bboxs'][data_slice])
        
    def to_numpy(self) -> None:
        self.poses: np.ndarray = np.concatenate(self._poses, axis=0)
        self.shapes: np.ndarray = np.concatenate(self._shapes, axis=0)
        self.style_vectors: np.ndarray = np.concatenate(self._style_vectors, axis=0)
        self.cam_ts: np.ndarray = np.concatenate(self._cam_ts, axis=0)
        self.joints_2ds: np.ndarray = np.concatenate(self._joints_2ds, axis=0)
        self.joints_3ds: np.ndarray = np.concatenate(self._joints_3ds, axis=0)
        self.garment_labelss: np.ndarray = np.concatenate(self._garment_labelss, axis=0)
        self.bboxs: np.ndarray = np.concatenate(self._bboxs, axis=0)
    
    def __len__(self):
        return self.poses.shape[0]
