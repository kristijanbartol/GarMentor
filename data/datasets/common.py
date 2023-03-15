from typing import List
import tqdm
import torch
import cv2
import numpy as np
import os


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
        bg_idx = torch.randint(low=0, high=len(backgrounds_paths), 
            size=(1,)).item()
        bg_path = backgrounds_paths[bg_idx]
        background = cv2.cvtColor(cv2.imread(bg_path), cv2.COLOR_BGR2RGB)
        background = cv2.resize(background, (img_wh, img_wh), 
            interpolation=cv2.INTER_LINEAR)
        background = background.transpose(2, 0, 1)
        bg_samples.append(background)
    bg_samples = np.stack(bg_samples, axis=0).squeeze()
    return torch.from_numpy(bg_samples / 255.).float()
