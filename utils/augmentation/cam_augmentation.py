import numpy as np
import torch


def augment_cam_t(mean_cam_t, xy_std=0.05, delta_z_range=(-0.5, 0.5)):
    batch_size = mean_cam_t.shape[0]
    device = mean_cam_t.device
    new_cam_t = mean_cam_t.clone()
    delta_tx_ty = torch.randn(batch_size, 2, device=device) * xy_std
    new_cam_t[:, :2] = mean_cam_t[:, :2] + delta_tx_ty

    l, h = delta_z_range
    delta_tz = (h - l) * torch.rand(batch_size, device=device) + l
    new_cam_t[:, 2] = mean_cam_t[:, 2] + delta_tz

    return new_cam_t


def augment_cam_t_numpy(mean_cam_t, xy_std=0.05, delta_z_range=(-0.5, 0.5)):
    new_cam_t = np.empty(mean_cam_t.shape, dtype=mean_cam_t.dtype)
    delta_tx_ty = np.random.randn(2,) * xy_std
    new_cam_t[:2] = mean_cam_t[:2] + delta_tx_ty
    new_cam_t[1] += 0.2

    l, h = delta_z_range
    delta_tz = (h - l) * np.random.randn(1,) + l
    new_cam_t[2] = mean_cam_t[2] + delta_tz

    return new_cam_t 
