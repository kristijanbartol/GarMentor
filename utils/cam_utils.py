from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Optional
import torch
import numpy as np

from configs.const import (
    FOCAL_LENGTH,
    IMG_WH,
    MEAN_CAM_T,
    WP_CAM
)
from utils.joints2d_utils import undo_keypoint_normalisation


def orthographic_project_torch(points3D, cam_params):
    """
    Scaled orthographic projection (i.e. weak perspective projection).
    :param points3D: (B, N, 3) batch of 3D point sets.
    :param cam_params: (B, 3) batch of weak-perspective camera parameters (scale, trans x, trans y)
    """
    proj_points = cam_params[:, None, [0]] * (points3D[:, :, :2] + cam_params[:, None, 1:])
    return proj_points


def get_intrinsics_matrix(
        img_width: int, 
        img_height: int, 
        focal_length: float
    ) -> np.ndarray:
    """
    Camera intrinsic matrix (calibration matrix) given focal length in pixels and img_width and
    img_height. Assumes that principal point is at (width/2, height/2).
    """
    K = np.array([[focal_length, 0., img_width/2.0],
                  [0., focal_length, img_height/2.0],
                  [0., 0., 1.]])
    return K


def perspective_project_torch(points, rotation, translation, cam_K=None,
                              focal_length=None, img_wh=None):
    """
    This function computes the perspective projection of a set of points in torch.
    Input:
        points (bs, N, 3): 3D points
        rotation (bs, 3, 3): Camera rotation
        translation (bs, 3): Camera translation
        Either
        cam_K (bs, 3, 3): Camera intrinsics matrix
        Or
        focal_length (bs,) or scalar: Focal length
        camera_center (bs, 2): Camera center
    """
    batch_size = points.shape[0]
    if cam_K is None:
        cam_K = torch.from_numpy(get_intrinsics_matrix(img_wh, img_wh, focal_length).astype(np.float32))
        cam_K = torch.cat(batch_size * [cam_K[None, :, :]], dim=0)
        cam_K = cam_K.to(points.device)

    # Transform points
    if rotation is not None:
        points = torch.einsum('bij,bkj->bki', rotation, points)
    points = points + translation.unsqueeze(1)

    # Apply perspective distortion
    projected_points = points / points[:, :, -1].unsqueeze(-1)

    # Apply camera intrinsics
    projected_points = torch.einsum('bij,bkj->bki', cam_K, projected_points)

    return projected_points[:, :, :-1]


def get_intrinsics_matrix(
        img_width: int, 
        img_height: int, 
        focal_length: float
    ) -> np.ndarray:
    """
    Camera intrinsic matrix (calibration matrix) given focal length in pixels and img_width and
    img_height. Assumes that principal point is at (width/2, height/2).
    """
    K = np.array([[focal_length, 0., img_width/2.0],
                  [0., focal_length, img_height/2.0],
                  [0., 0., 1.]])
    return K


def orthographic_project(
        points: np.ndarray, 
        wp_params: np.ndarray,
        cam_t: Optional[np.ndarray] = None
    ) -> np.ndarray:
    """ 
    Scaled orthographic projection (i.e. weak perspective projection).

    Parameters:
    -----------
        points: (N, 3) batch of 3D point sets.
        wp_params: (3,) weak-perspective camera parameters 
                        (scale, trans x, trans y).
        cam_t: (3,) optional translation of 3D points.
    """
    points = wp_params[None, [0]] * (points[:, :2] + wp_params[None, 1:])
    if cam_t is not None:
        points += np.expand_dims(cam_t, axis=0)
    return points


def perspective_project(
        points: np.ndarray, 
        rotation: np.ndarray, 
        translation: np.ndarray, 
        cam_K: Optional[np.ndarray] = None,
        focal_length: Optional[np.ndarray] = None,
        img_wh: Optional[np.ndarray] = None
    ) -> np.ndarray:
    """
    Computes the perspective projection of a set of points in torch.
    
    Parameters:
    -----------
        points (N, 3): 3D points
        rotation (3, 3): Camera rotation
        translation (3,): Camera translation
        Either
        cam_K (3, 3): Camera intrinsics matrix
        Or
        focal_length scalar: Focal length
    """
    if cam_K is None:
        cam_K = get_intrinsics_matrix(img_wh, img_wh, focal_length)

    # Transform points
    if rotation is not None:
        points = np.einsum('ij,kj->ki', rotation, points)
    points = points + np.expand_dims(translation, axis=0)

    # Apply perspective distortion
    projected_points = points / np.expand_dims(points[:, -1], axis=-1)

    # Apply camera intrinsics
    projected_points = np.einsum('ij,kj->ik', cam_K, projected_points)

    return projected_points[:, :-1]


def project_points(
        points: np.ndarray,
        projection_type: str,
        cam_t: Optional[np.ndarray] = None,
        wp_params: Optional[np.ndarray] = np.array(WP_CAM),
        focal_length: Optional[float] = FOCAL_LENGTH,
        img_wh: Optional[int] = IMG_WH
    ) -> np.ndarray:
    """
    Project 3D points to 2D image, given camera parameters.
    """
    if projection_type == 'perspective':
        points = perspective_project(
            points=points,
            rotation=None,
            translation=cam_t,
            cam_K=None,
            focal_length=focal_length,
            img_wh=img_wh
        )
    else:
        points = orthographic_project(
            points=points,
            wp_params=wp_params
        )
        points = undo_keypoint_normalisation(
            normalised_keypoints=points,
            img_wh=img_wh
        )
    return points


def convert_weak_perspective_to_camera_translation(cam_wp, focal_length, resolution):
    cam_t = np.array([cam_wp[1], cam_wp[2], 2 * focal_length / (resolution * cam_wp[0] + 1e-9)])
    return cam_t


def batch_convert_weak_perspective_to_camera_translation(wp_cams, focal_length, resolution):
    num = wp_cams.shape[0]
    cam_ts = np.zeros((num, 3), dtype=np.float32)
    for i in range(num):
        cam_t = convert_weak_perspective_to_camera_translation(wp_cams[i],
                                                               focal_length,
                                                               resolution)
        cam_ts[i] = cam_t.astype(np.float32)
    return cam_ts


def batch_convert_weak_perspective_to_camera_translation_torch(cam_wp, focal_length, resolution):
    cam_tx = cam_wp[:, 1]
    cam_ty = cam_wp[:, 2]
    cam_tz = 2 * focal_length / (resolution * cam_wp[:, 0] + 1e-9)
    cam_t = torch.stack([cam_tx, cam_ty, cam_tz], dim=-1)
    return cam_t

