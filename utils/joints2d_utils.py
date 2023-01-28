import numpy as np
import torch


def undo_keypoint_normalisation(normalised_keypoints, img_wh):
    """
    Converts normalised keypoints from [-1, 1] space to pixel space i.e. [0, img_wh]
    """
    keypoints = (normalised_keypoints + 1) * (img_wh/2.0)
    return keypoints


def check_joints2d_visibility_torch(joints2d,
                                    img_wh,
                                    visibility=None):
    """
    Checks if 2D joints are within the image dimensions.
    """
    if visibility is None:
        visibility = torch.ones(joints2d.shape[:2], device=joints2d.device, dtype=torch.bool)
    visibility[joints2d[:, :, 0] > img_wh] = 0
    visibility[joints2d[:, :, 1] > img_wh] = 0
    visibility[joints2d[:, :, 0] < 0] = 0
    visibility[joints2d[:, :, 1] < 0] = 0

    return visibility


def check_joints2d_occluded_torch(seg, joints2d, vis):
    """
    Check if 2D joints are not self-occluded in the rendered silhouette/seg, by checking if corresponding body parts are
    visible in the corresponding 14 part seg.
    :param seg14part: (B, D, D)
    :param vis: (B, 17)
    """
    # TODO: Rewrite this for fully parallel execution.
    new_vis = vis.clone()

    for joint_index in range(vis.shape[1]):
        if seg[:, joints2d[joint_index]] is True:
            new_vis[:, joint_index] = 0

    return new_vis

