from typing import Tuple
import numpy as np
import torch
import math

from configs.const import AGORA_BBOX_COEF


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


def check_joints2d_occluded_torch(seg14part, vis, pixel_count_threshold=50):
    """
    Check if 2D joints are not self-occluded in the rendered silhouette/seg, by checking if corresponding body parts are
    visible in the corresponding 14 part seg.
    :param seg14part: (B, D, D)
    :param vis: (B, 17)
    """
    new_vis = vis.clone()
    joints_to_check_and_corresponding_bodyparts = {7: 3, 8: 5, 9: 12, 10: 11, 13: 7, 14: 9, 15: 14, 16: 13}

    for joint_index in joints_to_check_and_corresponding_bodyparts.keys():
        part = joints_to_check_and_corresponding_bodyparts[joint_index]
        num_pixels_part = (seg14part == part).sum(dim=(1, 2))  # (B,)
        visibility_flag = (num_pixels_part > pixel_count_threshold)  # (B,)
        new_vis[:, joint_index] = (vis[:, joint_index] & visibility_flag)

    return new_vis


def _to_within_frame(
        x1: int, 
        x2: int, 
        y1: int, 
        y2: int, 
        img_height: int, 
        img_width: int
    ) -> Tuple[int, int, int, int]:
    """
    Place crop coordinate within the image frame by pushing the square
    inward in case some of the coordinates went outside. The square size
    remains the same.
    """
    if x1 < 0:
        x_diff = x2 - x1
        x1 = 0
        x2 = x_diff
    if x2 > img_width:
        x_diff = x2 - x1
        x2 = img_width
        x1 = img_width - x_diff
    if y1 < 0:
        y_diff = y2 - y1
        y1 = 0
        y2 = y_diff
    if y2 > img_height:
        y_diff = y2 - y1
        y2 = img_height
        y1 = img_height - y_diff

    return x1, x2, y1, y2


def extract_crop_coordinates(
        joints_2d: np.ndarray,
        h: int,
        w: int    
    ) -> Tuple[np.ndarray, bool]:
    """
    Crop coordinates (bbox) extraction algorithm from 2D joint coordinates.

    First, get the minimum and maximum x and y coordinates of 2D joints.
    Then, get the middle (center) point of the bounding box as an average
    of all joint coordinates. The bounding box height and width are the
    differences between the maximum and minimum x and y coordinates, i.e.,
    the differences multiple by some constant (x1.3) in order to cut the
    image around the subject and not just exactly the keypoints part.
    The bounding box coordinates are calculated as the offset from the 
    middle (center) point. The offset is half of the bounding box height 
    (for both x and y coordinates because we want the bounding box to 
    be quadratic).

    Note that the bounding box might seem a bit more over the 
    top of the head than beneath the feet because the mean point is
    a bit above the middle of the body due to more keypoints being in
    the upper part of the body. It doesn't matter much for the model
    because I anyways do not use the same orthographic projection as
    in ClothSURREAL and will not estimate the "camera parameters" for
    the ClothAGORA, or might even completely remove that from the
    model outputs.

    Finally, the coordinates should be moved inward in case of being
    outside of the image frame (`_to_within_frame` function). This
    operation might fail in case that the person is both large and very
    close to the edges. In this case, we simply flag this operation as
    unsuccessful and return the False flag, which will result in this
    example being skipped.
    """
    _, y_min = joints_2d.min(axis=0)
    _, y_max = joints_2d.max(axis=0)
    c_x, c_y = joints_2d.mean(axis=0)
    d_y = y_max - y_min
    h_y = d_y * AGORA_BBOX_COEF

    x1, y1 = (math.floor(c_x - h_y / 2), math.floor(c_y - h_y / 2))
    x2, y2 = (math.floor(c_x + h_y / 2), math.floor(c_y + h_y / 2))

    x1, x2, y1, y2 = _to_within_frame(
        x1=x1,
        x2=x2,
        y1=y1,
        y2=y2,
        img_height=h,
        img_width=w
    )
    success = (x1 > 0) and (y1 > 0) and (x2 < w) and y2 < h

    return np.array([[y1, y2], [x1, x2]], dtype=np.uint16), success
