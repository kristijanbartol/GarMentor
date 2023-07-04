from typing import Optional, Tuple, List
import torch
import numpy as np

from configs import paths
import configs.const
from configs.const import (
    SIMPLE_POSE_RANGES,
    SHAPE_MEANS,
    SHAPE_STDS,
    STYLE_MEANS,
    STYLE_STDS,
    NUM_SAMPLES,
    NUM_GARMENT_CLASSES,
    NUM_STYLE_PARAMS,
    SHAPE_MIN,
    SHAPE_MAX,
    STYLE_MIN,
    STYLE_MAX
)
from utils.rigid_transform_utils import quat_to_rotmat, aa_rotate_translate_points_pytorch3d
from utils.cam_utils import orthographic_project_torch
from utils.joints2d_utils import undo_keypoint_normalisation
from utils.label_conversions import convert_heatmaps_to_2Djoints_coordinates_torch, ALL_JOINTS_TO_COCO_MAP


# NOTE: Might create a Parameters class.

def _ranges_to_sizes_and_offsets(
        ranges: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
    sizes = ranges[:, 1] - ranges[:, 0] # type: ignore
    offsets = ranges[:, 0]
    return sizes, offsets


def _ranges_to_scaled_rand_array(
        ranges: np.ndarray,
        num_samples: int
    ) -> np.ndarray:
    sizes, offsets = _ranges_to_sizes_and_offsets(ranges)
    rand_values = np.random.rand(num_samples, 3)
    scaled_values = rand_values * sizes + offsets
    return scaled_values


def _load_poses(train_poses_path: str) -> np.ndarray:
    data = np.load(train_poses_path)
    fnames = data['fnames']
    poses = data['poses']
    indices = [i for i, x in enumerate(fnames)
                if (x.startswith('h36m') or x.startswith('up3d') or x.startswith('3dpw'))]
    poses_array = np.stack([poses[i] for i in indices], axis=0)
    return poses_array


def _sample_mocap_pose_common() -> Tuple[np.ndarray, np.ndarray]:
    poses_array = _load_poses(paths.TRAIN_POSES_PATH)
    num_poses_loaded = poses_array.shape[0]
    num_train_loaded = int(num_poses_loaded * 0.85)
    train_idxs = np.random.choice(
        np.arange(poses_array.shape[0]), 
        size=num_train_loaded
    )
    return poses_array[train_idxs], poses_array[~train_idxs]

    

def _split_new_samples(
        new_samples: np.ndarray, 
        train_samples: List[np.ndarray], 
        valid_samples: List[np.ndarray], 
        num_train: int, 
        num_valid: int, 
        intervals: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    for sample in new_samples:
        for interval in intervals:
            within_valid_interval = True
            for cidx, component_interval in enumerate(interval):
                if component_interval is not None:
                    if not component_interval[0] <= sample[cidx] < component_interval[1]:
                        within_valid_interval = False
            if within_valid_interval:
                if len(valid_samples) < num_valid:
                    valid_samples.append(sample)
            else:
                if len(train_samples) < num_train:
                    train_samples.append(sample)
    return train_samples, valid_samples


def _sample_global_orient(
        num_train: int,
        num_valid: int,
        intervals_type: str,     # ['intra', 'extra']
        range_type: str         # ['frontal', 'diverse']
    ) -> Tuple[np.ndarray, np.ndarray]:
    train_samples, valid_samples = [], []
    ranges = getattr(
        configs.const, 
        f'{range_type.upper()}_ORIENT_RANGES'
    )
    intervals = getattr(
        configs.const, 
        f'{intervals_type.upper()}_ORIENT_INTERVALS'
    )
    while len(train_samples) < num_train or len(valid_samples) < num_valid:
        new_samples = _ranges_to_scaled_rand_array(
            ranges=ranges, 
            num_samples=num_train+num_valid
        )
        train_samples, valid_samples = _split_new_samples(
            new_samples=new_samples,
            train_samples=train_samples,
            valid_samples=valid_samples,
            num_train=num_train,
            num_valid=num_valid,
            intervals=intervals[range_type]
        )
    return np.stack(train_samples, axis=0), np.stack(valid_samples, axis=0)


def _clip_samples(
        samples: np.ndarray,
        clip_min: Optional[float] = None,
        clip_max: Optional[float] = None
    ) -> np.ndarray:
    clip_min = clip_min if clip_min is not None else np.min(samples)
    clip_max = clip_max if clip_max is not None else np.max(samples)
    return np.clip(samples, a_min=clip_min, a_max=clip_max)


def _sample_normal_pc(
        pc_type: str,           # ['shape', 'style']
        mean_params: np.ndarray,      # (10,) OR (4,4)
        std_vector: np.ndarray,       # (10,) OR (4,4)
        num_train: int,
        num_valid: int,
        intervals_type: str,    # ['intra', 'extra']
        clip_min: Optional[float] = None,
        clip_max: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:             # (10,)
    """
    (Truncated) normal sampling of shape parameter deviations from the mean.
    """
    train_samples, valid_samples = [], []
    intervals = getattr(
        configs.const, 
        f'{intervals_type.upper()}_{pc_type.upper()}_INTERVALS'
    )
    while len(train_samples) < num_train or len(valid_samples) < num_valid:
        new_samples_shape = [num_train + num_valid] + list(mean_params.shape)
        new_samples = mean_params + np.random.randn(*new_samples_shape) * std_vector
        new_samples = _clip_samples(
            samples=new_samples,
            clip_min=clip_min,
            clip_max=clip_max
        )
        train_samples, valid_samples = _split_new_samples(
            new_samples=new_samples,
            train_samples=train_samples,
            valid_samples=valid_samples,
            num_train=num_train,
            num_valid=num_valid,
            intervals=intervals
        )
    return np.stack(train_samples, axis=0), np.stack(valid_samples, axis=0)


def sample_zero_pose(_) -> Tuple[np.ndarray, np.ndarray]:
    return (
        np.zeros((NUM_SAMPLES['zero_pose']['train'], 69)), 
        np.zeros((NUM_SAMPLES['zero_pose']['valid'], 69))
    )


def sample_simple_pose(_) -> Tuple[np.ndarray, np.ndarray]:
    pose_samples = []
    num_train = NUM_SAMPLES['simple_pose']['train']
    num_valid = NUM_SAMPLES['simple_pose']['valid']
    pose_samples = np.zeros((num_train + num_valid, 69))
    for joint_idx in SIMPLE_POSE_RANGES:
        rand_values = _ranges_to_scaled_rand_array(
            ranges=SIMPLE_POSE_RANGES[joint_idx],
            num_samples=num_train+num_valid
        )
        pose_samples[:, joint_idx*3:(joint_idx+1)*3] = rand_values
    return pose_samples[:num_train], pose_samples[:num_valid]


def sample_mocap_pose(_) -> Tuple[np.ndarray, np.ndarray]:
    train_poses, valid_poses = _sample_mocap_pose_common()
    return train_poses[:, 3:], valid_poses[:, 3:]


def sample_zero_global_orient(interval_type: str) -> Tuple[np.ndarray, np.ndarray]:
    return (
        np.zeros((NUM_SAMPLES['zero_global_orient']['train'], 3)), 
        np.zeros((NUM_SAMPLES['zero_global_orient']['valid'], 3))
    )


def sample_frontal_global_orient(intervals_type: str) -> Tuple[np.ndarray, np.ndarray]:
    return _sample_global_orient(
        num_train=NUM_SAMPLES['frontal_global_orient']['train'],
        num_valid=NUM_SAMPLES['frontal_global_orient']['valid'],
        intervals_type=intervals_type,
        range_type='frontal'
    )


def sample_diverse_global_orient(intervals_type: str) -> Tuple[np.ndarray, np.ndarray]:
    return _sample_global_orient(
        num_train=NUM_SAMPLES['diverse_global_orient']['train'],
        num_valid=NUM_SAMPLES['diverse_global_orient']['valid'],
        intervals_type=intervals_type,
        range_type='diverse'
    )


def sample_mocap_global_orient(_) -> Tuple[np.ndarray, np.ndarray]:
    train_orients, valid_orients = _sample_mocap_pose_common()
    return train_orients[:, :3], valid_orients[:, :3]


def sample_normal_shape(intervals_type: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    (Truncated) normal sampling of shape parameter deviations from the mean.
    """
    return _sample_normal_pc(
        pc_type='shape',
        mean_params=SHAPE_MEANS,
        std_vector=SHAPE_STDS,
        num_train=NUM_SAMPLES['normal_shape']['train'],
        num_valid=NUM_SAMPLES['normal_shape']['valid'],
        intervals_type=intervals_type,
        clip_min=SHAPE_MIN,
        clip_max=SHAPE_MAX
    )


def sample_uniform_shape(
        min_value: float,
        max_value: float       
    ) -> Tuple[np.ndarray, np.ndarray]:             # (10,)
    """
    Uniform sampling of shape parameters.
    """
    raise NotImplementedError('Need to finish the implementation...')
    #shape = (max_value - min_value) * np.random.randn(10,) + min_value
    #return shape


def sample_normal_style(intervals_type: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    (Truncated) normal sampling of style parameter deviations from the mean, for each garment.
    """
    normal_style_samples = _sample_normal_pc(
        pc_type='style',
        mean_params=STYLE_MEANS,
        std_vector=STYLE_STDS,
        num_train=NUM_SAMPLES['normal_style']['train'] * NUM_GARMENT_CLASSES,
        num_valid=NUM_SAMPLES['normal_style']['valid'] * NUM_GARMENT_CLASSES,
        intervals_type=intervals_type,
        clip_min=STYLE_MIN,
        clip_max=STYLE_MAX
    )
    normal_style_vectors = [
        normal_style_samples[x].reshape(-1, NUM_GARMENT_CLASSES, NUM_STYLE_PARAMS)
        for x in range(2)
    ]
    return tuple(normal_style_vectors)


def sample_uniform_style(
        num_garment_classes: int,
        min_value: float,
        max_value: float
    ) -> Tuple[np.ndarray, np.ndarray]:                  # (num_garment_classes, 4)
    """
    Uniform sampling of style parameters, for each garment.
    """
    raise NotImplementedError('Need to finish the implementation...')
    #style = (max_value - min_value) * np.random.randn(num_garment_classes, 4) + min_value
    #return style


def uniform_random_unit_vector(num_vectors):
    """
    Uniform sampling random 3D unit-vectors, i.e. points on unit sphere.
    """
    e = torch.randn(num_vectors, 3)
    e = torch.div(e, torch.norm(e, dim=-1, keepdim=True))
    return e  # (num_vectors, 3)


@DeprecationWarning
def normal_sample_params_deprecated(batch_size, mean_params, std_vector):
    """
    Gaussian sampling of shape parameter deviations from the mean.
    """
    shape = mean_params + torch.randn(batch_size, mean_params.shape[0], device=mean_params.device)*std_vector
    return shape  # (bs, num_smpl_betas)


def bingham_sampling_for_matrix_fisher_torch(A,
                                             num_samples,
                                             Omega=None,
                                             Gaussian_std=None,
                                             b=1.5,
                                             M_star=None,
                                             oversampling_ratio=8):
    """
    Sampling from a Bingham distribution with 4x4 matrix parameter A.
    Here we assume that A is a diagonal matrix (needed for matrix-Fisher sampling).
    Bing(A) is simulated by rejection sampling from ACG(I + 2A/b) (since ACG > Bingham everywhere).
    Rejection sampling is batched + differentiable (using re-parameterisation trick).

    For further details, see: https://arxiv.org/pdf/1310.8110.pdf and
    https://github.com/tylee-fdcl/Matrix-Fisher-Distribution

    :param A: (4,) tensor parameter of Bingham distribution on 3-sphere.
        Represents the diagonal of a 4x4 diagonal matrix.
    :param num_samples: scalar. Number of samples to draw.
    :param Omega: (4,) Optional tensor parameter of ACG distribution on 3-sphere.
    :param Gaussian_std: (4,) Optional tensor parameter (standard deviations) of diagonal Gaussian in R^4.
    :param num_samples:
    :param b: Hyperparameter for rejection sampling using envelope ACG distribution with
        Omega = I + 2A/b
    :param oversampling_ratio: scalar. To make rejection sampling batched, we sample num_samples * oversampling_ratio,
        then reject samples according to rejection criterion, and hopeffully the number of samples remaining is
        > num_samples.
    :return: samples: (num_samples, 4) and accept_ratio
    """
    assert A.shape == (4,)
    assert A.min() >= 0

    if Omega is None:
        Omega = torch.ones(4, device=A.device) + 2*A/b  # Will sample from ACG(Omega) with Omega = I + 2A/b.
    if Gaussian_std is None:
        Gaussian_std = Omega ** (-0.5)  # Sigma^0.5 = (Omega^-1)^0.5 = Omega^-0.5
    if M_star is None:
        M_star = np.exp(-(4 - b) / 2) * ((4 / b) ** 2)  # Bound for rejection sampling: Bing(A) <= Mstar(b)ACG(I+2A/b)

    samples_obtained = False
    while not samples_obtained:
        eps = torch.randn(num_samples * oversampling_ratio, 4, device=A.device).float()
        y = Gaussian_std * eps
        samples = y / torch.norm(y, dim=1, keepdim=True)  # (num_samples * oversampling_ratio, 4)

        with torch.no_grad():
            p_Bing_star = torch.exp(-torch.einsum('bn,n,bn->b', samples, A, samples))  # (num_samples * oversampling_ratio,)
            p_ACG_star = torch.einsum('bn,n,bn->b', samples, Omega, samples) ** (-2)  # (num_samples * oversampling_ratio,)
            # assert torch.all(p_Bing_star <= M_star * p_ACG_star + 1e-6)

            w = torch.rand(num_samples * oversampling_ratio, device=A.device)
            accept_vector = w < p_Bing_star / (M_star * p_ACG_star)  # (num_samples * oversampling_ratio,)
            num_accepted = accept_vector.sum().item()
        if num_accepted >= num_samples:
            samples = samples[accept_vector, :]  # (num_accepted, 4)
            samples = samples[:num_samples, :]  # (num_samples, 4)
            samples_obtained = True
            accept_ratio = num_accepted / num_samples * 4
        else:
            print('Failed sampling. {} samples accepted, {} samples required.'.format(num_accepted, num_samples))

    return samples, accept_ratio


def pose_matrix_fisher_sampling_torch(pose_U,
                                      pose_S,
                                      pose_V,
                                      num_samples,
                                      b=1.5,
                                      oversampling_ratio=8,
                                      sample_on_cpu=False):
    """
    Sampling from matrix-Fisher distributions defined over SMPL joint rotation matrices.
    MF distribution is simulated by sampling quaternions Bingham distribution (see above) and
    converting quaternions to rotation matrices.

    For further details, see: https://arxiv.org/pdf/1310.8110.pdf and
    https://github.com/tylee-fdcl/Matrix-Fisher-Distribution

    :param pose_U: (B, 23, 3, 3)
    :param pose_S: (B, 23, 3)
    :param pose_V: (B, 23, 3, 3)
    :param num_samples: scalar. Number of samples to draw.
    :param b: Hyperparameter for rejection sampling using envelope ACG distribution.
    :param oversampling_ratio: scalar. To make rejection sampling batched, we sample num_samples * oversampling_ratio,
        then reject samples according to rejection criterion, and hopeffully the number of samples remaining is
        > num_samples.
    :param sample_on_cpu: do sampling on CPU instead of GPU.
    :return: R_samples: (B, num samples, 23, 3, 3)
    """
    batch_size = pose_U.shape[0]
    num_joints = pose_U.shape[1]

    # Proper SVD
    with torch.no_grad():
        detU, detV = torch.det(pose_U.detach().cpu()).to(pose_U.device), torch.det(pose_V.detach().cpu()).to(pose_V.device)
    pose_U_proper = pose_U.clone()
    pose_S_proper = pose_S.clone()
    pose_V_proper = pose_V.clone()
    pose_S_proper[:, :, 2] *= detU * detV  # Proper singular values: s3 = s3 * det(UV)
    pose_U_proper[:, :, :, 2] *= detU.unsqueeze(-1)  # Proper U = U diag(1, 1, det(U))
    pose_V_proper[:, :, :, 2] *= detV.unsqueeze(-1)

    # Sample quaternions from Bingham(A)
    if sample_on_cpu:
        sample_device = 'cpu'
    else:
        sample_device = pose_S_proper.device
    bingham_A = torch.zeros(batch_size, num_joints, 4, device=sample_device)
    bingham_A[:, :, 1] = 2 * (pose_S_proper[:, :, 1] + pose_S_proper[:, :, 2])
    bingham_A[:, :, 2] = 2 * (pose_S_proper[:, :, 0] + pose_S_proper[:, :, 2])
    bingham_A[:, :, 3] = 2 * (pose_S_proper[:, :, 0] + pose_S_proper[:, :, 1])

    Omega = torch.ones(batch_size, num_joints, 4, device=bingham_A.device) + 2 * bingham_A / b  # Will sample from ACG(Omega) with Omega = I + 2A/b.
    Gaussian_std = Omega ** (-0.5)  # Sigma^0.5 = (Omega^-1)^0.5 = Omega^-0.5
    M_star = np.exp(-(4 - b) / 2) * ((4 / b) ** 2)  # Bound for rejection sampling: Bing(A) <= Mstar(b)ACG(I+2A/b)

    pose_quat_samples_batch = torch.zeros(batch_size, num_samples, num_joints, 4, device=pose_U.device).float()
    for i in range(batch_size):
        for joint in range(num_joints):
            quat_samples, accept_ratio = bingham_sampling_for_matrix_fisher_torch(A=bingham_A[i, joint, :],
                                                                                  num_samples=num_samples,
                                                                                  Omega=Omega[i, joint, :],
                                                                                  Gaussian_std=Gaussian_std[i, joint, :],
                                                                                  b=b,
                                                                                  M_star=M_star,
                                                                                  oversampling_ratio=oversampling_ratio)
            pose_quat_samples_batch[i, :, joint, :] = quat_samples

    pose_R_samples_batch = quat_to_rotmat(quat=pose_quat_samples_batch.view(-1, 4)).view(batch_size, num_samples, num_joints, 3, 3)
    pose_R_samples_batch = torch.matmul(pose_U_proper[:, None, :, :, :],
                                        torch.matmul(pose_R_samples_batch, pose_V_proper[:, None, :, :, :].transpose(dim0=-1, dim1=-2)))

    return pose_R_samples_batch


def compute_vertex_uncertainties_by_poseMF_shapeGaussian_sampling(pose_U,
                                                                  pose_S,
                                                                  pose_V,
                                                                  shape_distribution,
                                                                  glob_rotmats,
                                                                  num_samples,
                                                                  smpl_model,
                                                                  use_mean_shape=False):
    """
    Uncertainty = Per-vertex average Euclidean distance from the mean (computed from samples)
    Sampling procedure:
        1) Sample SMPL betas from shape distribution.
        2) Sample pose rotation matrices from pose distribution - matrix fisher M(USV^T)
        3) Pass sampled betas and rotation matrices to SMPL to get full shaped + posed body sample.
    Batch size should be 1 for pose USV and reposed_vertices_distribution.
    :param pose_U: Tensor, (1, 23, 3, 3)
    :param pose_S: Tensor, (1, 23, 3)
    :param pose_V: Tensor, (1, 23, 3, 3)
    :param shape_distribution: torch Normal distribution
    :param glob_rotmats: Tensor (B, 3, 3)
    :param num_samples: int, number of samples to draw
    :param use_mean_shape: bool, use mean shape for samples?
    :return avg_vertices_distance_from_mean: Array, (6890,), average Euclidean distance from mean for each vertex.
    :return vertices_samples
    """
    assert pose_U.shape[0] == pose_S.shape[0] == pose_V.shape[0] == 1  # batch size should be 1
    pose_sample_rotmats = pose_matrix_fisher_sampling_torch(pose_U=pose_U,
                                                            pose_S=pose_S,
                                                            pose_V=pose_V,
                                                            num_samples=num_samples,
                                                            b=1.5,
                                                            oversampling_ratio=8)  # (1, num_samples, 23, 3, 3)
    if use_mean_shape:
        shape_to_use = shape_distribution.loc.expand(num_samples, -1)  # (num_samples, num_shape_params)
    else:
        shape_to_use = shape_distribution.sample([num_samples])[:, 0, :]  # (num_samples, num_shape_params) (batch_size = 1 is indexed out)
    smpl_samples = smpl_model(body_pose=pose_sample_rotmats[0, :, :, :],
                              global_orient=glob_rotmats.unsqueeze(1).expand(num_samples, -1, -1, -1),
                              betas=shape_to_use,
                              pose2rot=False)  # (num_samples, 6890, 3)
    vertices_samples = smpl_samples.vertices
    joints_samples = smpl_samples.joints

    mean_vertices = vertices_samples.mean(dim=0)
    avg_vertices_distance_from_mean = torch.norm(vertices_samples - mean_vertices, dim=-1).mean(dim=0)  # (6890,)

    return avg_vertices_distance_from_mean, vertices_samples, joints_samples


def joints2D_error_sorted_verts_sampling(pred_vertices_samples,
                                         pred_joints_samples,
                                         input_joints2D_heatmaps,
                                         pred_cam_wp):
    """
    Sort 3D vertex mesh samples according to consistency (error) between projected 2D joint samples
    and input 2D joints.
    :param pred_vertices_samples: (N, 6890, 3) tensor of candidate vertex mesh samples.
    :param pred_joints_samples: (N, 90, 3) tensor of candidate J3D samples.
    :param input_joints2D_heatmaps: (1, 17, img_wh, img_wh) tensor of 2D joint locations and confidences.
    :param pred_cam_wp: (1, 3) array with predicted weak-perspective camera.
    :return: pred_vertices_samples_error_sorted: (N, 6890, 3) tensor of J2D-error-sorted vertex mesh samples.
    """
    # Project 3D joint samples to 2D (using COCO joints)
    pred_joints_coco_samples = pred_joints_samples[:, ALL_JOINTS_TO_COCO_MAP, :]
    pred_joints_coco_samples = aa_rotate_translate_points_pytorch3d(points=pred_joints_coco_samples,
                                                                    axes=torch.tensor([1., 0., 0.], device=pred_vertices_samples.device).float(),
                                                                    angles=np.pi,
                                                                    translations=torch.zeros(3, device=pred_vertices_samples.device).float())
    pred_joints2D_coco_samples = orthographic_project_torch(pred_joints_coco_samples, pred_cam_wp)
    pred_joints2D_coco_samples = undo_keypoint_normalisation(pred_joints2D_coco_samples,
                                                             input_joints2D_heatmaps.shape[-1])

    # Convert input 2D joint heatmaps into coordinates
    input_joints2D_coco, input_joints2D_coco_vis = convert_heatmaps_to_2Djoints_coordinates_torch(joints2D_heatmaps=input_joints2D_heatmaps,
                                                                                                  eps=1e-6)  # (1, 17, 2) and (1, 17)

    # Gather visible 2D joint samples and input
    pred_visible_joints2D_coco_samples = pred_joints2D_coco_samples[:, input_joints2D_coco_vis[0], :]  # (N, num vis joints, 2)
    input_visible_joints2D_coco = input_joints2D_coco[:, input_joints2D_coco_vis[0], :]  # (1, num vis joints, 2)

    # Compare 2D joint samples and input using Euclidean distance on image plane.
    j2d_l2es = torch.norm(pred_visible_joints2D_coco_samples - input_visible_joints2D_coco, dim=-1)  # (N, num vis joints)
    j2d_l2e_max, _ = torch.max(j2d_l2es, dim=-1)  # (N,)  # Max joint L2 error for each sample
    _, error_sort_idx = torch.sort(j2d_l2e_max, descending=False)

    pred_vertices_samples_error_sorted = pred_vertices_samples[error_sort_idx]

    return pred_vertices_samples_error_sorted
