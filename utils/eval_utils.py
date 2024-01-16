"""
Parts of the code are adapted from https://github.com/akanazawa/hmr
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors


def compute_similarity_transform(S1, S2):
    """
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    """
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert(S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1**2)

    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R.
    R = V.dot(Z.dot(U.T))

    # 5. Recover scale.
    scale = np.trace(R.dot(K)) / var1

    # 6. Recover translation.
    t = mu2 - scale*(R.dot(mu1))

    # 7. Error:
    S1_hat = scale*R.dot(S1) + t

    if transposed:
        S1_hat = S1_hat.T

    return S1_hat


def procrustes_analysis_batch(S1, S2):
    """Batched version of compute_similarity_transform."""
    S1_hat = np.zeros_like(S1)
    for i in range(S1.shape[0]):
        S1_hat[i] = compute_similarity_transform(S1[i], S2[i])
    return S1_hat


def pa_mpjpe(predicted, target):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    assert predicted.shape == target.shape
    
    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)
    
    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0**2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0**2, axis=(1, 2), keepdims=True))
    
    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1)) # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY # Scale
    t = muX - a*np.matmul(muY, R) # Translation
    
    return a, R, t


def scale_and_translation_transform_batch(P, T):
    """
    First Normalises batch of input 3D meshes P such that each mesh has mean (0, 0, 0) and
    RMS distance from mean = 1.
    Then transforms P such that it has the same mean and RMSD as T.
    :param P: (batch_size, N, 3) batch of N 3D meshes to transform.
    :param T: (batch_size, N, 3) batch of N reference 3D meshes.
    :return: P transformed
    """
    P_mean = np.mean(P, axis=1, keepdims=True)
    P_trans = P - P_mean
    P_scale = np.sqrt(np.sum(P_trans ** 2, axis=(1, 2), keepdims=True) / P.shape[1])
    P_normalised = P_trans / P_scale

    T_mean = np.mean(T, axis=1, keepdims=True)
    T_scale = np.sqrt(np.sum((T - T_mean) ** 2, axis=(1, 2), keepdims=True) / T.shape[1])

    P_transformed = P_normalised * T_scale + T_mean

    return P_transformed


def scale_and_translation_transform_batch_torch(P, T):
    """
    First Normalises batch of input 3D meshes P such that each mesh has mean (0, 0, 0) and
    RMS distance from mean = 1.
    Then transforms P such that it has the same mean and RMSD as T.
    :param P: (batch_size, N, 3) batch of N 3D meshes to transform.
    :param T: (batch_size, N, 3) batch of N reference 3D meshes.
    :return: P transformed
    """
    P_mean = torch.mean(P, dim=1, keepdim=True)
    P_trans = P - P_mean
    P_scale = torch.sqrt(torch.sum(P_trans ** 2, dim=(1, 2), keepdim=True) / P.shape[1])
    P_normalised = P_trans / P_scale

    T_mean = torch.mean(T, dim=1, keepdim=True)
    T_scale = torch.sqrt(torch.sum((T - T_mean) ** 2, dim=(1, 2), keepdim=True) / T.shape[1])

    P_transformed = P_normalised * T_scale + T_mean

    return P_transformed


def shape_parameters_to_a_pose(body_shape,
                               smpl):
    """
    Return mesh of person in A-pose, given the body shape parameters.
    :param body_shape:
    :param smpl: SMPL model
    :return:
    """
    a_pose = torch.zeros((1, 69), device=body_shape.device)
    a_pose[:, 47] = -np.pi/3.0
    a_pose[:, 50] = np.pi/3.0

    a_pose_output = smpl(betas=body_shape,
                         body_pose=a_pose)
    a_pose_vertices = a_pose_output.vertices
    return a_pose_vertices


def make_xz_ground_plane(vertices):
    """
    Given a vertex mesh, translates the mesh such that the lowest coordinate of the mesh
    lies on the x-z plane.
    :param vertices: (N, 6890, 3)
    :return:
    """
    lowest_y = vertices[:, :, 1].min(axis=-1, keepdims=True)
    vertices[:, :, 1] = vertices[:, :, 1] - lowest_y
    return vertices


# From Sergey Prokudin:
# https://gist.github.com/sergeyprokudin/c4bf4059230da8db8256e36524993367
def calc_chamfer_distance(x, y, metric='l2', direction='bi'):
    """Chamfer distance between two point clouds
    Parameters
    ----------
    x: numpy array [n_points_x, n_dims]
        first point cloud
    y: numpy array [n_points_y, n_dims]
        second point cloud
    metric: string or callable, default ‘l2’
        metric to use for distance computation. Any metric from scikit-learn or scipy.spatial.distance can be used.
    direction: str
        direction of Chamfer distance.
            'y_to_x':  computes average minimal distance from every point in y to x
            'x_to_y':  computes average minimal distance from every point in x to y
            'bi': compute both
    Returns
    -------
    chamfer_dist: float
        computed bidirectional Chamfer distance:
            sum_{x_i \in x}{\min_{y_j \in y}{||x_i-y_j||**2}} + sum_{y_j \in y}{\min_{x_i \in x}{||x_i-y_j||**2}}
    """
    
    if direction=='y_to_x':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        chamfer_dist = np.mean(min_y_to_x)
    elif direction=='x_to_y':
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_x_to_y)
    elif direction=='bi':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_y_to_x) + np.mean(min_x_to_y)
    else:
        raise ValueError("Invalid direction type. Supported types: \'y_x\', \'x_y\', \'bi\'")
        
    return chamfer_dist


# TODO: Implement more robust BCC (see STEPS).
def bcc(body_verts,
        pred_cloth_verts,
        gt_cloth_verts,
        threshold=0.01
    ):
    ''' BCC metric (ClothWild), implemented with known 3D GT.
    
        Takes predicted and GT vertices and determines the masks individually
        w.r.t. SMPL body. For both predicted and GT vertices, the corresponding
        SMPL vertices are first found and are set as masks. The corresponding
        vertices are simply the closest vertices found by the same NN algorithm 
        that CD uses. Then, the BCC metric is calculated as a ratio between the
        correctly classified vertices and the sum of correctly and incorrectly
        classified points:

        BCC = (# correct) / (# correct + # incorrect)

        Note that there are two types of incorrect classification. One is when
        the vertex is classified as covered while it is not. The other is when
        the vertex should be classified as covered but is not. The sum of these
        two types is the total number of incorrect classifications.

        Other names:
        - BCC-3D
        - (NOTE: naming it BCC would still be correct if properly explained)

        STEPS:
        1. Segment the body into parts.
        2. Assign the corresponding labels for clothing mesh by finding the
           closest body vertex and assign its label.
        3. For each body vertex, determine whether it is covered by cloth.
           It is considered covered if there is at least one vertex that
           satisfies the two conditions:
           3.1 The clothing vertex is within the specified distance.
           3.2 The clothing vertex belongs to the corresponding body part.
        4. Determine covered vertices for both meshes and calculate IoU.

        Simplified function:
        For now, only find the corresponding clothing vertices by proximity.
    '''
    def find_closest_vertex(body_vertex, clothing_verts):
        distances = np.linalg.norm(clothing_verts - body_vertex, axis=1)
        closest_index = np.argmin(distances)
        proximity = np.min(distances)
        return clothing_verts[closest_index], proximity

    pred_body_mask_idxs = set()
    gt_body_mask_idxs = set()
    for bidx, bv in enumerate(body_verts):
        _, dist = find_closest_vertex(bv, pred_cloth_verts)
        if dist < threshold:
            pred_body_mask_idxs.add(bidx)
        _, dist = find_closest_vertex(bv, gt_cloth_verts)
        if dist < threshold:
            gt_body_mask_idxs.add(bidx)

    intersection_idxs = pred_body_mask_idxs & gt_body_mask_idxs
    union_idxs = pred_body_mask_idxs | gt_body_mask_idxs
    iou = float(len(intersection_idxs)) / float(len(union_idxs))

    return iou, intersection_idxs


# TODO: Implement masked CD.
def masked_chamfer_distance(
        body_verts,
        intersection_body_idxs,
        pred_cloth_verts,
        gt_cloth_verts
    ):
    ''' A novel metric.
    
        Takes predicted and GT vertices. First, calculates the Chamfer distance
        between the (masked) predicted vertices and GT vertices. Then, it
        calculates the Chamfer distance between the (masked) GT vertices and
        the predicted vertices. The final distance is an average.

        (???) What is the mask?

        Does it make sense to come up with a new metric? It seems that, if both
        meshes are provided along with the SMPL bodies, then the original CD 
        should work nicely.

        Other possible names:
        - average corresponding Chamfer distance
    '''
    pass

