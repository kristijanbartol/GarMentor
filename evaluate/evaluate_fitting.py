from typing import Dict
import numpy as np
from sklearn.neighbors import NearestNeighbors


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


# From Sergey Prokudin:
# https://gist.github.com/sergeyprokudin/c4bf4059230da8db8256e36524993367
def calc_chamfer_distance(x, y, metric='l2', direction='bi'):
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


def evaluate_meshes(
        pred_cloth_verts,
        gt_cloth_verts, 
        smpl_output
    ) -> Dict:
    body_verts = smpl_output.vertices[0].detach().cpu().numpy()
    pred_full_verts = np.concatenate([
        body_verts,
        pred_cloth_verts
    ], axis=0)
    gt_full_verts = np.concatenate([
        body_verts,
        gt_cloth_verts
    ], axis=0)
    return {
        'cd_cloth': calc_chamfer_distance(pred_cloth_verts, gt_cloth_verts),
        'cd_full': calc_chamfer_distance(pred_full_verts, gt_full_verts),
        'bcc_3d': bcc(body_verts, pred_cloth_verts, gt_cloth_verts)
    }


def evaluate(
        pred_style_params,
        gt_style_params,
        pose_params,
        shape_params,
        smpl_output,
        parametric_model,
        garment_part
    ):
    pred_cloth, _ = parametric_model.run(
        pose=pose_params,
        shape=shape_params,
        style_vector=pred_style_params[0 if garment_part == 'upper' else 1],
        smpl_output=smpl_output,
        garment_part=garment_part
    )
    gt_cloth, _ = parametric_model.run(
        pose=pose_params,
        shape=shape_params,
        style_vector=gt_style_params[0 if garment_part == 'upper' else 1],
        smpl_output=smpl_output,
        garment_part=garment_part
    )
    eval_dict = evaluate_meshes(
        pred_cloth_verts=pred_cloth.detach().cpu().numpy(),
        gt_cloth_verts=gt_cloth.detach().cpu().numpy(),
        smpl_output=smpl_output
    )
    print(f'CD: {eval_dict["cd_cloth"] * 1000.}')
    print(f'CD with body: {eval_dict["cd_full"] * 1000.}')
    print(f'BCC-3D: {eval_dict["bcc_3d"]}')
