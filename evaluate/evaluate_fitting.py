from typing import Dict
import numpy as np

from utils.eval_utils import (
    calc_chamfer_distance,
    masked_chamfer_distance,
    bcc
)


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
