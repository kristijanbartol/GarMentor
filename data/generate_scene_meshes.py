import os
import argparse
import numpy as np
from itertools import compress
import pickle5
import sys

sys.path.append('/garmentor/')

from psbody.mesh import Mesh

from data.const import (
    SCENE_OBJ_SAVEDIR,
    SUBJECT_OBJ_SAVEDIR,
    TRAIN_CAM_DIR
)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--part_idx', '-I', type=int,
                        help='Part index specifying PKL file index for camera information.')
    parser.add_argument('--scene_idx', '-S', type=int,
                        help='Scene index inside the specified part.')
    args = parser.parse_args()
    
    pkl_fname = f'train_{args.part_idx}.pkl'
    pkl_fpath = os.path.join(TRAIN_CAM_DIR, pkl_fname)
    
    with open(pkl_fpath, 'rb') as pkl_f:
        data = pickle5.load(pkl_f)
        num_scenes = data['X'].shape[0]
        
        if args.scene_idx >= num_scenes:
            print(f'ERROR: The selected part has only {num_scenes} scenes. '
                  f'Adjust your scene index to be < {num_scenes}.')
        
        kids = data['kid'].iloc[args.scene_idx]
        not_kids = [not kid for kid in kids]
        
        # Select only the non-kids.
        Xs = list(compress(
            data['X'].iloc[args.scene_idx],
            not_kids)
        )
        Ys = list(compress(
            data['Y'].iloc[args.scene_idx],
            not_kids)
        )
        Zs = list(compress(
            data['Z'].iloc[args.scene_idx],
            not_kids)
        )
        gt_paths_smplx = list(compress(
            data['gt_path_smplx'].iloc[args.scene_idx],
            not_kids)
        )
        
        camX = data['camX'].iloc[args.scene_idx]
        camY = data['camY'].iloc[args.scene_idx]
        camZ = data['camZ'].iloc[args.scene_idx]
        camYaw = data['camYaw'].iloc[args.scene_idx]
        img_path = data['imgPath'].iloc[args.scene_idx]
        
        output_dirpath = os.path.join(
            SCENE_OBJ_SAVEDIR,
            img_path.split('.')[0]
        )
        
        for subject_idx, gt_path_smplx in enumerate(gt_paths_smplx):
            rel_gt_path_smplx = os.path.relpath(gt_path_smplx, 'smplx_gt')
            rel_gt_path_smplx = rel_gt_path_smplx.split('.')[0]
            mesh_basepath = os.path.join(SUBJECT_OBJ_SAVEDIR, rel_gt_path_smplx)
            
            body_mesh = Mesh(filename=f'{mesh_basepath}-body.obj')
            upper_mesh = Mesh(filename=f'{mesh_basepath}-upper.obj')
            lower_mesh = Mesh(filename=f'{mesh_basepath}-lower.obj')
            
            trans = np.array([
                Xs[subject_idx],
                Ys[subject_idx],
                Zs[subject_idx]
            ])

            body_mesh.translate_vertices(trans)
            upper_mesh.translate_vertices(trans)
            lower_mesh.translate_vertices(trans)
            
            body_mesh.write_obj(output_dirpath, f'{subject_idx:02d}-body.obj')
            upper_mesh.write_obj(output_dirpath, f'{subject_idx:02d}-upper.obj')
            lower_mesh.write_obj(output_dirpath, f'{subject_idx:02d}-lower.obj')
