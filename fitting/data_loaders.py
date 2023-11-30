from typing import Dict, Tuple
import numpy as np
import os
import cv2

MASKS_DIR = 'demo/masks/'
PARAMS_DIR = 'demo/npz/'


def prepare_gar():
    '''
    Prepares GAR silhouettes and parameters.

    The files are directly copied from the GAR data directory. In particular,
    the silhouette masks are copied from segmentations/ and the parameter
    file (values.npy) is copied from the upper directory. Below are the details
    on how the files are prepared for consecutive loading.

    First, the silhouettes are prepare based on the .npz files copied from the
    dataset folders of GAR. Each .npz file contains lower, upper, and whole
    body and clothing silhouettes. To prepare them for loaders, each of these
    masks are stored in a separate file, named explicitly. The original file
    names are kept in a list and the new files are stored in masks/ directory. 
    The indices of the samples are extracted from the file names.

    Second, the parameter values are extracted from the values.npy file, 
    containing all the values of the validation dataset. In particular, the
    values specified by the previously extracted indices are selected. They
    are stored in separate .npz files and they are named in the same way as
    the original mask files, but stored in a npz/ directory. Each file contains
    pose, shape, upper and lower style parameters.

    Finally, the base names of the files follow the names of the images
    provided in the images/ directory. Note that images are not for optimization
    whatsover, only for reference and for the final qualitative visualization.
    '''
    DATASET = 'gar'
    masks_files = []
    mask_subdir = os.path.join(MASKS_DIR, DATASET)
    params_subdir = os.path.join(PARAMS_DIR, DATASET)
    # Prepare masks
    for masks_npz in [x for x in os.listdir(mask_subdir) if '.npz' in x]:
        masks_files.append(masks_npz)
        masks = np.load(
            os.path.join(mask_subdir, masks_npz), 
            allow_pickle=True
        )['seg_maps']
        # NOTE: Lower and upper are accidentantaly swapped during data generation.
        for mask_idx, mask_part in enumerate(['lower', 'upper', 'whole']):
            mask_name = f'{masks_npz[:-4]}_{mask_part}.png'
            mask_png = np.tile(masks[mask_idx][:, :, np.newaxis], (1, 1, 3)).astype(np.float32)
            cv2.imwrite(
                os.path.join(mask_subdir, mask_name), 
                mask_png
            )
    
    # Prepare param values
    sample_idxs = [int(x.split('.')[0]) for x in masks_files]
    npz_data = np.load(
        os.path.join(params_subdir, 'values.npy'), 
        allow_pickle=True
    ).item()
    for idx, sample_idx in enumerate(sample_idxs):
        params_dict = {
            'pose': npz_data['poses'][sample_idx],
            'shape': npz_data['shapes'][sample_idx],
            'upper_style': npz_data['style_vectors'][sample_idx, 0],
            'lower_style': npz_data['style_vectors'][sample_idx, 1],
        }
        np.savez(
            os.path.join(params_subdir, masks_files[idx]), 
            **params_dict
        )


def load_gt(
        img_name: str,
        dataset: str
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    masks_dict = {}
    for part_label in ['lower', 'upper', 'whole']:
        masks_dict[part_label] = cv2.imread(os.path.join(
            MASKS_DIR,
            dataset,
            f'{img_name.split(".")[0]}_{part_label}.png'
        ))
    params_dict = np.load(
        os.path.join(PARAMS_DIR, dataset, f'{img_name[:-4]}.npz'), 
        allow_pickle=True
    )
    return masks_dict, params_dict
