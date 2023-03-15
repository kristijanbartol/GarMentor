import os
import sys
import pickle
import numpy as np
from os import path as osp
from tqdm import tqdm

sys.path.append('/garmentor/')

from data.mesh_managers.textured_garments import TexturedGarmentsMeshManager
from models.parametric_model import ParametricModel
from models.smpl_conversions import smplx2smpl
from utils.augmentation.smpl_augmentation import normal_sample_style_numpy
from utils.garment_classes import GarmentClasses
from configs.poseMF_shapeGaussian_net_config import get_cfg_defaults

from data.const import (
    GARMENTOR_DIR,
    SUBJECT_OBJ_SAVEDIR,
    MGN_DATASET,
    SCANS_DIR
)


def _is_already_processed(output_dir: str, pickle_fpath: str) -> bool:
    """Checks whether the given pickle file has already been processed by 
    looking for the corresponding output files in the provided folder
    """
    filename = osp.basename(osp.splitext(pickle_fpath)[0])  # e.g. 10004_w_Amaya_0_0
    # These files must be present if the subject has already been processed
    required_extensions = [
        '-body.jpg', '-body.mtl', '-body.obj',
        '-upper.jpg', '-upper.mtl', '-upper.obj',
        '-lower.jpg', '-lower.mtl', '-lower.obj'
    ]
    required_files = [filename+extension for extension in required_extensions]  # e.g. 10004_w_Amaya_0_0-body.jpg
    for file in required_files:
        if not osp.isfile(osp.join(output_dir, file)):    # e.g. /data/garmentor/agora/subjects/shirt-pant/trainset_3dpeople_adults_bfh/10004_w_Amaya_0_0-body.jpg
            return False
    return True

def _save_style_parameters(
    style_upper: np.ndarray,
    style_lower: np.ndarray,
    output_dir: str,
    #dir_hierarchy: List[str],
    base_dir: str,
    subject_fname: str
    ) -> None:
    """Saves the garment style parameters to a file where they can be retrieved
    from later on. Currently hardcoded to work with 4 parameters (first 2 for
    upper and last 2 for lower body garments)
    Args:
        style_upper (np.ndarray): Style parameters for the upper garment
        style_lower (np.ndarray): Style parameters for the lower garment
        output_dir (str): The directory where the style parameter file should
            be saved to
        dir_hierarchy (List[str]): Each element represents one directory in the
            hierarchy that the subject, for which the style parameters are
            saved, is contained in. The first entry represents the root of this
            hierarchy. Example: ['trainset_3dpeople_adults_bfh'] will read as
            'trainset_3dpeople_adults_bfh/subject.obj' hierarchy.
        base_dir (str): Directory where the subject resides in. Currently only
            supports exactly one directory.

    """
    pkl_fpath = osp.join(output_dir, 'style_parameters.pkl')
    if osp.isfile(pkl_fpath):
        with open(pkl_fpath, 'rb') as pkl_file:
            param_file = pickle.load(pkl_file)
    else:
        param_file = {}
    if not base_dir in param_file.keys():
        param_file[base_dir] = {}
    param_file[base_dir][subject_fname] = {
        'upper': style_upper,
        'lower': style_lower
    }
    pkl_fpath_buffer = osp.splitext(pkl_fpath)[0] + "_buffer.pkl"
    with open(pkl_fpath_buffer, 'wb') as pkl_file:
        pickle.dump(param_file, pkl_file)
    os.replace(pkl_fpath_buffer, pkl_fpath)


if __name__ == '__main__':

    cfg = get_cfg_defaults()

    mean_style = np.zeros(cfg.MODEL.NUM_STYLE_PARAMS, 
                        dtype=np.float32)
    delta_style_std_vector = np.ones(
        cfg.MODEL.NUM_STYLE_PARAMS,
        dtype=np.float32
    ) * cfg.TRAIN.SYNTH_DATA.AUGMENT.GARMENTOR.STYLE_STD

    UPPER_GARMENT_TYPE = "t-shirt"
    LOWER_GARMENT_TYPE = "short-pant"

    garment_classes = GarmentClasses(
        upper_class=UPPER_GARMENT_TYPE,
        lower_class=LOWER_GARMENT_TYPE
    )
    subject_garment_subdir = str(garment_classes)

    parametric_models = {
        'male': ParametricModel(
            gender='male', 
            garment_classes=garment_classes
        ),
        'female': ParametricModel(
            gender='female',
            garment_classes=garment_classes
        )
    }
    
    texture_dirpaths = [
        os.path.join(MGN_DATASET, x) for x in os.listdir(MGN_DATASET)
    ]

    textured_mesh_manager = TexturedGarmentsMeshManager(store_maps_to_disk=True)
    
    if not os.path.exists(GARMENTOR_DIR):
        os.makedirs(GARMENTOR_DIR)
    
    invalid_subjects = []
    for scan_dir in [x for x in os.listdir(SCANS_DIR) if 'kids' not in x]:
        scan_dirpath = os.path.join(SCANS_DIR, scan_dir)
        for pkl_fname in tqdm(
            [x for x in os.listdir(scan_dirpath) \
                if osp.splitext(x)[1] == '.pkl'],
            desc=f"Processing {scan_dir}"
        ):
            pkl_fpath = os.path.join(scan_dirpath, pkl_fname)   # e.g. /data/agora/smplx_gt/trainset_3dpeople_adults_bfh/10004_w_Amaya_0_0.pkl

            mesh_dir = os.path.join(
                SUBJECT_OBJ_SAVEDIR,
                subject_garment_subdir,
                scan_dir
            )
            os.makedirs(mesh_dir, exist_ok=True)
            mesh_basename = osp.splitext(pkl_fname)[0]      

            # Check if this subject (with the current garment combination) is
            # already present in the output directory
            if _is_already_processed(mesh_dir, pkl_fpath):
                print(
                    f"Subject {osp.join(scan_dir, mesh_basename)} already "
                    "processed, skipping..."
                )
                continue

            with open(pkl_fpath, 'rb') as pkl_f:
                metadata = pickle.load(pkl_f)
                try:
                    theta = np.concatenate(
                        [metadata['global_orient'],metadata['body_pose']],
                        axis=1
                    )
                    beta = metadata['betas']
                    gender = metadata['gender']
                except KeyError as e:
                    print(
                        f"Subject {osp.join(scan_dir, mesh_basename)} is "
                        f"missing a required attribute ({e}), skipping...")
                    invalid_subjects.append(osp.join(scan_dir, mesh_basename))
                    continue
                
            beta, theta = smplx2smpl(
                '/data/hierprob3d/',
                '/data/garmentor/conversion_output/',
                beta,
                theta,
                gender=gender,
                overwrite_previous_output=True
            )
            
            theta = theta[0]
            beta = beta[0]
                
            style_vector = normal_sample_style_numpy(
                num_garment_classes=GarmentClasses.NUM_CLASSES,
                mean_params=mean_style,
                std_vector=delta_style_std_vector)

            # decode style parameter for saving
            style_upper = \
                style_vector[parametric_models[gender].labels['upper']]
            style_lower = \
                style_vector[parametric_models[gender].labels['lower']]

            _save_style_parameters(
                style_upper,
                style_lower,
                osp.join(
                    SUBJECT_OBJ_SAVEDIR,
                    subject_garment_subdir
                ),
                scan_dir,
                mesh_basename
            )

            smpl_output_dict = parametric_models[gender].run(
                pose=theta,
                shape=beta,
                style_vector=style_vector
            )
            meshes = textured_mesh_manager.create_meshes(
                smpl_output_dict=smpl_output_dict
            )
            textured_meshes = textured_mesh_manager.texture_meshes(
                meshes=meshes,
                garment_classes=garment_classes
            )
            mesh_basepath = os.path.join(mesh_dir, mesh_basename)
            textured_mesh_manager.save_meshes(
                meshes=textured_meshes,
                rel_path=mesh_basepath
            )
            textured_mesh_manager.add_usemtl(mesh_basepath)

    if len(invalid_subjects) > 0:
        print(
            "The following subjects were invalid and could not be processed, "
            "they have been excluded from the dataset:")
        with open(
            os.path.join(
                SUBJECT_OBJ_SAVEDIR,
                subject_garment_subdir,
                "invalid_subjects.txt"
            ),
            "w"
        ) as file:
            for sub in invalid_subjects:
                print(sub)
                file.write(f"{sub}\n")
