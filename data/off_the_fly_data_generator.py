import numpy as np
import torch
import os
import cv2

from configs import paths
from configs.poseMF_shapeGaussian_net_config import get_poseMF_shapeGaussian_cfg_defaults
from utils.augmentation.smpl_augmentation import normal_sample_params_numpy
from utils.augmentation.cam_augmentation import augment_cam_t_numpy

from tailornet_for_garmentor.models.tailornet_model import get_best_runner as get_tn_runner
from tailornet_for_garmentor.models.smpl4garment import SMPL4Garment
from tailornet_for_garmentor.utils.rotation import normalize_y_rotation
from tailornet_for_garmentor.utils.interpenetration import original_remove_interpenetration_fast


if __name__ == '__main__':
    # Generate off-the-fly data for specific gender.
    gender = 'male'

    # Create SMPL (+garment mesh) and TailorNet gender-specific models.
    # NOTE: SMPL4Garment is the original, chumpy-based class, which is desired here.
    smpl_model = SMPL4Garment(gender=gender)
    tailornet_model = get_tn_runner(gender=gender, garment_class='t-shirt')

    # Load configurations.
    pose_shape_cfg = get_poseMF_shapeGaussian_cfg_defaults()

    # Useful arrays that are re-used and can be pre-defined.
    x_axis = np.array([1., 0., 0.], dtype=np.float32)
    delta_betas_std_vector = np.ones(pose_shape_cfg.MODEL.NUM_SMPL_BETAS, dtype=np.float32) * pose_shape_cfg.TRAIN.SYNTH_DATA.AUGMENT.SMPL.SHAPE_STD
    mean_shape = np.zeros(pose_shape_cfg.MODEL.NUM_SMPL_BETAS, dtype=np.float32)
    delta_style_std_vector = np.ones(pose_shape_cfg.MODEL.NUM_STYLE_PARAMS, dtype=np.float32) * pose_shape_cfg.TRAIN.SYNTH_DATA.AUGMENT.GARMENTOR.STYLE_STD
    mean_style = np.zeros(pose_shape_cfg.MODEL.NUM_STYLE_PARAMS, dtype=np.float32)
    mean_cam_t = np.tensor(pose_shape_cfg.TRAIN.SYNTH_DATA.MEAN_CAM_T, dtype=np.float32)
    mean_cam_t = np.broadcast_to(mean_cam_t[None, :], (pose_shape_cfg.TRAIN.BATCH_SIZE, -1))
    
    # Load SMPL poses.
    # NOTE (kbartol): Because the poses are loaded, they do not have to be stored again afterwards.
    data = np.load(paths.TRAIN_POSES_PATH)
    fnames = data['fnames']
    poses = data['poses']
    
    # Not amass poses.
    # (kbartol): OK, not amass.
    indices = [i for i, x in enumerate(fnames)
                if (x.startswith('h36m') or x.startswith('up3d') or x.startswith('3dpw'))]
    fnames = [fnames[i] for i in indices]
    poses = [poses[i] for i in indices]

    # Number of samples is based on the number of different poses, for now.
    # TODO: Increase the number of samples to obtain the diversity of shapes for each pose.
    num_samples = poses.shape[0]
    
    # Finally got the poses.
    poses = np.stack(poses, axis=0)
    
    # Load LSUN backgrounds.
    # NOTE (kbartol): We don't actually need to store the backgrounds, nor the RGB information.
    # NOTE (kbartol): The RGB information is only used by the edge detector.
    backgrounds_paths = sorted([os.path.join(paths.TRAIN_BACKGROUNDS_PATH, f)
                                        for f in os.listdir(paths.TRAIN_BACKGROUNDS_PATH)
                                        if f.endswith('.jpg')])
    img_wh = pose_shape_cfg.DATA.PROXY_REP_SIZE

    # Parameters and smaller data variables for storing.
    betas = []
    gammas = []
    joints_2Ds = []
    joints_3Ds = []
    # NOTE: The remaining data is previously stored or too large to save at once:
    #       - backgrounds
    #       - textures
    #       - RGB images
    #       - cloth segmentations
    
    for idx in range(poses.shape[0]):
        # Target pose.
        target_pose = poses[idx]

        # Convert target_pose from axis angle to rotmats.
        target_pose_rotmats = batch_rodrigues(target_pose.contiguous().view(-1, 3)).view(-1, 24, 3, 3)
        target_glob_rotmats = target_pose_rotmats[:, 0, :, :]
        target_pose_rotmats = target_pose_rotmats[:, 1:, :, :]
        
        # TODO (kbartol): Particularly verify these rotations.
        # Flipping pose targets such that they are right way up in 3D space - i.e. wrong way up when projected
        # Then pose predictions will also be right way up in 3D space - network doesn't need to learn to flip.
        _, target_glob_rotmats = aa_rotate_rotmats_pytorch3d(rotmats=target_glob_rotmats,
                                                                angles=np.pi,
                                                                axes=x_axis,
                                                                rot_mult_order='post')

        ###########
        ## BETAS ##
        ###########
        # Randomly sample body shape.
        target_shape = normal_sample_params_numpy(batch_size=pose_shape_cfg.TRAIN.BATCH_SIZE,
                                                  mean_params=mean_shape,
                                                  std_vector=delta_betas_std_vector)
        betas.append(target_shape)

        ############
        ## GAMMAS ##
        ############
        # Randomly sample garment parameters.
        target_style = normal_sample_params_numpy(batch_size=pose_shape_cfg.TRAIN.BATCH_SIZE,
                                                  mean_params=mean_style,
                                                  std_vector=delta_style_std_vector)
        gammas.append(target_style)

        # Random sample camera translation.
        target_cam_t = augment_cam_t_numpy(mean_cam_t,
                                        xy_std=pose_shape_cfg.TRAIN.SYNTH_DATA.AUGMENT.CAM.XY_STD,
                                        delta_z_range=pose_shape_cfg.TRAIN.SYNTH_DATA.AUGMENT.CAM.DELTA_Z_RANGE)

        # Compute parameterized clothing displacements.
        theta_normalized = normalize_y_rotation(target_pose)
        with torch.no_grad():
            # Run TailorNet model to get the displacements based on the provided parameters.
            pred_verts_d = tailornet_model.forward(
                thetas=torch.from_numpy(theta_normalized[None, :].astype(np.float32)).cuda(),
                betas=torch.from_numpy(target_shape[None, :].astype(np.float32)).cuda(),
                gammas=torch.from_numpy(target_style[None, :].astype(np.float32)).cuda(),
            )[0].cpu().numpy()
            
        # Get the predicted body and garment meshes - simply SMPL + the above displacements.
        body, pred_gar = smpl_model.run(beta=target_shape, theta=target_pose, garment_class='t-shirt', garment_d=pred_verts_d)
        pred_gar = original_remove_interpenetration_fast(pred_gar, body)

        # Finally, take only the vertices of target body and garment meshes.
        target_body_verts = body.v
        target_garment_verts = pred_gar.v

        # Create target joint variables.
        target_joints = target_smpl_output.joints
        target_joints_h36m = target_joints[:, BASE_JOINTS_TO_H36M_MAP]
        target_joints_h36mlsp = target_joints_h36m[:, H36M_TO_J14, :]

        # ------------ INPUT PROXY REPRESENTATION GENERATION + 2D TARGET JOINTS ------------
        # Pose targets were flipped such that they are right way up in 3D space - i.e. wrong way up when projected
        # Need to flip target_vertices_for_rendering 180° about x-axis so they are right way up when projected
        # Need to flip target_joints_coco 180° about x-axis so they are right way up when projected

        # TODO (kbartol): Particularly verify these rotations.
        target_body_vertices_for_rendering = aa_rotate_translate_points_pytorch3d(points=target_body_verts,
                                                                                axes=x_axis,
                                                                                angles=np.pi,
                                                                                translations=torch.zeros(3, device=device).float())
        target_garment_vertices_for_rendering = aa_rotate_translate_points_pytorch3d(points=target_garment_verts,
                                                                                axes=x_axis,
                                                                                angles=np.pi,
                                                                                translations=torch.zeros(3, device=device).float())
        target_joints_coco = aa_rotate_translate_points_pytorch3d(points=target_joints[:, BASE_JOINTS_TO_COCO_MAP],
                                                                    axes=x_axis,
                                                                    angles=np.pi,
                                                                    translations=torch.zeros(3, device=device).float())
        target_joints2d_coco = perspective_project_torch(target_joints_coco,
                                                            None,
                                                            target_cam_t,
                                                            focal_length=pose_shape_cfg.TRAIN.SYNTH_DATA.FOCAL_LENGTH,
                                                            img_wh=pose_shape_cfg.DATA.PROXY_REP_SIZE)

        # Check if joints are within image dimensions before cropping + recentering.
        target_joints2d_visib_coco = check_joints2d_visibility_torch(target_joints2d_coco,
                                                                        pose_shape_cfg.DATA.PROXY_REP_SIZE)  # (batch_size, 17)

        # TODO (kbartol): Move all the augmentations to the training part.

        # Set up light conditions.
        # NOTE (kbartol): Light conditions can be augmented only in training time.
        # NOTE (kbartol): This basically means that light conditions will not be augmented,
        #                 otherwise, the image also have to be rendered in training time.
        lights_rgb_settings = augment_light(batch_size=1,
                                            device=device,
                                            rgb_augment_config=pose_shape_cfg.TRAIN.SYNTH_DATA.AUGMENT.RGB)
        
        renderer_output = pytorch3d_renderer(body_vertices=target_body_vertices_for_rendering,
                                                garment_vertices=target_garment_vertices_for_rendering,
                                                cam_t=target_cam_t,
                                                lights_rgb_settings=lights_rgb_settings)
        
        iuv_in = renderer_output['iuv_images'].permute(0, 3, 1, 2).contiguous()  # (bs, 3, img_wh, img_wh)
        iuv_in[:, 1:, :, :] = iuv_in[:, 1:, :, :] * 255
        iuv_in = iuv_in.round()
        rgb_in = renderer_output['rgb_images'].permute(0, 3, 1, 2).contiguous()  # (bs, 3, img_wh, img_wh)

        # Prepare seg for extreme crop augmentation
        seg_extreme_crop = random_extreme_crop(seg=iuv_in[:, 0, :, :],
                                                extreme_crop_probability=pose_shape_cfg.TRAIN.SYNTH_DATA.AUGMENT.PROXY_REP.EXTREME_CROP_PROB)

        # Crop to person bounding box after bbox scale and centre augmentation
        crop_outputs = batch_crop_pytorch_affine(input_wh=(pose_shape_cfg.DATA.PROXY_REP_SIZE, pose_shape_cfg.DATA.PROXY_REP_SIZE),
                                                    output_wh=(pose_shape_cfg.DATA.PROXY_REP_SIZE, pose_shape_cfg.DATA.PROXY_REP_SIZE),
                                                    num_to_crop=pose_shape_cfg.TRAIN.BATCH_SIZE,
                                                    device=device,
                                                    rgb=rgb_in,
                                                    iuv=iuv_in,
                                                    joints2D=target_joints2d_coco,
                                                    bbox_determiner=seg_extreme_crop,
                                                    orig_scale_factor=pose_shape_cfg.DATA.BBOX_SCALE_FACTOR,
                                                    delta_scale_range=pose_shape_cfg.TRAIN.SYNTH_DATA.AUGMENT.BBOX.DELTA_SCALE_RANGE,
                                                    delta_centre_range=pose_shape_cfg.TRAIN.SYNTH_DATA.AUGMENT.BBOX.DELTA_CENTRE_RANGE,
                                                    out_of_frame_pad_val=-1)
        iuv_in = crop_outputs['iuv']
        target_joints2d_coco = crop_outputs['joints2D']
        rgb_in = crop_outputs['rgb']

        # Check if joints are within image dimensions after cropping + recentering.
        target_joints2d_visib_coco = check_joints2d_visibility_torch(target_joints2d_coco,
                                                                        pose_shape_cfg.DATA.PROXY_REP_SIZE,
                                                                        visibility=target_joints2d_visib_coco)  # (bs, 17)
        # Check if joints are occluded by the body.
        seg_14_part_occlusion_check = convert_densepose_seg_to_14part_labels(iuv_in[:, 0, :, :])
        target_joints2d_visib_coco = check_joints2d_occluded_torch(seg_14_part_occlusion_check,
                                                                    target_joints2d_visib_coco,
                                                                    pixel_count_threshold=50)  # (bs, 17)

        # Apply segmentation/IUV-based render augmentations + 2D joints augmentations
        seg_aug, target_joints2d_coco_input, target_joints2d_visib_coco = augment_proxy_representation(
            seg=iuv_in[:, 0, :, :],  # Note: out of frame pixels marked with -1
            joints2D=target_joints2d_coco,
            joints2D_visib=target_joints2d_visib_coco,
            proxy_rep_augment_config=pose_shape_cfg.TRAIN.SYNTH_DATA.AUGMENT.PROXY_REP)

        # Add background to input RGB.
        bg_idx = np.randint(low=0, high=len(backgrounds_paths), size=(1,)).item()
        bg_path = backgrounds_paths[bg_idx]
        background = cv2.cvtColor(cv2.imread(bg_path), cv2.COLOR_BGR2RGB)
        background = cv2.resize(background, (img_wh, img_wh), interpolation=cv2.INTER_LINEAR)
        background = background.transpose(2, 0, 1)
        background = (background / 255.).float()

        rgb_in = batch_add_rgb_background(backgrounds=background,
                                            rgb=rgb_in,
                                            seg=seg_aug)

        # Apply RGB-based render augmentations + 2D joints augmentations
        rgb_in, target_joints2d_coco_input, target_joints2d_visib_coco = augment_rgb(rgb=rgb_in,
                                                                                        joints2D=target_joints2d_coco_input,
                                                                                        joints2D_visib=target_joints2d_visib_coco,
                                                                                        rgb_augment_config=pose_shape_cfg.TRAIN.SYNTH_DATA.AUGMENT.RGB)
        # Compute edge-images edges
        edge_detector_output = edge_detect_model(rgb_in)
        edge_in = edge_detector_output['thresholded_thin_edges'] if pose_shape_cfg.DATA.EDGE_NMS else edge_detector_output['thresholded_grad_magnitude']

        # Compute 2D joint heatmaps
        j2d_heatmaps = convert_2Djoints_to_gaussian_heatmaps_torch(target_joints2d_coco_input,
                                                                    pose_shape_cfg.DATA.PROXY_REP_SIZE,
                                                                    std=pose_shape_cfg.DATA.HEATMAP_GAUSSIAN_STD)
        j2d_heatmaps = j2d_heatmaps * target_joints2d_visib_coco[:, :, None, None]

        # Concatenate edge-image and 2D joint heatmaps to create input proxy representation
        proxy_rep_input = torch.cat([edge_in, j2d_heatmaps], dim=1).float()  # (batch_size, C, img_wh, img_wh)

    # Store pre-generated data.
    store_dict = {
        'gender': np.array([gender] * num_samples),
        'thetas': poses,
        'betas': 
    }
