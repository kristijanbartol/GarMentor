import copy
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from smplx.lbs import batch_rodrigues
from psbody.mesh import Mesh
from tqdm import tqdm

from metrics.train_loss_and_metrics_tracker import TrainingLossesAndMetricsTracker

from utils.checkpoint_utils import load_training_info_from_checkpoint
from utils.cam_utils import perspective_project_torch, orthographic_project_torch
from utils.rigid_transform_utils import rot6d_to_rotmat, aa_rotate_translate_points_pytorch3d, aa_rotate_rotmats_pytorch3d
from utils.label_conversions import ALL_JOINTS_TO_H36M_MAP, convert_2Djoints_to_gaussian_heatmaps_torch, \
    H36M_TO_J14, BASE_JOINTS_TO_COCO_MAP, BASE_JOINTS_TO_H36M_MAP, ALL_JOINTS_TO_COCO_MAP
from utils.joints2d_utils import check_joints2d_visibility_torch, check_joints2d_occluded_torch
from utils.image_utils import batch_add_rgb_background
from utils.augmentation.rgb_augmentation import augment_rgb
from utils.augmentation.proxy_rep_augmentation import augment_proxy_representation


def train_poseMF_shapeGaussian_net(pose_shape_model,
                                   pose_shape_cfg,
                                   smpl_model,
                                   edge_detect_model,
                                   device,
                                   train_dataset,
                                   val_dataset,
                                   criterion,
                                   optimiser,
                                   metrics,
                                   model_save_dir,
                                   logs_save_path,
                                   save_val_metrics=['PVE-SC', 'MPJPE-PA'],
                                   checkpoint=None,
                                   vis_logger=None):
    # Set up dataloaders
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=pose_shape_cfg.TRAIN.BATCH_SIZE,
                                  shuffle=True,
                                  drop_last=True,
                                  num_workers=pose_shape_cfg.TRAIN.NUM_WORKERS,
                                  pin_memory=pose_shape_cfg.TRAIN.PIN_MEMORY)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=pose_shape_cfg.TRAIN.BATCH_SIZE,
                                shuffle=True,
                                drop_last=True,
                                num_workers=pose_shape_cfg.TRAIN.NUM_WORKERS,
                                pin_memory=pose_shape_cfg.TRAIN.PIN_MEMORY)
    dataloaders = {'train': train_dataloader,
                   'val': val_dataloader}

    # Load checkpoint benchmarks if provided
    if checkpoint is not None:
        # Resuming training - note that current model and optimiser state dicts are loaded out
        # of train function (should be in run file).
        current_epoch, best_epoch, best_model_wts, best_epoch_val_metrics = load_training_info_from_checkpoint(checkpoint,
                                                                                                               save_val_metrics)
        load_logs = True
    else:
        current_epoch = 0
        best_epoch = 0
        best_epoch_val_metrics = {}
        # metrics that decide whether to save model after each epoch or not
        for metric in save_val_metrics:
            best_epoch_val_metrics[metric] = np.inf
        best_model_wts = copy.deepcopy(pose_shape_model.state_dict())
        load_logs = False

    # Instantiate metrics tracker
    metrics_tracker = TrainingLossesAndMetricsTracker(metrics_to_track=metrics,
                                                      img_wh=pose_shape_cfg.DATA.PROXY_REP_SIZE,
                                                      log_save_path=logs_save_path,
                                                      load_logs=load_logs,
                                                      current_epoch=current_epoch)

    # Useful tensors that are re-used and can be pre-defined
    x_axis = torch.tensor([1., 0., 0.],
                          device=device, dtype=torch.float32)
    mean_cam_t = torch.tensor(pose_shape_cfg.TRAIN.SYNTH_DATA.MEAN_CAM_T,
                              device=device, dtype=torch.float32)
    mean_cam_t = mean_cam_t[None, :].expand(pose_shape_cfg.TRAIN.BATCH_SIZE, -1)

    # Starting training loop
    for epoch in range(current_epoch, pose_shape_cfg.TRAIN.NUM_EPOCHS):
        print('\nEpoch {}/{}'.format(epoch, pose_shape_cfg.TRAIN.NUM_EPOCHS - 1))
        print('-' * 10)
        metrics_tracker.initialise_loss_metric_sums()

        for split in ['train', 'val']:
            if split == 'train':
                print('Training.')
                pose_shape_model.train()
            else:
                print('Validation.')
                pose_shape_model.eval()

            for _, sample_batch in enumerate(tqdm(dataloaders[split])):
                #############################################################
                # ---------------- SYNTHETIC DATA GENERATION ----------------
                #############################################################
                with torch.no_grad():
                    # ------------ RANDOM POSE, SHAPE, BACKGROUND, TEXTURE, CAMERA SAMPLING ------------
                    # Load target pose and random background/texture
                    target_pose = sample_batch['pose'].to(device)  # (bs, 72)
                    background = sample_batch['background'].to(device)  # (bs, 3, img_wh, img_wh)

                    # Convert target_pose from axis angle to rotmats
                    target_pose_rotmats = batch_rodrigues(target_pose.contiguous().view(-1, 3)).view(-1, 24, 3, 3)
                    target_glob_rotmats = target_pose_rotmats[:, 0, :, :]
                    target_pose_rotmats = target_pose_rotmats[:, 1:, :, :]

                    target_shape = sample_batch['shape'].to(device)    # (bs, 10)

                    target_style_vector = sample_batch['style_vector'].to(device)   # (bs, num_garment_classes=4, 10)
                    assert(target_style_vector.shape[1] == 4)
                    garment_labels = sample_batch['garment_labels'].to(device)      # (bs, num_garment_classes=4)

                    target_cam_t = sample_batch['cam_t'].to(device)    # (bs, 3)
                    
                    target_smpl_output = smpl_model(body_pose=target_pose_rotmats,
                                                    global_orient=target_glob_rotmats.unsqueeze(1),
                                                    betas=target_shape,
                                                    pose2rot=False)

                    target_vertices = target_smpl_output.vertices
                    target_joints2d_coco = target_smpl_output.joints[:, ALL_JOINTS_TO_COCO_MAP]
                    target_joints_h36m = target_smpl_output.joints[:, ALL_JOINTS_TO_H36M_MAP]
                    target_joints_h36mlsp = target_joints_h36m[:, H36M_TO_J14, :]
                    
                    target_reposed_vertices = smpl_model(body_pose=torch.zeros_like(target_pose)[:, 3:],
                                                         global_orient=torch.zeros_like(target_pose)[:, :3],
                                                         betas=target_shape).vertices

                    # ------------ INPUT PROXY REPRESENTATION GENERATION + 2D TARGET JOINTS ------------
                    target_joints2d_coco = perspective_project_torch(target_joints2d_coco,
                                                                     None,
                                                                     target_cam_t,
                                                                     focal_length=pose_shape_cfg.TRAIN.SYNTH_DATA.FOCAL_LENGTH,
                                                                     img_wh=pose_shape_cfg.DATA.PROXY_REP_SIZE)

                    # Check if joints within image dimensions before cropping + recentering.
                    target_joints2d_visib_coco = check_joints2d_visibility_torch(target_joints2d_coco,
                                                                                 pose_shape_cfg.DATA.PROXY_REP_SIZE)  # (batch_size, 17)

                    seg_maps = sample_batch['seg_maps'].to(device)
                    rgb_in = sample_batch['rgb_img'].to(device)
                    
                    # Check if joints are occluded by the body.
                    if pose_shape_cfg.TRAIN.SYNTH_DATA.AUGMENT.PROXY_REP.CHECK_JOINTS_OCCLUDED:
                        target_joints2d_visib_coco = check_joints2d_occluded_torch(seg_maps[:, -1],
                                                                                target_joints2d_visib_coco,
                                                                                pixel_count_threshold=50)  # (bs, 17)

                    # Apply segmentation/IUV-based render augmentations + 2D joints augmentations
                    if pose_shape_cfg.TRAIN.SYNTH_DATA.AUGMENT.PROXY_REP.FLAG:
                        _, target_joints2d_coco_input, target_joints2d_visib_coco = augment_proxy_representation(
                            seg=seg_maps[:, -1],
                            joints2D=target_joints2d_coco,
                            joints2D_visib=target_joints2d_visib_coco,
                            proxy_rep_augment_config=pose_shape_cfg.TRAIN.SYNTH_DATA.AUGMENT.PROXY_REP)
                    
                    # Add background rgb
                    # NOTE: The last seg map (-1) is the whole body seg map.
                    if rgb_in is not None:
                        rgb_in = batch_add_rgb_background(backgrounds=background,
                                                        rgb=rgb_in,
                                                        seg=seg_maps[:, -1])                        
                    
                        # Apply RGB-based render augmentations + 2D joints augmentations
                        rgb_in, target_joints2d_coco_input, target_joints2d_visib_coco = augment_rgb(
                            rgb=rgb_in,
                            joints2D=target_joints2d_coco,
                            joints2D_visib=target_joints2d_visib_coco,
                            rgb_augment_config=pose_shape_cfg.TRAIN.SYNTH_DATA.AUGMENT.RGB
                        )
                        # Compute edge-images edges
                        edge_detector_output = edge_detect_model(rgb_in)
                        edge_in = edge_detector_output['thresholded_thin_edges'] \
                            if pose_shape_cfg.DATA.EDGE_NMS else edge_detector_output['thresholded_grad_magnitude']
                    else:
                        edge_in = torch.zeros((
                            pose_shape_cfg.TRAIN.BATCH_SIZE, 
                            1,
                            pose_shape_cfg.DATA.PROXY_REP_SIZE, 
                            pose_shape_cfg.DATA.PROXY_REP_SIZE),
                        device=device)

                    # Compute 2D joint heatmaps
                    j2d_heatmaps = convert_2Djoints_to_gaussian_heatmaps_torch(target_joints2d_coco_input,
                                                                               pose_shape_cfg.DATA.PROXY_REP_SIZE,
                                                                               std=pose_shape_cfg.DATA.HEATMAP_GAUSSIAN_STD)
                    j2d_heatmaps = j2d_heatmaps * target_joints2d_visib_coco[:, :, None, None]

                    # Concatenate edge-image and 2D joint heatmaps to create input proxy representation
                    #proxy_rep_input = torch.cat([edge_in, seg_maps, j2d_heatmaps], dim=1).float()  # (batch_size, C, img_wh, img_wh)
                    proxy_rep_input = torch.cat([edge_in, j2d_heatmaps], dim=1).float()  # (batch_size, C, img_wh, img_wh)

                with torch.set_grad_enabled(split == 'train'):
                    #############################################################
                    # ---------------------- FORWARD PASS -----------------------
                    #############################################################
                    pred_pose_F, pred_pose_U, pred_pose_S, pred_pose_V, pred_pose_rotmats_mode, \
                        pred_shape_dist, pred_style_dist, pred_glob, pred_cam_wp = pose_shape_model(proxy_rep_input)
                    # Pose F, U, V and rotmats_mode are (bs, 23, 3, 3) and Pose S is (bs, 23, 3)

                    pred_glob_rotmats = rot6d_to_rotmat(pred_glob)  # (bs, 3, 3)

                    pred_smpl_output_mode = smpl_model(body_pose=pred_pose_rotmats_mode,
                                                       global_orient=pred_glob_rotmats.unsqueeze(1),
                                                       betas=pred_shape_dist.loc,
                                                       pose2rot=False)

                    pred_vertices_mode = pred_smpl_output_mode.vertices
                    pred_joints_coco_mode = pred_smpl_output_mode.joints[:, ALL_JOINTS_TO_COCO_MAP]
                    pred_joints_h36m_mode = pred_smpl_output_mode.joints[:, ALL_JOINTS_TO_H36M_MAP]
                    pred_joints_h36mlsp_mode = pred_joints_h36m_mode[:, H36M_TO_J14, :]  # (bs, 14, 3)
                    
                    pred_joints2d_coco_mode = orthographic_project_torch(pred_joints_coco_mode,
                                                                         pred_cam_wp)  # (bs, 17, 2)
                    
                    with torch.no_grad():
                        pred_reposed_smpl_output_mean = smpl_model(body_pose=torch.zeros_like(target_pose)[:, 3:],
                                                                   global_orient=torch.zeros_like(target_pose)[:, :3],
                                                                   betas=pred_shape_dist.loc)
                        pred_reposed_vertices_mean = pred_reposed_smpl_output_mean.vertices  # (bs, 6890, 3)

                    pred_joints2d_coco_samples = pred_joints2d_coco_mode[:, None, :, :]  # (batch_size, 1, 17, 2)

                    #############################################################
                    # ----------------- LOSS AND BACKWARD PASS ------------------
                    #############################################################
                    pred_dict_for_loss = {'pose_params_F': pred_pose_F,
                                          'pose_params_U': pred_pose_U,
                                          'pose_params_S': pred_pose_S,
                                          'pose_params_V': pred_pose_V,
                                          'shape_params': pred_shape_dist,
                                          'style_params': pred_style_dist,
                                          'verts': pred_vertices_mode,
                                          'joints3D': pred_joints_h36mlsp_mode,
                                          'joints2D': pred_joints2d_coco_samples,
                                          'glob_rotmats': pred_glob_rotmats}

                    target_dict_for_loss = {'pose_params_rotmats': target_pose_rotmats,
                                            'shape_params': target_shape,
                                            'style_params': target_style_vector,
                                            'garment_labels': garment_labels,
                                            'verts': target_vertices,
                                            'joints3D': target_joints_h36mlsp,
                                            'joints2D': target_joints2d_coco,
                                            'joints2D_vis': target_joints2d_visib_coco,
                                            'glob_rotmats': target_glob_rotmats}

                    optimiser.zero_grad()
                    loss = criterion(target_dict_for_loss, pred_dict_for_loss)
                    if split == 'train':
                        loss.backward()
                        optimiser.step()

                #############################################################
                # --------------------- TRACK METRICS ----------------------
                #############################################################
                pred_dict_for_loss['joints2D'] = pred_joints2d_coco_mode
                if criterion.loss_config.J2D_LOSS_ON == 'samples':
                    pred_dict_for_loss['joints2Dsamples'] = pred_joints2d_coco_samples
                elif criterion.loss_config.J2D_LOSS_ON == 'means+samples':
                    pred_dict_for_loss['joints2Dsamples'] = pred_joints2d_coco_samples[:, 1:, :, :]
                del pred_dict_for_loss['pose_params_F']
                del pred_dict_for_loss['pose_params_U']
                del pred_dict_for_loss['pose_params_S']
                del pred_dict_for_loss['pose_params_V']
                del pred_dict_for_loss['shape_params']
                del pred_dict_for_loss['style_params']
                metrics_tracker.update_per_batch(split=split,
                                                 loss=loss,
                                                 pred_dict=pred_dict_for_loss,
                                                 target_dict=target_dict_for_loss,
                                                 batch_size=pose_shape_cfg.TRAIN.BATCH_SIZE,
                                                 pred_reposed_vertices=pred_reposed_vertices_mean,
                                                 target_reposed_vertices=target_reposed_vertices)
                
                #############################################################
                # ---------------- GENERATE VISUALIZATIONS ------------------
                #############################################################
                if vis_logger is not None:
                    vis_logger.vis_rgb(rgb_in)
                    vis_logger.vis_edge(edge_in)
                    vis_logger.vis_j2d_heatmaps(j2d_heatmaps)

        #############################################################
        # ------------- UPDATE METRICS HISTORY and SAVE -------------
        #############################################################
        metrics_tracker.update_per_epoch()

        save_model_weights_this_epoch = metrics_tracker.determine_save_model_weights_this_epoch(save_val_metrics,
                                                                                                best_epoch_val_metrics)

        if save_model_weights_this_epoch:
            for metric in save_val_metrics:
                best_epoch_val_metrics[metric] = metrics_tracker.epochs_history['val_' + metric][-1]
            print("Best epoch val metrics updated to ", best_epoch_val_metrics)
            best_model_wts = copy.deepcopy(pose_shape_model.state_dict())
            best_epoch = epoch
            print("Best model weights updated!")

        if epoch % pose_shape_cfg.TRAIN.EPOCHS_PER_SAVE == 0:
            save_dict = {'epoch': epoch,
                         'best_epoch': best_epoch,
                         'best_epoch_val_metrics': best_epoch_val_metrics,
                         'model_state_dict': pose_shape_model.state_dict(),
                         'best_model_state_dict': best_model_wts,
                         'optimiser_state_dict': optimiser.state_dict()}
            torch.save(save_dict,
                       os.path.join(model_save_dir, 'epoch_{}'.format(str(epoch).zfill(3)) + '.tar'))
            print('Model saved! Best Val Metrics:\n',
                  best_epoch_val_metrics,
                  '\nin epoch {}'.format(best_epoch))

    print('Training Completed. Best Val Metrics:\n',
          best_epoch_val_metrics)

    pose_shape_model.load_state_dict(best_model_wts)
    return pose_shape_model
