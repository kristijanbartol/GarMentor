import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from smplx.lbs import batch_rodrigues
from pytorch3d.transforms import matrix_to_axis_angle
import time
from trimesh import Trimesh
import sys

sys.path.append('/garmentor/')

from metrics.eval_metrics_tracker import EvalMetricsTracker
from rendering.body import BodyRenderer
from utils.cam_utils import orthographic_project_torch
from utils.mesh_utils import concatenate_mesh_list
from utils.rigid_transform_utils import rot6d_to_rotmat, aa_rotate_translate_points_pytorch3d, aa_rotate_rotmats
from utils.joints2d_utils import undo_keypoint_normalisation
from utils.label_conversions import (
    convert_2Djoints_to_gaussian_heatmaps_torch,
    convert_multiclass_to_binary_labels, 
    ALL_JOINTS_TO_COCO_MAP, 
    ALL_JOINTS_TO_H36M_MAP, 
    H36M_TO_J14
)
from utils.sampling_utils import pose_matrix_fisher_sampling_torch
from utils.eval_utils import pa_mpjpe
from utils.measurements.mesh_measurements import get_measurements


GENDER_MAP = {
    'm': 'male',
    'f': 'female'
}


def evaluate_pose_MF_shapeGaussian_net(pose_shape_model,
                                       pose_shape_cfg,
                                       smpl_model_male,
                                       smpl_model_female,
                                       parametric_model,
                                       edge_detect_model,
                                       renderer,
                                       texture,
                                       lights_rgb_settings,
                                       fixed_cam_t,
                                       fixed_orthographic_scale,
                                       device,
                                       eval_dataset,
                                       metrics,
                                       exec_time_components,
                                       save_path,
                                       num_workers=4,
                                       pin_memory=True,
                                       save_per_frame_metrics=True,
                                       num_samples_for_metrics=10,
                                       sample_on_cpu=False,
                                       vis_logger=None):
    
    smpl_faces = np.load('/data/hierprob3d/smpl/smpl_faces.npy').astype(np.int32)
    measurements_errors = []

    eval_dataloader = DataLoader(eval_dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 drop_last=False,
                                 num_workers=num_workers,
                                 pin_memory=pin_memory)

    # Instantiate metrics tracker
    metrics_tracker = EvalMetricsTracker(metrics,
                                         exec_time_components,
                                         save_path=save_path,
                                         save_per_frame_metrics=save_per_frame_metrics)
    metrics_tracker.initialise_metric_sums()
    metrics_tracker.initialise_per_frame_metric_lists()

    if any('silhouette' in metric for metric in metrics):
        silhouette_renderer = BodyRenderer(device=device,
                                                  batch_size=1,
                                                  img_wh=pose_shape_cfg.DATA.PROXY_REP_SIZE,
                                                  projection_type='orthographic',
                                                  render_rgb=False,
                                                  bin_size=32)

    if save_per_frame_metrics:
        fname_per_frame = []
        pose_per_frame = []
        shape_per_frame = []
        cam_per_frame = []

    pose_shape_model.eval()
    sample_freq = 25
    sample_count = 0
    for batch_num, samples_batch in enumerate(tqdm(eval_dataloader)):
        if batch_num % sample_freq != 0:
            continue
        sample_count += 1
        
        exec_times_dict = dict(zip(exec_time_components, [0.] * len(exec_time_components)))
        with torch.no_grad():
            # ------------------ INPUTS ------------------
            image = samples_batch['image'].to(device)
            heatmaps = samples_batch['heatmaps'].to(device)
            
            start_time = time.time()
            edge_detector_output = edge_detect_model(image)
            exec_times_dict['edge-time'] = time.time() - start_time
            
            proxy_rep_img = edge_detector_output['thresholded_thin_edges'] if pose_shape_cfg.DATA.EDGE_NMS else edge_detector_output['thresholded_grad_magnitude']
            proxy_rep_input = torch.cat([proxy_rep_img, heatmaps], dim=1)

            # ------------------ Targets ------------------
            target_pose = samples_batch['pose'].to(device)
            target_shape = samples_batch['shape'].to(device)
            target_shape_clothed = samples_batch['shape_clothed'].to(device)
            target_gender = samples_batch['gender'][0]
            fname = samples_batch['fname']
            if any('joints2D' in metric for metric in metrics):
                target_joints2d_coco = samples_batch['keypoints']
            if any('silhouette' in metric for metric in metrics):
                target_silhouette = samples_batch['silhouette']

            # Flipping pose targets such that they are right way up in 3D space - i.e. wrong way up when projected
            target_pose_rotmats = batch_rodrigues(target_pose.contiguous().view(-1, 3)).view(-1, 24, 3, 3)
            target_glob_rotmats = target_pose_rotmats[:, 0, :, :]
            target_glob_vecs, _ = aa_rotate_rotmats(rotmats=target_glob_rotmats,
                                                    axis=[1, 0, 0],
                                                    #angle=np.pi,
                                                    angle=2*np.pi,
                                                    rot_mult_order='pre')
            target_pose[:, :3] = target_glob_vecs

            if target_gender == 'm':
                target_smpl_output_unclothed = smpl_model_male(body_pose=target_pose[:, 3:],
                                                     global_orient=target_pose[:, :3],
                                                     betas=target_shape)
                target_smpl_output_clothed = smpl_model_male(body_pose=target_pose[:, 3:],
                                                     global_orient=target_pose[:, :3],
                                                     betas=target_shape_clothed)

                #target_reposed_smpl_output_unclothed = smpl_model_male(betas=torch.zeros_like(target_shape).to(device))
                target_reposed_smpl_output_unclothed = smpl_model_male(betas=target_shape.to(device))
                target_reposed_smpl_output_clothed = smpl_model_male(betas=target_shape_clothed)
            elif target_gender == 'f':
                continue
                target_smpl_output = smpl_model_female(body_pose=target_pose[:, 3:],
                                                       global_orient=target_pose[:, :3],
                                                       betas=target_shape)
                target_reposed_smpl_output = smpl_model_female(betas=target_shape)
                
                target_smpl_output_clothed = smpl_model_female(body_pose=target_pose[:, 3:],
                                                     global_orient=target_pose[:, :3],
                                                     betas=target_shape_clothed)
                target_reposed_smpl_output_clothed = smpl_model_female(betas=target_shape_clothed)

            target_reposed_vertices_unclothed = target_reposed_smpl_output_unclothed.vertices
            target_reposed_vertices_clothed = target_reposed_smpl_output_clothed.vertices
            
            target_vertices_unclothed = target_smpl_output_unclothed.vertices
            target_vertices_clothed = target_smpl_output_clothed.vertices
            
            target_joints_h36mlsp = target_smpl_output_unclothed.joints[:, ALL_JOINTS_TO_H36M_MAP, :][:, H36M_TO_J14, :]

            # ------------------------------- PREDICTIONS -------------------------------
            start_time = time.time()
            pred_pose_F, pred_pose_U, pred_pose_S, pred_pose_V, pred_pose_rotmats_mode, \
                pred_shape_dist, pred_style_dist, pred_glob, pred_cam_wp = pose_shape_model(proxy_rep_input)
            exec_times_dict['inference-time'] = time.time() - start_time
            # Pose F, U, V and rotmats_mode are (bsize, 23, 3, 3) and Pose S is (bsize, 23, 3)

            orthographic_scale = pred_cam_wp[:, [0, 0]]
            cam_t = torch.cat([pred_cam_wp[:, 1:],
                               torch.ones(pred_cam_wp.shape[0], 1, device=device).float() * 2.5],
                              dim=-1)

            if pred_glob.shape[-1] == 3:
                pred_glob_rotmats = batch_rodrigues(pred_glob)  # (1, 3, 3)
            elif pred_glob.shape[-1] == 6:
                pred_glob_rotmats = rot6d_to_rotmat(pred_glob)  # (1, 3, 3)
            
            pred_pose_axis_angle = matrix_to_axis_angle(pred_pose_rotmats_mode)
            pred_glob_axis_angle = matrix_to_axis_angle(pred_glob_rotmats).unsqueeze(0)
            pred_pose_axis_angle = torch.cat((pred_glob_axis_angle, pred_pose_axis_angle), dim=1).reshape(1, 72)
            
            pred_smpl_output_original = smpl_model_male(body_pose=pred_pose_rotmats_mode,
                                          global_orient=pred_glob_rotmats.unsqueeze(1),
                                          #betas=pred_shape_dist.loc,
                                          betas=torch.zeros(target_shape.shape).to(device),
                                          pose2rot=False)

            smpl_output_dict = parametric_model.run(
                #pose=pred_pose_axis_angle[0].cpu().numpy(),
                pose=target_pose[0].cpu().numpy(),
                shape=np.zeros(target_shape.shape[1:]),
                #shape=pred_shape_dist.loc[0].cpu().numpy(),
                style_vector=np.zeros((4, 4))
                #style_vector=pred_style_dist.loc[0].cpu().numpy()
            )
            
            exec_times_dict['tailornet-time'] = parametric_model.exec_times['tailornet-time']
            exec_times_dict['smpl-time'] = parametric_model.exec_times['smpl-time']
            exec_times_dict['interpenetrations-time'] = parametric_model.exec_times['interpenetrations-time']
            
            smpl_reposed_output_dict = parametric_model.run(
                #pose=pred_pose_axis_angle[0].cpu().numpy(),
                pose=np.zeros(pred_pose_axis_angle.shape[1:]),
                shape=np.zeros(target_shape.shape[1:]),
                #shape=pred_shape_dist.loc[0].cpu().numpy(),
                #style_vector=pred_style_dist.loc[0].cpu().numpy()
                style_vector=np.zeros((4, 4))
            )
            
            pred_vertices_body = smpl_output_dict['upper'].body_verts
            pred_reposed_vertices_body = smpl_reposed_output_dict['upper'].body_verts
            pred_vertices_merged = np.concatenate(
                (smpl_output_dict['upper'].body_verts,
                 smpl_output_dict['upper'].garment_verts,
                 smpl_output_dict['lower'].garment_verts
            ), axis=0)
            pred_reposed_vertices_merged = np.concatenate(
                (smpl_reposed_output_dict['upper'].body_verts,
                 smpl_reposed_output_dict['upper'].garment_verts,
                 smpl_reposed_output_dict['lower'].garment_verts
            ), axis=0)
            pred_joints_all_mode = pred_smpl_output_original.joints
            pred_joints_h36mlsp_mode = pred_joints_all_mode[:, ALL_JOINTS_TO_H36M_MAP, :][:, H36M_TO_J14, :]  # (1, 14, 3)
            pred_joints_coco_mode = pred_joints_all_mode[:, ALL_JOINTS_TO_COCO_MAP, :]
            pred_vertices_original = pred_smpl_output_original.vertices

            pred_reposed_smpl_output_mean = smpl_model_male(betas=pred_shape_dist.loc)
            pred_reposed_vertices_original = pred_reposed_smpl_output_mean.vertices  # (1, 6890, 3)

            # Pose targets were flipped such that they are right way up in 3D space - i.e. wrong way up when projected
            # Need to flip pred vertices and pred joints before projecting to 2D for 2D metrics
            if any('joints2D' in metric for metric in metrics):
                pred_joints_coco_mode = aa_rotate_translate_points_pytorch3d(points=pred_joints_coco_mode,
                                                                             axes=torch.tensor([1., 0., 0.], device=device).float(),
                                                                             angles=2*np.pi,
                                                                             translations=torch.zeros(3, device=device).float())
                pred_joints2d_coco_mode = orthographic_project_torch(pred_joints_coco_mode, pred_cam_wp)  # (1, 17, 2)
                pred_joints2d_coco_mode = undo_keypoint_normalisation(pred_joints2d_coco_mode, pose_shape_cfg.DATA.PROXY_REP_SIZE)
            if any('silhouette' in metric for metric in metrics):
                pred_vertices_flipped_mode = aa_rotate_translate_points_pytorch3d(points=pred_vertices_mode_zeroshape_zerostyle,
                                                                                  axes=torch.tensor([1., 0., 0.], device=device),
                                                                                  angles=np.pi,
                                                                                  translations=torch.zeros(3, device=device))

            if 'silhouette-IOU' in metrics:
                wp_render_output = silhouette_renderer(vertices=pred_vertices_flipped_mode,
                                                       cam_t=cam_t,
                                                       orthographic_scale=orthographic_scale)
                iuv_mode = wp_render_output['iuv_images']
                part_seg_mode = iuv_mode[:, :, :, 0].round()
                pred_silhouette_mode = convert_multiclass_to_binary_labels(part_seg_mode)

            if any('samples' in metric for metric in metrics):
                assert pred_pose_F.shape[0] == 1, "Batch size must be 1 for min samples metrics!"
                pred_pose_rotmats_samples = pose_matrix_fisher_sampling_torch(pose_U=pred_pose_U,
                                                                              pose_S=pred_pose_S,
                                                                              pose_V=pred_pose_V,
                                                                              num_samples=num_samples_for_metrics,
                                                                              b=1.5,
                                                                              oversampling_ratio=8,
                                                                              sample_on_cpu=sample_on_cpu)  # (1, num samples, 23, 3, 3)
                pred_shape_samples = pred_shape_dist.rsample([num_samples_for_metrics]).transpose(0, 1)  # (1, num_samples, num_smpl_betas)
                pred_smpl_output_samples = smpl_model(body_pose=pred_pose_rotmats_samples[0, :, :, :, :],
                                                      global_orient=pred_glob_rotmats.unsqueeze(1).expand(num_samples_for_metrics, -1, -1, -1),
                                                      betas=pred_shape_samples[0, :, :],
                                                      pose2rot=False)
                pred_vertices_samples = pred_smpl_output_samples.vertices
                pred_vertices_samples[0] = pred_vertices_mode[0]  # (num samples, 6890, 3) - Including mode as one of the samples for 3D samples min metrics
                pred_joints_h36mlsp_samples = pred_smpl_output_samples.joints[:, ALL_JOINTS_TO_H36M_MAP, :][:, H36M_TO_J14, :]
                pred_joints_h36mlsp_samples[0] = pred_joints_h36mlsp_mode[0]  # (num samples, 14, 3) - Including mode as one of the samples for 3D samples min metrics

                pred_reposed_vertices_samples = smpl_model(body_pose=torch.zeros(num_samples_for_metrics, 69, device=device).float(),
                                                           global_orient=torch.zeros(num_samples_for_metrics, 3, device=device).float(),
                                                           betas=pred_shape_samples[0, :, :]).vertices
                pred_reposed_vertices_samples[0] = pred_reposed_vertices_mean[0]   # (num samples, 6890, 3) - Including mode as one of the samples for 3D samples min metrics

                if 'joints2Dsamples-L2E' in metrics:
                    pred_joints_coco_samples = pred_smpl_output_samples.joints[:, ALL_JOINTS_TO_COCO_MAP, :]  # (num samples, 17, 3)
                    # Pose targets were flipped such that they are right way up in 3D space - i.e. wrong way up when projected
                    # Need to flip pred_joints_coco 180° about x-axis so they are right way up when projected
                    pred_joints_coco_samples = aa_rotate_translate_points_pytorch3d(points=pred_joints_coco_samples,
                                                                                    axes=torch.tensor([1., 0., 0.], device=device).float(),
                                                                                    angles=np.pi,
                                                                                    translations=torch.zeros(3, device=device).float())
                    pred_joints2d_coco_samples = orthographic_project_torch(pred_joints_coco_samples, pred_cam_wp)  # (num samples, 17, 2)
                    pred_joints2d_coco_samples = undo_keypoint_normalisation(pred_joints2d_coco_samples, pose_shape_cfg.DATA.PROXY_REP_SIZE)

                if 'silhouettesamples-IOU' in metrics:
                    pred_silhouette_samples = []
                    for i in range(num_samples_for_metrics):
                        pred_vertices_flipped_sample = aa_rotate_translate_points_pytorch3d(points=pred_smpl_output_samples.vertices[[i]],
                                                                                            axes=torch.tensor([1., 0., 0.], device=device),
                                                                                            angles=np.pi,
                                                                                            translations=torch.zeros(3, device=device))
                        iuv_sample = silhouette_renderer(vertices=pred_vertices_flipped_sample,
                                                         cam_t=cam_t,
                                                         orthographic_scale=orthographic_scale)['iuv_images']
                        part_seg_sample = iuv_sample[:, :, :, 0].round()
                        pred_silhouette_samples.append(convert_multiclass_to_binary_labels(part_seg_sample))
                    pred_silhouette_samples = torch.stack(pred_silhouette_samples, dim=1)  # (1, num samples, img wh, img wh)

            # ------------------------------- TRACKING METRICS -------------------------------
            pred_dict = {'verts': np.expand_dims(pred_vertices_body[:6890], 0),
                         #'verts': pred_vertices_original.cpu().detach().numpy(),
                         'verts_clothed': np.expand_dims(pred_vertices_merged, 0),
                         #'verts_clothed': np.expand_dims(pred_vertices_body[:6890], 0),
                         #'verts_clothed': pred_vertices_original.cpu().detach().numpy(),
                         'reposed_verts': np.expand_dims(pred_reposed_vertices_body[:6890], 0),
                         'reposed_verts_clothed': np.expand_dims(pred_reposed_vertices_merged, 0),
                         'joints3D': pred_joints_h36mlsp_mode.cpu().detach().numpy()}
            target_dict = {#'verts': np.expand_dims(target_vertices_body, 0),
                           'verts': target_vertices_unclothed.cpu().detach().numpy(),
                           #'verts_clothed': np.expand_dims(target_vertices_clothed, 0),
                           'verts_clothed': target_vertices_clothed.cpu().detach().numpy(),
                           'reposed_verts': target_reposed_vertices_unclothed.cpu().detach().numpy(),
                           'reposed_verts_clothed': target_reposed_vertices_clothed.cpu().detach().numpy(),
                           'joints3D': target_joints_h36mlsp.cpu().detach().numpy()}

            if 'joints2D-L2E' in metrics:
                pred_dict['joints2D'] = pred_joints2d_coco_mode.cpu().detach().numpy()
                target_dict['joints2D'] = target_joints2d_coco.numpy()
            if 'silhouette-IOU' in metrics:
                pred_dict['silhouettes'] = pred_silhouette_mode.cpu().detach().numpy()
                target_dict['silhouettes'] = target_silhouette.numpy()

            if any('samples_min' in metric for metric in metrics):
                pred_dict['verts_samples'] = pred_vertices_samples.cpu().detach().numpy()
                pred_dict['reposed_verts_samples'] = pred_reposed_vertices_samples.cpu().detach().numpy()
                pred_dict['joints3D_samples'] = pred_joints_h36mlsp_samples.cpu().detach().numpy()
            if 'joints2Dsamples-L2E' in metrics:
                pred_dict['joints2Dsamples'] = pred_joints2d_coco_samples[None, :, :, :].cpu().detach().numpy()
            if 'silhouettesamples-IOU' in metrics:
                pred_dict['silhouettessamples'] = pred_silhouette_samples.cpu().detach().numpy()

            metrics_tracker.update_per_batch(pred_dict,
                                             target_dict,
                                             exec_times_dict,
                                             1,
                                             return_transformed_points=False,
                                             return_per_frame_metrics=False)
            
            target_body_measurements = np.array(get_measurements(
                target_reposed_smpl_output_unclothed.vertices[0].cpu().detach().numpy(), smpl_faces))
            pred_body_measurements = np.array(get_measurements(
                pred_reposed_smpl_output_mean.vertices[0].cpu().detach().numpy(), smpl_faces))
            
            measurements_error = np.abs(target_body_measurements - pred_body_measurements)
            measurements_errors.append(measurements_error)
            
            print(f"Chamfer-T: {np.array(metrics_tracker.per_frame_metrics['Chamfer-T']).mean()}")
            print(f"Chamfer: {np.array(metrics_tracker.per_frame_metrics['Chamfer']).mean()}")
            print(f"MPJPE-PA: {np.array(metrics_tracker.per_frame_metrics['MPJPE-PA']).mean() * 1000.}")
            print(f"MPJPE-SC: {np.array(metrics_tracker.per_frame_metrics['MPJPE-SC']).mean() * 1000.}")
            print(f"PVE-PA: {np.array(metrics_tracker.per_frame_metrics['PVE-PA']).mean() * 1000.}")
            print(f"PVE-SC: {np.array(metrics_tracker.per_frame_metrics['PVE-SC']).mean() * 1000.}")
            print(f"PVE-T-SC: {np.array(metrics_tracker.per_frame_metrics['PVE-T-SC']).mean() * 1000.}")
            # NOTE: Not relevant for us in this paper as others either don't separate body from clothes or
            #       they anyways the state-of-the-art pose&shape estimation model and we can't compete then.
            print(f'Measurements error: {np.mean(np.array(measurements_errors), axis=0) * 1000.}')
            
            pred_vertices_merged_trimesh, pred_faces_merged_trimesh = concatenate_mesh_list(
                vertices_list = [
                    smpl_output_dict['upper'].body_verts,
                    smpl_output_dict['upper'].garment_verts,
                    smpl_output_dict['lower'].garment_verts
                ],
                faces_list = [
                    smpl_output_dict['upper'].body_faces,
                    smpl_output_dict['upper'].garment_verts,
                    smpl_output_dict['lower'].garment_verts
                ]
            )
            pred_reposed_vertices_merged_trimesh, pred_reposed_faces_merged_trimesh = concatenate_mesh_list(
                vertices_list=[
                    smpl_reposed_output_dict['upper'].body_verts,
                    smpl_reposed_output_dict['upper'].garment_verts,
                    smpl_reposed_output_dict['lower'].garment_verts
                ],
                faces_list=[
                    smpl_reposed_output_dict['upper'].body_faces,
                    smpl_reposed_output_dict['upper'].garment_faces,
                    smpl_reposed_output_dict['lower'].garment_faces
                ]
            )
            '''
            Trimesh(
                vertices=smpl_output_dict['upper'].body_verts, 
                faces=smpl_output_dict['upper'].body_faces
            ).export(f'output/pred/{batch_num:05d}_body.obj')
            Trimesh(
                vertices=smpl_output_dict['upper'].garment_verts, 
                faces=smpl_output_dict['upper'].garment_faces
            ).export(f'output/pred/{batch_num:05d}_upper.obj')
            Trimesh(
                vertices=smpl_output_dict['lower'].garment_verts, 
                faces=smpl_output_dict['lower'].garment_faces
            ).export(f'output/pred/{batch_num:05d}_lower.obj')
            Trimesh(vertices=pred_vertices_merged_trimesh, faces=pred_faces_merged_trimesh).export(
                f'output/pred/{batch_num:05d}.obj')
            Trimesh(
                vertices=smpl_reposed_output_dict['upper'].body_verts, 
                faces=smpl_reposed_output_dict['upper'].body_faces
            ).export(f'output/pred_reposed/{batch_num:05d}_body.obj')
            Trimesh(
                vertices=smpl_reposed_output_dict['upper'].garment_verts, 
                faces=smpl_reposed_output_dict['upper'].garment_faces
            ).export(f'output/pred_reposed/{batch_num:05d}_upper.obj')
            Trimesh(
                vertices=smpl_reposed_output_dict['lower'].garment_verts, 
                faces=smpl_reposed_output_dict['lower'].garment_faces
            ).export(f'output/pred_reposed/{batch_num:05d}_lower.obj')
            Trimesh(vertices=pred_reposed_vertices_merged_trimesh, faces=pred_reposed_faces_merged_trimesh).export(
                f'output/pred_reposed/{batch_num:05d}.obj')
            '''
            #Trimesh(vertices=pred_reposed_vertices_merged_trimesh * 1000., faces=pred_reposed_faces_merged_trimesh).export(
            #    f'output/pred_reposed/{batch_num:05d}.obj')
            #Trimesh(vertices=target_vertices_clothed[0].cpu().numpy(), faces=smpl_faces).export(
            #    f'output/target/{batch_num:05d}.obj')
            #Trimesh(vertices=target_reposed_vertices_clothed[0].cpu().numpy(), faces=smpl_faces).export(
            #    f'output/target_reposed/{batch_num:05d}.obj'
            #)
            
            if vis_logger is not None:
                vis_logger.vis_rgb(image, label='input_image')
                vis_logger.vis_edge(proxy_rep_img)
                #vis_logger.vis_keypoints(samples_batch['keypoints'], 'detector_keypoints')
                #vis_logger.vis_keypoints(pred_joints2d_coco_mode, 'pred_joints')
                
                a, R, t = pa_mpjpe(pred_vertices_original.cpu().numpy(), target_vertices_unclothed.cpu().numpy())
                pred_vertices_original = torch.from_numpy(a * np.matmul(pred_vertices_original.cpu().numpy(), R) + t).to(device)
                
                target_renderer_output = renderer(vertices=target_reposed_vertices_unclothed,
                                                  textures=texture,
                                                  cam_t=fixed_cam_t,
                                                  lights_rgb_settings=lights_rgb_settings)
                pred_renderer_output = renderer(vertices=torch.from_numpy(pred_reposed_vertices_body[:6890]).unsqueeze(0).float().to(device),
                                                  textures=texture,
                                                  cam_t=fixed_cam_t,
                                                  orthographic_scale=fixed_orthographic_scale,
                                                  lights_rgb_settings=lights_rgb_settings)
                
                vis_logger.vis_rgb(target_renderer_output['rgb_images'].permute(0, 3, 1, 2), label='target_render')
                vis_logger.vis_rgb(pred_renderer_output['rgb_images'].permute(0, 3, 1, 2), label='pred_render')

            if save_per_frame_metrics:
                fname_per_frame.append(fname)
                pose_per_frame.append(np.concatenate([pred_glob_rotmats[:, None, :, :].cpu().detach().numpy(),
                                                      pred_pose_rotmats_mode.cpu().detach().numpy()],
                                                     axis=1))
                shape_per_frame.append(pred_shape_dist.loc.cpu().detach().numpy())
                cam_per_frame.append(pred_cam_wp.cpu().detach().numpy())

    # ------------------------------- DISPLAY METRICS AND SAVE PER-FRAME METRICS -------------------------------
    metrics_tracker.compute_final_metrics()

    if save_per_frame_metrics:
        fname_per_frame = np.concatenate(fname_per_frame, axis=0)
        np.save(os.path.join(save_path, 'fname_per_frame.npy'), fname_per_frame)

        pose_per_frame = np.concatenate(pose_per_frame, axis=0)
        np.save(os.path.join(save_path, 'pose_per_frame.npy'), pose_per_frame)

        shape_per_frame = np.concatenate(shape_per_frame, axis=0)
        np.save(os.path.join(save_path, 'shape_per_frame.npy'), shape_per_frame)

        cam_per_frame = np.concatenate(cam_per_frame, axis=0)
        np.save(os.path.join(save_path, 'cam_per_frame.npy'), cam_per_frame)
