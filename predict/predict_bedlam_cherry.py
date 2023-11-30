import os
import torch
import torchvision
import numpy as np
import argparse
import cv2
import sys

sys.path.append('/GarMentor/')

from models.poseMF_shapeGaussian_net import PoseMFShapeGaussianNet
from models.smpl_official import SMPL
from models.pose2D_hrnet import PoseHighResolutionNet
from predict.predict_hrnet import predict_hrnet
from utils.label_conversions import convert_2Djoints_to_gaussian_heatmaps_torch

from configs.poseMF_shapeGaussian_net_config import get_cfg_defaults
from configs.pose2D_hrnet_config import get_pose2D_hrnet_cfg_defaults
from configs import paths


def predict_bedlam(pose_shape_model,
                   pose_shape_cfg,
                   hrnet_model,
                   hrnet_cfg,
                   device,
                   object_detect_model=None,
                   joints2Dvisib_threshold=0.75):
    """
    Predictor for SingleInputKinematicPoseMFShapeGaussianwithGlobCam on unseen test data.
    Input --> ResNet --> image features --> FC layers --> MF over pose and Diagonal Gaussian over shape.
    Also get cam and glob separately to distribution predictor.
    Pose predictions follow the kinematic chain.
    """
    hrnet_model.eval()
    pose_shape_model.eval()
    if object_detect_model is not None:
        object_detect_model.eval()
    
    img_wh = pose_shape_cfg.DATA.PROXY_REP_SIZE

    preds_dict = {}
    crop_dir = 'output/crop/bedlam-cherry'
    for scene_name in os.listdir('output/crop/bedlam-cherry'):
        scene_dir = os.path.join(crop_dir, scene_name)
        for seq_name in os.listdir(scene_dir):
            seq_dir = os.path.join(scene_dir, seq_name)
            for img_name in os.listdir(seq_dir):
                img_path = os.path.join(seq_dir, img_name)
                mask_paths = [f'{os.path.join(img_path.replace("crop", "masks")).split(".")[0]}_{x}.png' for x in ['upper-cloth', 'lower-cloth', 'whole-body']]

                with torch.no_grad():
                    # ------------------------- INPUT LOADING AND PROXY REPRESENTATION GENERATION -------------------------
                    #image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
                    image = cv2.imread(img_path)
                    image = torch.from_numpy(image.transpose(2, 1, 0)).float().to(device) / 255.0
                    masks = np.stack([cv2.resize(cv2.imread(x), (img_wh, img_wh)) for x in mask_paths])
                    masks = torch.from_numpy(masks).float().to(device) / 255.
                    proxy_rep_segmaps = ((masks[:, :, :, 0] + masks[:, :, :, 1] + masks[:, :, :, 2]) / 3).unsqueeze(0)
                    # NOTE: The models used currently have two issues with their training data:
                    #       1. The segmaps are flipped;
                    #       2. The upper and lower garments are swapped (the order was (lower, upper, whole)).
                    proxy_rep_segmaps = torch.flip(proxy_rep_segmaps, dims=[2,])
                    proxy_rep_segmaps[:, [0, 1]] = proxy_rep_segmaps[:, [1, 0]]
                    # Predict Person Bounding Box + 2D Joints
                    hrnet_output = predict_hrnet(hrnet_model=hrnet_model,
                                                hrnet_config=hrnet_cfg,
                                                object_detect_model=object_detect_model,
                                                image=image,
                                                object_detect_threshold=pose_shape_cfg.DATA.BBOX_THRESHOLD,
                                                bbox_scale_factor=pose_shape_cfg.DATA.BBOX_SCALE_FACTOR)

                    # Create proxy representation with 1) Edge detection and 2) 2D joints heatmaps generation
                    proxy_rep_heatmaps = convert_2Djoints_to_gaussian_heatmaps_torch(joints2D=hrnet_output['joints2D'].unsqueeze(0),
                                                                                    img_wh=img_wh,
                                                                                    std=pose_shape_cfg.DATA.HEATMAP_GAUSSIAN_STD)
                    hrnet_joints2Dvisib = hrnet_output['joints2Dconfs'] > joints2Dvisib_threshold
                    hrnet_joints2Dvisib[[0, 1, 2, 3, 4, 5, 6, 11, 12]] = True  # Only removing joints [7, 8, 9, 10, 13, 14, 15, 16] if occluded
                    proxy_rep_heatmaps = proxy_rep_heatmaps * hrnet_joints2Dvisib[None, :, None, None]
                    if pose_shape_cfg.MODEL.NUM_IN_CHANNELS == 3:
                        # Verify:
                        #cv2.imwrite('rgb.png', proxy_rep_segmaps[0, 0].repeat(3, 1, 1).permute(1, 2, 0).cpu().detach().numpy() * 255)
                        proxy_rep_input = torch.cat([proxy_rep_segmaps], dim=1).float()  # (1, 3, img_wh, img_wh)
                    else:
                        proxy_rep_input = torch.cat([proxy_rep_segmaps, proxy_rep_heatmaps], dim=1).float()  # (1, 20, img_wh, img_wh)

                    # ------------------------------- POSE AND SHAPE DISTRIBUTION PREDICTION -------------------------------
                    pred_pose_F, pred_pose_U, pred_pose_S, pred_pose_V, pred_pose_rotmats_mode, \
                        pred_shape_dist, pred_style_dist, pred_glob, pred_cam_wp = pose_shape_model(proxy_rep_input)
                    # Pose F, U, V and rotmats_mode are (bsize, 23, 3, 3) and Pose S is (bsize, 23, 3)
                        
                    # TODO: Store style predictions
                    preds_dict[img_path] = pred_style_dist.loc.cpu().detach().numpy()[0]              
    np.savez('output/preds/npz/bedlam-cherry/preds.npz', **preds_dict)


def run_predict(device,
                pose_shape_weights_path,
                pose2D_hrnet_weights_path,
                pose_shape_cfg_path=None,
                already_cropped_images=False,
                joints2Dvisib_threshold=0.75,
                gender='neutral'):

    # ------------------------- Models -------------------------
    # Configs
    pose2D_hrnet_cfg = get_pose2D_hrnet_cfg_defaults()
    pose_shape_cfg = get_cfg_defaults()
    if pose_shape_cfg_path is not None:
        pose_shape_cfg.merge_from_file(pose_shape_cfg_path)
        print('\nLoaded Distribution Predictor config from', pose_shape_cfg_path)
    else:
        print('\nUsing default Distribution Predictor config.')

    # Bounding box / Object detection model
    if not already_cropped_images:
        object_detect_model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).to(device)
    else:
        object_detect_model = None

    # HRNet model for 2D joint detection
    hrnet_model = PoseHighResolutionNet(pose2D_hrnet_cfg).to(device)
    hrnet_checkpoint = torch.load(pose2D_hrnet_weights_path, map_location=device)
    hrnet_model.load_state_dict(hrnet_checkpoint, strict=False)
    print('\nLoaded HRNet weights from', pose2D_hrnet_weights_path)

    # SMPL model
    print('\nUsing {} SMPL model with {} shape parameters.'.format(gender, str(pose_shape_cfg.MODEL.NUM_SMPL_BETAS)))
    smpl_model = SMPL(paths.SMPL_DIR,
                      batch_size=1,
                      gender=gender,
                      num_betas=pose_shape_cfg.MODEL.NUM_SMPL_BETAS).to(device)
    smpl_immediate_parents = smpl_model.parents.tolist()

    # 3D shape and pose distribution predictor
    pose_shape_dist_model = PoseMFShapeGaussianNet(smpl_parents=smpl_immediate_parents,
                                                   config=pose_shape_cfg).to(device)
    checkpoint = torch.load(pose_shape_weights_path, map_location=device)
    pose_shape_dist_model.load_state_dict(checkpoint['best_model_state_dict'])
    print('\nLoaded Distribution Predictor weights from', pose_shape_weights_path)

    # ------------------------- Predict -------------------------
    torch.manual_seed(0)
    np.random.seed(0)
    predict_bedlam(pose_shape_model=pose_shape_dist_model,
                   pose_shape_cfg=pose_shape_cfg,
                   hrnet_model=hrnet_model,
                   hrnet_cfg=pose2D_hrnet_cfg,
                   device=device,
                   object_detect_model=object_detect_model,
                   joints2Dvisib_threshold=joints2Dvisib_threshold)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pose_shape_weights', '-W3D', type=str, default='experiments/drapenet-50k-significant_augment/saved_models/epoch_023.tar')
    parser.add_argument('--pose_shape_cfg', type=str, default=None)
    parser.add_argument('--pose2D_hrnet_weights', '-W2D', type=str, default='/data/hierprob3d/pose_hrnet_w48_384x288.pth')
    parser.add_argument('--cropped_images', '-C', action='store_true', help='Images already cropped and centred.')
    parser.add_argument('--joints2Dvisib_threshold', '-T', type=float, default=0.75)
    parser.add_argument('--gender', '-G', type=str, default='neutral', choices=['neutral', 'male', 'female'], help='Gendered SMPL models may be used.')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('\nDevice: {}'.format(device))

    run_predict(device=device,
                pose_shape_weights_path=args.pose_shape_weights,
                pose_shape_cfg_path=args.pose_shape_cfg,
                pose2D_hrnet_weights_path=args.pose2D_hrnet_weights,
                already_cropped_images=args.cropped_images,
                joints2Dvisib_threshold=args.joints2Dvisib_threshold,
                gender=args.gender)
