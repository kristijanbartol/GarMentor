import os
import torch
import torch.optim as optim
import argparse
import _thread as thread
import visdom as vis

from data.on_the_fly_smpl_train_dataset import OnTheFlySMPLTrainDataset
from renderers.non_textured_renderer import NonTexturedRenderer
from renderers.pytorch3d_textured_renderer import TexturedIUVRenderer

from models.poseMF_shapeGaussian_net import PoseMFShapeGaussianNet
from models.smpl_official import SMPL
from tailornet_for_garmentor.models.tailornet_model import get_best_runner as get_tn_runner
from models.canny_edge_detector import CannyEdgeDetector

from losses.matrix_fisher_loss import PoseMFShapeGaussianLoss

from configs.poseMF_shapeGaussian_net_config import get_cfg_defaults
from configs import paths

from train.train_network import train_poseMF_shapeGaussian_net

from utils.visualize import VisLogger

from tailornet_for_garmentor.models.smpl4garment import SMPL4Garment


def run_train(device,
              gender,
              experiment_dir,
              pose_shape_cfg_opts=None,
              resume_from_epoch=None,
              visdom=None):

    pose_shape_cfg = get_cfg_defaults()

    model_save_dir = os.path.join(experiment_dir, 'saved_models')
    logs_save_path = os.path.join(experiment_dir, 'log.pkl')
    config_save_path = os.path.join(experiment_dir, 'pose_shape_cfg.yaml')
    print('\nSaving model checkpoints to:', model_save_dir)
    print('Saving logs to:', logs_save_path)
    print('Saving config to:', config_save_path)

    if resume_from_epoch is None:
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        else:
            print('\nWARNING: {} already exists - may be overwriting previous experiments!'.format(experiment_dir))
        if pose_shape_cfg_opts is not None:
            pose_shape_cfg.merge_from_list(pose_shape_cfg_opts)
        with open(config_save_path, 'w') as f:
            f.write(pose_shape_cfg.dump())
        checkpoint = None
    else:
        assert os.path.exists(model_save_dir), 'Experiment to resume not found.'
        checkpoint_path = os.path.join(model_save_dir, 'epoch_{}'.format(str(resume_from_epoch).zfill(3)) + '.tar')
        checkpoint = torch.load(checkpoint_path, map_location=device)
        pose_shape_cfg.merge_from_file(config_save_path)
        print('\nResuming from:', checkpoint_path)

    print('\n', pose_shape_cfg)
    # ------------------------- Datasets -------------------------
    train_dataset = OnTheFlySMPLTrainDataset(poses_path=paths.TRAIN_POSES_PATH,
                                             textures_path=paths.TRAIN_TEXTURES_PATH,
                                             backgrounds_dir_path=paths.TRAIN_BACKGROUNDS_PATH,
                                             params_from='not_amass',
                                             img_wh=pose_shape_cfg.DATA.PROXY_REP_SIZE)
    val_dataset = OnTheFlySMPLTrainDataset(poses_path=paths.VAL_POSES_PATH,
                                           textures_path=paths.VAL_TEXTURES_PATH,
                                           backgrounds_dir_path=paths.VAL_BACKGROUNDS_PATH,
                                           params_from='all',
                                           img_wh=pose_shape_cfg.DATA.PROXY_REP_SIZE)
    print("\nTraining poses found:", len(train_dataset))
    print("Training textures found (grey, nongrey):", len(train_dataset.grey_textures), len(train_dataset.nongrey_textures))
    print("Training backgrounds found:", len(train_dataset.backgrounds_paths))
    print("Validation poses found:", len(val_dataset))
    print("Validation textures found (grey, nongrey):", len(val_dataset.grey_textures), len(val_dataset.nongrey_textures))
    print("Validation backgrounds found:", len(val_dataset.backgrounds_paths), '\n')

    # ------------------------- Models -------------------------
    # Edge detector
    edge_detect_model = CannyEdgeDetector(non_max_suppression=pose_shape_cfg.DATA.EDGE_NMS,
                                          gaussian_filter_std=pose_shape_cfg.DATA.EDGE_GAUSSIAN_STD,
                                          gaussian_filter_size=pose_shape_cfg.DATA.EDGE_GAUSSIAN_SIZE,
                                          threshold=pose_shape_cfg.DATA.EDGE_THRESHOLD).to(device)
    # SMPL model
    smpl_model = SMPL(paths.SMPL,
                      num_betas=pose_shape_cfg.MODEL.NUM_SMPL_BETAS,
                      gender=gender).to(device)
    smpl4garment_model = SMPL4Garment(gender=gender)
    tailornet_model = get_tn_runner(gender=gender, garment_class='t-shirt')

    # 3D shape and pose distribution predictor
    pose_shape_model = PoseMFShapeGaussianNet(smpl_parents=smpl_model.parents.tolist(),
                                              config=pose_shape_cfg).to(device)

    # Pytorch3D renderer for synthetic data generation
    pytorch3d_renderer = NonTexturedRenderer(device=device,
                                             batch_size=pose_shape_cfg.TRAIN.BATCH_SIZE,
                                             num_body_verts=27554,
                                             num_garment_verts=7702,
                                             img_wh=pose_shape_cfg.DATA.PROXY_REP_SIZE,
                                             body_faces=smpl_model.body_faces,
                                             garment_faces=smpl_model.garment_faces,
                                             bin_size=0)
    
    # Visualizer class to log the training progress.
    vis_logger = VisLogger(visdom=visdom, renderer=pytorch3d_renderer) if visdom is not None else None

    # ------------------------- Loss Function + Optimiser -------------------------
    criterion = PoseMFShapeGaussianLoss(loss_config=pose_shape_cfg.LOSS.STAGE1,
                                        img_wh=pose_shape_cfg.DATA.PROXY_REP_SIZE)
    optimiser = optim.Adam(pose_shape_model.parameters(),
                           lr=pose_shape_cfg.TRAIN.LR)

    # ------------------------- Train -------------------------
    if resume_from_epoch is not None:
        pose_shape_model.load_state_dict(checkpoint['model_state_dict'])
        optimiser.load_state_dict(checkpoint['optimiser_state_dict'])

    train_poseMF_shapeGaussian_net(pose_shape_model=pose_shape_model,
                                   pose_shape_cfg=pose_shape_cfg,
                                   smpl_model=smpl_model,
                                   edge_detect_model=edge_detect_model,
                                   device=device,
                                   train_dataset=train_dataset,
                                   val_dataset=val_dataset,
                                   criterion=criterion,
                                   optimiser=optimiser,
                                   metrics=['PVE', 'PVE-SC', 'PVE-T-SC', 'MPJPE', 'MPJPE-SC', 'MPJPE-PA', 'joints2D-L2E'],
                                   model_save_dir=model_save_dir,
                                   logs_save_path=logs_save_path,
                                   checkpoint=checkpoint,
                                   vis_logger=vis_logger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gender', '-G', type=str, choices=['male', 'female'], default='male',
                        help='Select gender (both training data and SMPL model will be of that gender).')
    parser.add_argument('--experiment_dir', '-E', type=str,
                        help='Path to directory where logs and checkpoints are saved.')
    parser.add_argument('--pose_shape_cfg_opts', '-O', nargs='*', default=None,
                        help='Command line options to modify experiment config e.g. ''-O TRAIN.NUM_EPOCHS 120'' will change number of training epochs to 120 in the config.')
    parser.add_argument('--resume_from_epoch', '-R', type=int, default=None,
                        help='Epoch to resume experiment from. If resuming, experiment_dir must already exist, with saved model checkpoints and config yaml file.')
    parser.add_argument('--vis', dest='vis', action='store_true', 
                        help='(optional) whether or not to visualize training progress details over time using Visdom')
    parser.add_argument('--vport', type=int, default=8888,
                        help='Epoch to resume experiment from. If resuming, experiment_dir must already exist, with saved model checkpoints and config yaml file.')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('\nDevice: {}'.format(device))
    
    if args.vis or args.vport != 8888:
        thread.start_new_thread(os.system, (f'visdom -p {args.vport} > /dev/null 2>&1',))
        visdom = vis.Visdom(port=args.vport)
    else:
        visdom = None

    run_train(device=device,
              gender=args.gender,
              experiment_dir=args.experiment_dir,
              pose_shape_cfg_opts=args.pose_shape_cfg_opts,
              resume_from_epoch=args.resume_from_epoch,
              visdom=visdom)
