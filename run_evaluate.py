import os
import numpy as np
import torch
import argparse
import _thread as thread
import visdom as vis

from configs import paths
from configs.poseMF_shapeGaussian_net_config import get_cfg_defaults

from data.datasets.pw3d_eval_dataset import PW3DEvalDataset
from data.datasets.ssp3d_eval_dataset import SSP3DEvalDataset
from models.fully_parametric_net import FullyParametricNet

from models.smpl_official import SMPL
from models.poseMF_shapeGaussian_net import PoseMFShapeGaussianNet
from models.canny_edge_detector import CannyEdgeDetector
from models.parametric_model import ParametricModel
#from evaluate.evaluate_poseMF_shapeGaussian_net import evaluate_pose_MF_shapeGaussian_net
from evaluate.original_evaluate import evaluate_pose_MF_shapeGaussian_net
from rendering.body import BodyRenderer
from utils.garment_classes import GarmentClasses
from vis.logger import VisLogger


def run_evaluate(device,
                 pose_shape_weights_path,
                 pose_shape_cfg_path=None,
                 num_samples_for_metrics=10,
                 gender='male',
                 upper_class='t-shirt',
                 lower_class='pant',
                 visdom=None):

    # ------------------ Models ------------------
    # Config
    pose_shape_cfg = get_cfg_defaults()
    if pose_shape_cfg_path is not None:
        pose_shape_cfg.merge_from_file(pose_shape_cfg_path)
        print('\nLoaded Distribution Predictor config from', pose_shape_cfg_path)
    else:
        print('\nUsing default Distribution Predictor config.')

    # Edge detector
    edge_detect_model = CannyEdgeDetector(non_max_suppression=pose_shape_cfg.DATA.EDGE_NMS,
                                          gaussian_filter_std=pose_shape_cfg.DATA.EDGE_GAUSSIAN_STD,
                                          gaussian_filter_size=pose_shape_cfg.DATA.EDGE_GAUSSIAN_SIZE,
                                          threshold=pose_shape_cfg.DATA.EDGE_THRESHOLD).to(device)

    # SMPL neutral/male/female models
    smpl_model = SMPL(paths.SMPL,
                      batch_size=1,
                      num_betas=pose_shape_cfg.MODEL.NUM_SMPL_BETAS).to(device)
    smpl_immediate_parents = smpl_model.parents.tolist()
    smpl_model_male = SMPL(paths.SMPL,
                           batch_size=1,
                           gender='male').to(device)
    smpl_model_female = SMPL(paths.SMPL,
                             batch_size=1,
                             gender='female').to(device)
    
    upper_class = 't-shirt'
    lower_class = 'pant'
    
    parametric_model = ParametricModel(gender='male', 
                                       garment_classes=GarmentClasses(
                                           upper_class=upper_class,
                                           lower_class=lower_class
                                       ),
                                       eval=True)

    # 3D shape and pose distribution predictor 
    pose_shape_dist_model = PoseMFShapeGaussianNet(smpl_parents=smpl_immediate_parents,
                                                config=pose_shape_cfg).to(device)
    
    checkpoint = torch.load(pose_shape_weights_path, map_location=device)
    pose_shape_dist_model.load_state_dict(checkpoint['best_model_state_dict'])
    print('\nLoaded Distribution Predictor weights from', pose_shape_weights_path)

    # ------------------ Dataset + Metrics ------------------
    metrics = ['PVE', 'PVE-SC', 'PVE-PA', 'PVE-T-SC', 'MPJPE', 'MPJPE-SC', 'MPJPE-PA', 'Chamfer', 'Chamfer-T', 'joints2D-L2E']
    exec_time_components = ['edge-time', 'inference-time', 'tailornet-time', 'smpl-time', 'interpenetrations-time']

    save_path = './3dpw_eval'
    eval_dataset = PW3DEvalDataset(pw3d_dir_path=paths.PW3D_PATH,
                                   config=pose_shape_cfg,
                                   visible_joints_threshold=0.6)

    print("\nEvaluating with {} eval examples.".format(str(len(eval_dataset))))
    print("Metrics:", metrics)
    print("Saving to:", save_path)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    if visdom is not None:
        # Visualizer class to log the evaluation samples.
        vis_logger = VisLogger(visdom=visdom) if visdom is not None else None
        # Pytorch3D renderer for vertices' visualization
        renderer = BodyRenderer(device=device,
                                        batch_size=1,
                                        img_wh=512,
                                        #projection_type='orthographic',
                                        projection_type='perspective',
                                        render_rgb=True,
                                        bin_size=32)
        plain_texture = torch.ones(1, 1200, 800, 3, device=device).float() * 0.7
        lights_rgb_settings = {'location': torch.tensor([[0., -0.8, -2.0]], device=device, dtype=torch.float32),
                            'ambient_color': 0.5 * torch.ones(1, 3, device=device, dtype=torch.float32),
                            'diffuse_color': 0.3 * torch.ones(1, 3, device=device, dtype=torch.float32),
                            'specular_color': torch.zeros(1, 3, device=device, dtype=torch.float32)}
        fixed_cam_t = torch.tensor([[0., -0.2, 2.5]], device=device)
        fixed_orthographic_scale = torch.tensor([[0.95, 0.95]], device=device)
    else:
        vis_logger, renderer, plain_texture, lights_rgb_settings, fixed_cam_t, fixed_orthographic_scale = [None] * 6

    # ------------------ Evaluate ------------------
    torch.manual_seed(0)
    np.random.seed(0)
    evaluate_pose_MF_shapeGaussian_net(pose_shape_model=pose_shape_dist_model,
                                       pose_shape_cfg=pose_shape_cfg,
                                       smpl_model_male=smpl_model_male,
                                       smpl_model_female=smpl_model_female,
                                       parametric_model=parametric_model,
                                       edge_detect_model=edge_detect_model,
                                       renderer=renderer,
                                       texture=plain_texture,
                                       lights_rgb_settings=lights_rgb_settings,
                                       fixed_cam_t=fixed_cam_t,
                                       fixed_orthographic_scale=fixed_orthographic_scale,
                                       device=device,
                                       eval_dataset=eval_dataset,
                                       metrics=metrics,
                                       exec_time_components=exec_time_components,
                                       save_path=save_path,
                                       num_samples_for_metrics=num_samples_for_metrics,
                                       sample_on_cpu=True,
                                       vis_logger=vis_logger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pose_shape_weights', '-W3D', type=str, default='./model_files/poseMF_shapeGaussian_net_weights.tar')
    parser.add_argument('--pose_shape_cfg', type=str, default=None)
    parser.add_argument('--num_samples', '-N', type=int, default=10, 
                        help='Number of samples to use for sample-based evaluation metrics.')
    parser.add_argument('--gender', '-G', type=str, choices=['male', 'female'],
                        help='Gender string.')
    parser.add_argument('--upper_class', '-U', type=str, choices=['t-shirt', 'shirt'],
                        help='Upper class string.')
    parser.add_argument('--lower_class', '-L', type=str, choices=['pant', 'short-pant'],
                        help='Lower class string.')
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

    run_evaluate(device=device,
                 pose_shape_weights_path=args.pose_shape_weights,
                 pose_shape_cfg_path=args.pose_shape_cfg,
                 num_samples_for_metrics=args.num_samples,
                 gender=args.gender,
                 upper_class=args.upper_class,
                 lower_class=args.lower_class,
                 visdom=visdom)



