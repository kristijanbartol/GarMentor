import os
import sys
import argparse
import visdom as vis
from pathlib import Path

sys.path.append('/garmentor/')

from models.smpl_official import easy_create_smpl_model
from vis.visualizers.keypoints import KeypointsVisualizer
from vis.visualizers.body import BodyVisualizer
from vis.visualizers.clothed import ClothedVisualizer
from vis.visualizers.clothed3d import ClothedVisualizer3D
from vis.logger import VisLogger
from data.datasets.prepare.surreal.tn import DataPreGenerator


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    data_generator = DataPreGenerator()
    pose, shape, style_vector, cam_t = data_generator.generate_random_params()

    keypoints_visualizer = KeypointsVisualizer()
    vis_img = keypoints_visualizer.vis_from_params(
        pose=pose,
        shape=shape,
        cam_t=cam_t
    )

    body_visualizer = BodyVisualizer(device='cpu')
    body_img, body_mask = body_visualizer.vis_from_params(
        pose=pose,
        shape=shape,
        cam_t=cam_t,
        gender='male'
    )

    '''
    clothed_visualizer = ClothedVisualizer(
        gender='male',
        upper_class='t-shirt',
        lower_class='pant',
        device='cpu'
    )
    body_img, body_mask, joints_3d = clothed_visualizer.vis_from_params(
        pose=pose,
        shape=shape,
        style_vector=style_vector,
        cam_t=cam_t
    )
    '''

    '''
    visualizer_3d = ClothedVisualizer3D(
        gender='male',
        upper_class='t-shirt',
        lower_class='pant'
    )
    meshes = visualizer_3d.vis_from_params(
        pose=pose,
        shape=shape,
        style_vector=style_vector
    )
    visualizer_3d.save_vis(
        meshes=meshes,
        save_basepath='tests/functional/out/mesh'
    )
    '''

    # First, create visdom process as `visdom -p {args.vport} > /dev/null 2>&1`
    vport = 8889
    visdom = vis.Visdom(port=vport)

    smpl_model = easy_create_smpl_model(
        gender='male',
        device='cpu'
    )
    logger = VisLogger(
        device='cpu',
        visdom=visdom,
        smpl_model=smpl_model
    )
