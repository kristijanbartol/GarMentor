import sys
from pathlib import Path

sys.path.append('/garmentor/')

from vis.visualizers.keypoints import KeypointsVisualizer
from vis.visualizers.body import BodyVisualizer
from data.generate.pregenerator import DataPreGenerator

if __name__ == '__main__':
    data_generator = DataPreGenerator()
    pose, shape, _, cam_t = data_generator.generate_random_params()

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
