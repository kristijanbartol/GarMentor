import sys

sys.path.append('/garmentor/')

from vis.visualizers.keypoints import KeypointsVisualizer
from data.generate.pregenerator import DataPreGenerator

if __name__ == '__main__':
    data_generator = DataPreGenerator()
    pose, shape, _, cam_t = data_generator.generate_random_params()

    keypoints_visualizer = KeypointsVisualizer()
    keypoints_visualizer.vis_from_params(
        pose=pose,
        shape=shape,
        cam_t=cam_t
    )
