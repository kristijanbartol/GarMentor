import sys
from pathlib import Path

sys.path.append('/garmentor/')

from models.smpl_official import easy_create_smpl_model
from vis.visualizers.keypoints import KeypointsVisualizer
from vis.visualizers.body import BodyVisualizer
from vis.visualizers.clothed import ClothedVisualizer
from vis.visualizers.clothed3d import ClothedVisualizer3D
from vis.logger import VisLogger
from data.generate.pregenerator import DataPreGenerator


if __name__ == '__main__':
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
