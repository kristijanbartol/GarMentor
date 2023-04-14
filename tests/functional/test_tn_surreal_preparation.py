import sys
import cv2

sys.path.append('/garmentor')


from data.prepare.surreal.tn import TnSurrealDataGenerator
from utils.garment_classes import GarmentClasses
from vis.visualizers.tn_clothed import TnClothedVisualizer
from vis.visualizers.keypoints import KeypointsVisualizer


if __name__ == '__main__':
    upper_class = 't-shirt'
    lower_class = 'pant'
    gender = 'male'

    surreal_generator = TnSurrealDataGenerator(
        preextract_kpt=True
    )

    garment_classes = GarmentClasses(upper_class, lower_class)
    clothed_visualizer = TnClothedVisualizer(
        device='cuda:0',
        gender=gender,
        garment_classes=garment_classes
    )
    kpt_visualizer = KeypointsVisualizer()

    rgb_img, seg_maps, sample_values = surreal_generator.generate_sample(
        idx = 10,
        gender='male',
        clothed_visualizer=clothed_visualizer
    )

    cv2.imshow('rgb', rgb_img)
    cv2.waitKey(0)

    for i in range(5):
        cv2.imshow(f'seg{i}', seg_maps[i])
        cv2.waitKey(0)
  
    # closing all open windows
    cv2.destroyAllWindows()

    kpts_img = kpt_visualizer.vis_keypoints(
        kpts=sample_values.joints_2d,
        back_img=rgb_img
    )
    cv2.imshow('kpts', kpts_img)
    cv2.waitKey(0)
