from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.cuda
import os
from PIL import Image
import sys
import cv2

sys.path.append('/GarMentor')

import configs.paths as paths
from vis.visualizers.clothed import DNClothedVisualizer


def get_masks_as_images(seg_masks):
    mask_imgs = []
    for seg_idx in range(seg_masks.shape[0]):
        mask_img = (seg_masks[seg_idx] * 255).astype(np.uint8)
        mask_imgs.append(Image.fromarray(mask_img))
    return mask_imgs


def prepare_img(rgb_img):
    rgb_img = (rgb_img * 255).astype(np.uint8)
    #return Image.fromarray(rgb_img)
    return rgb_img

    
def check_style_blacklist(style):
    if not os.path.exists('invalid_styles_eval.npy'):
        invalid_styles = []
    else:
        invalid_styles = np.load('invalid_styles_eval.npy')
    is_blacklisted = False
    for invalid_style in invalid_styles:
        if np.array_equal(style[0], invalid_style[0]) and np.array_equal(style[1], invalid_style[1]):
            print(f'WARNING: Loaded blacklisted style!')
            is_blacklisted = True
            break
    return is_blacklisted

def verify_cuda(
        rgb_img,
        current_style
    ) -> None:
    if rgb_img is None:
        if not os.path.exists('invalid_styles_eval.npy'):
            invalid_styles = []
        else:
            invalid_styles = list(np.load('invalid_styles_eval.npy'))
        invalid_styles.append(current_style)
        np.save('invalid_styles_eval.npy', np.array(invalid_styles))
        print(f'Runtime error! Logged this sample to the invalid ones ({len(invalid_styles)}). Now exiting...')
        exit()

def verify_appearance(rgb_img) -> bool:
    print(torch.any(rgb_img, dim=-1).sum())
    if torch.any(rgb_img, dim=-1).sum() > 0.125 * rgb_img.shape[0] * rgb_img.shape[1]:
        print(f'WARNING: Appearance wrong!')
        return False
    else:
        return True

def produce_features(
        clothed_visualizer,
        pred_dict
    ):
    if check_style_blacklist(pred_dict['style_mean']):
        return None, None, False
    rgb_img, seg_maps, _ = clothed_visualizer.vis_from_params(
        pose=pred_dict['pose_gt'],
        shape=pred_dict['shape_gt'],
        style_vector=pred_dict['style_mean']
    )
    verify_cuda(
        rgb_img=rgb_img,
        current_style=pred_dict['style_mean']
    )
    rgb_ok = verify_appearance(rgb_img)
    return rgb_img, seg_maps, rgb_ok


def save_sample(
        save_path, 
        img_path,
        pred_rgb, 
        pred_segmaps
    ):
    pred_rgb = prepare_img(rgb_img=pred_rgb)
    pred_segmaps = get_masks_as_images(seg_masks=pred_segmaps)

    fig = plt.figure(figsize=(10, 7))
    fig.add_subplot(2, 4, 1)
    plt.imshow(pred_rgb)
    fig.add_subplot(2, 4, 2)
    plt.imshow(pred_segmaps[0])
    fig.add_subplot(2, 4, 3)
    plt.imshow(pred_segmaps[1])
    fig.add_subplot(2, 4, 4)
    plt.imshow(pred_segmaps[2])

    img = cv2.imread(img_path)
    mask_paths = [f'{os.path.join(img_path.replace("crop", "masks")).split(".")[0]}_{x}.png' for x in ['upper-cloth', 'lower-cloth', 'whole-body']]
    masks = [cv2.imread(mask_paths[x]) for x in range(3)]

    fig.add_subplot(2, 4, 5)
    plt.imshow(img[:, :, ::-1])
    fig.add_subplot(2, 4, 6)
    plt.imshow(masks[0])
    fig.add_subplot(2, 4, 7)
    plt.imshow(masks[1])
    fig.add_subplot(2, 4, 8)
    plt.imshow(masks[2])

    fig.savefig(save_path)
    print(f'Saved figure {save_path}...')


def generate_sample(
        pred_dict,
        clothed_visualizer
    ):
    # TODO: Also get the predicted meshes to calculate metrics.
    rgb_img, seg_maps, rgb_ok = produce_features(
        clothed_visualizer=clothed_visualizer,
        pred_dict=pred_dict
    )
    if not rgb_ok:
        return None, None
    rgb_img = rgb_img.cpu().numpy()
    
    return (
        rgb_img,
        seg_maps
    )


def generate_pred_sample(
        pred_dict,
        clothed_visualizer
    ):
    #pred_dict['style_mean'] = np.flip(pred_dict['style_mean'], axis=0).copy()
    pred_rgb, pred_segmaps = generate_sample(
        pred_dict=pred_dict,
        clothed_visualizer=clothed_visualizer
    )
    return pred_rgb, pred_segmaps


def _prepare_pred_dict(style):
    return {
        'style_mean': style,
        'pose_gt': np.zeros(72,),
        'shape_gt': np.zeros(10,)
    }


def evaluate(
        pred_dict,
        clothed_visualizer
    ):
    save_superdir = 'output/preds/qualitative/bedlam-cherry/'
    for img_path in pred_dict:
        save_path = os.path.join(
            save_superdir, 
            '/'.join(img_path.split('/')[-3:])
        )
        save_dir = '/'.join(save_path.split('/')[:-1])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        sample_dict = _prepare_pred_dict(style=pred_dict[img_path])
        pred_rgb, pred_segmaps = generate_pred_sample(
            pred_dict=sample_dict,
            clothed_visualizer=clothed_visualizer
        )
        if pred_rgb is None:
            continue
        save_sample(
            save_path=save_path,
            img_path=img_path,
            pred_rgb=pred_rgb, 
            pred_segmaps=pred_segmaps
        )
        torch.cuda.empty_cache()


if __name__ == '__main__':
    clothed_visualizer = DNClothedVisualizer(
        device='cuda:0',
        gender='female'
    )
    pred_dict = np.load('output/preds/npz/bedlam-cherry/preds.npz')
    evaluate(
        pred_dict=pred_dict,
        clothed_visualizer=clothed_visualizer
    )
