from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.cuda
import os
from PIL import Image
import sys

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
        pred_rgb, 
        pred_segmaps, 
        gt_rgb,
        gt_segmaps,
        clothed_visualizer
    ):
    pred_rgb = prepare_img(rgb_img=pred_rgb)
    pred_segmaps = get_masks_as_images(seg_masks=pred_segmaps)
    gt_rgb = prepare_img(rgb_img=gt_rgb)
    gt_segmaps = get_masks_as_images(seg_masks=gt_segmaps)

    fig = plt.figure(figsize=(10, 7))
    fig.add_subplot(2, 4, 1)
    plt.imshow(pred_rgb)
    fig.add_subplot(2, 4, 2)
    plt.imshow(pred_segmaps[0])
    fig.add_subplot(2, 4, 3)
    plt.imshow(pred_segmaps[1])
    fig.add_subplot(2, 4, 4)
    plt.imshow(pred_segmaps[2])
    fig.add_subplot(2, 4, 5)
    plt.imshow(gt_rgb)
    fig.add_subplot(2, 4, 6)
    plt.imshow(gt_segmaps[0])
    fig.add_subplot(2, 4, 7)
    plt.imshow(gt_segmaps[1])
    fig.add_subplot(2, 4, 8)
    plt.imshow(gt_segmaps[2])

    fig.savefig(save_path)


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


def generate_sample_pair(
        pred_dict,
        clothed_visualizer
    ):
    pred_rgb, pred_segmaps = generate_sample(
        pred_dict=pred_dict,
        clothed_visualizer=clothed_visualizer
    )
    pred_dict['style_mean'] = pred_dict['style_gt']
    gt_rgb, gt_segmaps = generate_sample(
        pred_dict=pred_dict,
        clothed_visualizer=clothed_visualizer
    )
    return pred_rgb, pred_segmaps, gt_rgb, gt_segmaps


def evaluate(
        pred_dict,
        clothed_visualizer
    ):
    save_dir = 'output/preds/qualitative/valid/'
    for sample_idx in range(pred_dict['pose_gt'].shape[0]):
        save_path = os.path.join(save_dir, f'{sample_idx}.png')
        if os.path.exists(save_path):
            continue
        sample_dict = {key: pred_dict[key][sample_idx] for key in pred_dict}
        pred_rgb, pred_segmaps, gt_rgb, gt_segmaps = generate_sample_pair(
            pred_dict=sample_dict,
            clothed_visualizer=clothed_visualizer
        )
        if pred_rgb is None or gt_rgb is None:
            continue
        save_sample(
            save_path=save_path,
            pred_rgb=pred_rgb, 
            pred_segmaps=pred_segmaps,
            gt_rgb=gt_rgb,
            gt_segmaps=gt_segmaps,
            clothed_visualizer=clothed_visualizer
        )
        torch.cuda.empty_cache()


if __name__ == '__main__':
    clothed_visualizer = DNClothedVisualizer(
        device='cuda:0',
        gender='female'
    )
    pred_dict = np.load('output/preds/npz/valid/pred_10.npz')
    evaluate(
        pred_dict=pred_dict,
        clothed_visualizer=clothed_visualizer
    )
