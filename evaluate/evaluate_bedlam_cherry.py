from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.cuda
import os
from PIL import Image
import sys
import cv2
import trimesh

sys.path.append('/GarMentor')

import configs.paths as paths
from vis.visualizers.clothed import DNClothedVisualizer
from models.parametric_model import DNParametricModel
from utils.eval_utils import (
    calc_chamfer_distance,
    bcc
)


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


def evaluate_qualitative(
        pred_dict
    ):
    clothed_visualizer = DNClothedVisualizer(
        device='cuda:0',
        gender='female'
    )
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


def evaluate_quantitative(
        npz_folder,
        img_folder,
        garment_folder,
        pred_dict
    ):
    parametric_model = DNParametricModel(
        device='cuda:0', 
        gender='female'
    )
    cds = []
    bccs = []
    for npz_fname in os.listdir(npz_folder):
        npz_dict = np.load(
            os.path.join(npz_folder, npz_fname), 
            allow_pickle=True
        )
        npz_basename = os.path.basename(npz_fname).split('.')[0]
        for idx in range(npz_dict['imgname'].shape[0]):
            pred_dict_key = os.path.join(
                'output/crop/bedlam-cherry',
                npz_basename,
                npz_dict['imgname'][idx]
            )
            if pred_dict_key in pred_dict and 'suburb' in pred_dict_key:
                seq_id = npz_dict['seq_id'][idx]
                garment_path = os.path.join(
                    garment_folder,
                    npz_dict['sub'][idx],
                    'clothing_simulations/',
                    seq_id,
                    f'{seq_id}.npz'
                )
                if os.path.exists(garment_path):
                    gt_dict = {
                        x: npz_dict[x][idx] for x in npz_dict.keys()
                    }
                    garment_start_frame = gt_dict['start_frame']
                    gt_verts = np.load(garment_path)['vertices_seq'][idx+garment_start_frame]
                    gt_faces = np.load(garment_path)['faces']

                    print(f'Processing {pred_dict_key}...')
                    drapenet_dict = parametric_model.run(
                        pose=gt_dict['pose_orig'],
                        shape=gt_dict['shape'][1:],
                        style_vector=pred_dict[pred_dict_key]
                    )
                    pred_verts = np.concatenate([
                        drapenet_dict['upper'].garment_verts,
                        drapenet_dict['lower'].garment_verts
                    ], axis=0)
                    pred_faces = np.concatenate([
                        drapenet_dict['upper'].garment_faces,
                        drapenet_dict['lower'].garment_faces + drapenet_dict['upper'].garment_verts.shape[0]
                    ], axis=0)
                    pred_verts += gt_dict['trans_orig']
                    cd = calc_chamfer_distance(
                        x=pred_verts,
                        y=gt_verts
                    )
                    _bcc = bcc(
                        body_verts=drapenet_dict['upper'].body_verts,
                        pred_cloth_verts=pred_verts,
                        gt_cloth_verts=gt_verts
                    )[0]
                    cds.append(cd)
                    bccs.append(_bcc)
                    print(f'Chamfer distance: {cds[-1]} ({np.array(cds).mean()})')
                    print(f'BCC: {bccs[-1]} ({np.array(bccs).mean()})')

                    pred_mesh = trimesh.Trimesh(
                        vertices=pred_verts,
                        faces=pred_faces
                    )
                    gt_mesh = trimesh.Trimesh(
                        vertices=gt_verts,
                        faces=gt_faces
                    )
                    img_basename = npz_dict['imgname'][idx].split('/')[1].split('.')[0]
                    with open(f'output/meshes/pred-{img_basename}.obj', 'w', encoding='utf-8') as f:
                        pred_mesh.export(f, file_type='obj')
                    with open(f'output/meshes/gt-{img_basename}.obj', 'w', encoding='utf-8') as f:
                        gt_mesh.export(f, file_type='obj')


if __name__ == '__main__':
    pred_dict = np.load('output/preds/npz/bedlam-cherry/preds.npz')
    #evaluate_qualitative(
    #    pred_dict=pred_dict
    #)
    evaluate_quantitative(
        npz_folder='/data/bedlam/npz_processed/',
        img_folder='/data/bedlam/images/',
        garment_folder='/data/bedlam/clothing_npz_30fps/',
        pred_dict=pred_dict
    )
