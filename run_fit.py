import argparse
import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.nn.parameter import Parameter
from data.mesh_managers.colored_garments import create_meshes_torch

from evaluate.evaluate_fitting import evaluate
from fitting.data_loaders import load_gt, prepare_gar
from models.parametric_model import TorchParametricModel
from rendering.clothed import TorchClothedRenderer
from utils.sampling_utils import sample_random_style


def fit_style(
        pose_params,
        shape_params,
        z_style,  
        gt_mask,  
        renderer, 
        parametric_model,
        smpl_output,
        body_part,
        device,
        iters=100
    ):
    #lr = 1e-2
    lr = 1.5e-3
    optimizer = torch.optim.Adam([{'params': z_style, 'lr': lr}])

    best_z_style = z_style.clone().detach()
    best_loss = 1e10

    for step in range(iters):
        cloth_verts, cloth_faces = parametric_model.run(
            pose=pose_params,
            shape=shape_params,
            style_vector=z_style,
            smpl_output=smpl_output,
            garment_part=body_part
        )
        meshes = create_meshes_torch(
            verts_list=[
                smpl_output.vertices,
                cloth_verts.unsqueeze(0)
            ],
            faces_list=[
                torch.from_numpy(smpl_output.faces.astype(np.int32)).unsqueeze(0).to(device=device),
                cloth_faces.int().unsqueeze(0).to(device=device)
            ]
        )
        pred_mask = renderer(meshes=meshes)
        intersection = (pred_mask * gt_mask).sum()
        union = pred_mask.sum() + gt_mask.sum() - intersection

        silh_loss = (1 - intersection / union) * 224
        reg_loss = z_style.norm() / 10
        loss = silh_loss + reg_loss

        print(f'============ iteration {step} =============')
        print(f'total loss: {loss.item()}, silh loss: {silh_loss.item()}, reg loss: {reg_loss.item()}')

        if loss < best_loss:
            best_z_style = z_style

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 80 == 0 and step != 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * 0.7

        if step == 0:
            first_fit = torch.tile(pred_mask, (3, 1, 1)).cpu().detach().numpy().astype(np.float32).swapaxes(0, 2)
        with torch.no_grad():
            #cv2.imwrite(f'output/fitting/{step}_pred.png', torch.tile(pred_mask, (3, 1, 1)).cpu().detach().numpy().astype(np.float32).swapaxes(0, 2) * 255)
            #cv2.imwrite(f'output/fitting/{step}_gt.png', torch.tile(gt_mask, (3, 1, 1)).cpu().detach().numpy().astype(np.float32).swapaxes(0, 2) * 255)

            pred_rgb = torch.tile(pred_mask, (3, 1, 1)).cpu().detach().numpy().astype(np.float32).swapaxes(0, 2)
            gt_rgb = torch.tile(gt_mask, (3, 1, 1)).cpu().detach().numpy().astype(np.float32).swapaxes(0, 2)

            fig = plt.figure(figsize=(20, 5))
            fig.add_subplot(1, 3, 1)
            plt.imshow(first_fit)
            fig.add_subplot(1, 3, 2)
            plt.imshow(pred_rgb)
            fig.add_subplot(1, 3, 3)
            plt.imshow(gt_rgb)

            fig.savefig('output/fitting/optim.png')
        
    return best_z_style


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='gar')
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    parametric_model = TorchParametricModel(
        device=args.device,
        gender='female'
    )
    renderer = TorchClothedRenderer(
        device=args.device,
        img_wh=256    
    )
    prepare_gar()
    img_names = os.listdir(os.path.join('/data/tailornet/fitting/images/', args.dataset))

    for img_name in img_names:
        # NOTE: Currently, I am using DrapeNet silhouettes and parameters, but DIG model for optimization.
        #       It should work but I shouldn't forget that detail.
        optimal_styles = []
        masks_dict, params_dict = load_gt(
            img_name, 
            'gar', 
            img_size=512
        )
        pose_params = torch.from_numpy(params_dict['pose']).unsqueeze(dim=0).float().to(device=args.device)
        shape_params = torch.from_numpy(params_dict['shape']).unsqueeze(dim=0).float().to(device=args.device)
        gt_params = torch.stack([
            torch.from_numpy(params_dict['upper_style']).to(args.device),
            torch.from_numpy(params_dict['lower_style']).to(args.device)
        ], dim=0).unsqueeze(1)

        smpl_output = parametric_model.get_body_output(
            pose=pose_params[:, 3:],
            shape=shape_params, 
            global_orient=pose_params[:, :3], 
            return_verts=True
        )
        for body_part in ['upper', 'lower']:
            gt_mask = torch.from_numpy(masks_dict[body_part]).float().unsqueeze(0).to(args.device)

            init_style = Parameter(sample_random_style(garment_part=body_part).unsqueeze(dim=0).to(device=args.device))
            #init_style = Parameter(torch.from_numpy(np.load('style_params.npy')).unsqueeze(dim=0).to(device=args.device))

            optimal_styles.append(fit_style(
                pose_params=pose_params,
                shape_params=shape_params,
                z_style=init_style,
                gt_mask=gt_mask,
                renderer=renderer,
                parametric_model=parametric_model, 
                smpl_output=smpl_output,
                body_part=body_part,
                device=args.device, 
                iters=2
                )
            )
            evaluate(
                pred_style_params=optimal_styles, 
                gt_style_params=gt_params,
                pose_params=pose_params,
                shape_params=shape_params,
                smpl_output=smpl_output,
                parametric_model=parametric_model,
                garment_part=body_part
            )
