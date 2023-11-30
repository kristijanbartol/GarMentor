import argparse
import numpy as np
import os
import torch

from models.parametric_model import DNParametricModel
from rendering.clothed import TorchClothedRenderer
from utils.sampling_utils import sample_random_style


D_WIDTH = 512
DIM_THETA = 72
DIM_THETA_P = 128 
DIM_LATENT_G = 12
NUM_G = 100


def get_mesh_sdf(
        verts,  
        style, 
        model_G
    ):
    ### 2: evaluate analytical normals
    verts_torch = torch.from_numpy(verts).float().cuda()
    verts_torch.requires_grad = True
    num_points = len(verts_torch)
    x_cloth_points = style.unsqueeze(0).repeat(num_points, 1)
    pred_sdf_verts = model_G(verts_torch, x_cloth_points, num_points)
    pred_sdf_verts.sum().backward(retain_graph=True)

    normals = verts_torch.grad
    normals = normals / (torch.norm(normals, dim=1, keepdim=True) + 0.0001)

    ### 3: new verts= old vert + sdf (=0) * normals, to bring bck differentiability
    new_verts = verts_torch.detach() - pred_sdf_verts * normals

    return new_verts


def fit_style(
        pose_params,
        shape_params,
        z_style,  
        gt_mask,  
        renderer, 
        device,
        iters=100
    ):
    lr = 1e-2
    optimizer = torch.optim.Adam([{'params': z_style, 'lr': lr}])

    with torch.no_grad():
        smpl_output = smpl_server.forward(smpl_params)
        smpl_verts_pred = smpl_output['smpl_verts'].squeeze()
        smpl_tfs = smpl_output['smpl_tfs'].squeeze()
        
        smpl_verts_pred.requires_grad = False 
        smpl_tfs.requires_grad = False 

    smpl_faces = torch.Tensor(smpl_server.smpl.faces.astype(np.int32))

    best_loss = 1e10
    best_z_style = z_style.clone().detach()

    for step in range(iters):

        with torch.no_grad():
            cloth_mesh = reconstruct(z_style, model_G, just_vf=False, resolution=256)
            gar_verts, gar_faces = cloth_mesh.vertices, cloth_mesh.faces  # type: ignore

        torch_verts = get_mesh_sdf(gar_verts, z_style, model_G)

        # TODO: For memory efficiency, the upper and lower parts need to be optimized independently.
        verts_gar_deformed = deform(
            torch_verts, 
            smpl_tfs, 
            tfs_c_inv, 
            pose_params, 
            shape_params, 
            shapedirs, 
            tfs_weighted_zero, 
            embedder, 
            model_lbs, 
            model_lbs_delta, 
            model_blend_weight
        )

        meshes = create_meshes(
            verts_list=[
                smpl_verts_pred.unsqueeze(0),
                verts_gar_deformed.unsqueeze(0)
            ], # type: ignore
            faces_list=[
                smpl_faces.unsqueeze(0).to(device=device),
                torch.Tensor(gar_faces).unsqueeze(0).to(device=device)
            ] # type: ignore
        )
        pred_mask = renderer(meshes=meshes)
        intersection = (pred_mask * gt_mask).sum()
        union = pred_mask.sum() + gt_mask.sum() - intersection

        silh_loss = (1 - intersection / union) * 224
        reg_loss = z_style.norm() / 10
        loss = silh_loss + reg_loss

        print(f'============ iteration {step} =============')
        print(f'total loss: {loss.item()}, silh loss: ({silh_loss.item()}, reg loss: {reg_loss.item()}')

        if loss < best_loss:
            best_z_style = z_style

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return best_z_style.cpu().numpy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='gar')
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    parametric_model = DNParametricModel(
        device=args.device,
        gender='female'
    )
    # TODO: Need TorchClothedRenderer.
    renderer = TorchClothedRenderer(
        device=args.device,
        img_wh=512    
    )

    img_names = os.listdir(os.path.join(IMG_ROOTDIR, args.dataset))

    for img_name in img_names:
        # NOTE: Currently, I am using DrapeNet silhouettes and parameters, but DIG model for optimization.
        #       It should work but I shouldn't forget that detail.
        optimal_styles = []
        for body_part in ['upper', 'lower']:
            masks_dict, params_dict = load_gt(img_name, 'gar')

            init_style = sample_random_style(garment_part=body_part)

            pose_params = torch.from_numpy(params_dict['pose']).unsqueeze(dim=0).float().to(device=args.device)
            shape_params = torch.from_numpy(params_dict['shape']).unsqueeze(dim=0).float().to(device=args.device)
            pose_params.requires_grad = False 
            shape_params.requires_grad = False 

            optimal_styles.append(fit_style(
                pose_params=pose_params,
                shape_params=shape_params,
                z_style=init_style,
                gt_mask=masks_dict[body_part], 
                renderer=renderer,
                device=args.device, 
                iters=100
            )
        )
        # TODO: Evaluate fitting (now use known style parameters).
