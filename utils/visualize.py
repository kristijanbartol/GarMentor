import torch
import numpy as np

from utils.rigid_transform_utils import aa_rotate_translate_points_pytorch3d


class VisLogger():
    
    _nrow = 4
    _ndist_samples = 200
    
    def __init__(self, visdom, renderer):
        self.visdom = visdom
        self.renderer = renderer
    
    def vis_rgb(self, rgb_in):
        self.visdom.images(rgb_in[:self._nrow], nrow=self._nrow, win='rgb_in')

    def vis_edge(self, edge_in):
        edge_in_rgb = torch.cat((edge_in,) * 3, dim=1).squeeze(dim=2) * 255
        self.visdom.images(edge_in_rgb[:self._nrow], nrow=self._nrow, win='edge_in')

    def vis_j2d_heatmaps(self, j2d_heatmaps):
        j2d_heatmaps_rgb = torch.stack(
            (torch.sum(j2d_heatmaps, dim=1).unsqueeze(dim=1),) * 3, dim=1).squeeze(dim=2) * 255
        self.visdom.images(j2d_heatmaps_rgb[:self._nrow], nrow=self._nrow, win='j2d_heatmaps')
        
    def vis_pred_rgb(self, pred_verts, x_axis, angle, trans, texture, cam_t, settings):
        pred_verts = aa_rotate_translate_points_pytorch3d(
            points=pred_verts,
            axes=x_axis,
            angles=angle,
            translations=trans)
        
        renderer_pred_output = self.renderer(
            vertices=pred_verts,
            textures=texture,
            cam_t=cam_t,
            lights_rgb_settings=settings)
        
        rgb_out = renderer_pred_output['rgb_images'].permute(0, 3, 1, 2).contiguous()  # (N, C, img_wh, img_wh)
        self.visdom.images(rgb_out[:self._nrow], nrow=self._nrow, win='rgb_out')
        
    def vis_shape_dist(self, pred_shape_dist, target_shape):
        pred_shape_means = pred_shape_dist.loc.cpu().detach().numpy() 
        pred_shape_stds = pred_shape_dist.scale.cpu().detach().numpy()
        target_shape_np = target_shape.cpu().detach().numpy()
        
        for irow in range(self._nrow):
            i_pred_shape_means, i_pred_shape_stds = pred_shape_means[irow], pred_shape_stds[irow]
            
            dists = []
            for pca_idx, (mean, std) in enumerate(zip(i_pred_shape_means, i_pred_shape_stds)):
                dist = np.ones((self._ndist_samples, 2)) * pca_idx
                dist[:, 0] = np.random.normal(loc=mean, scale=std, size=(self._ndist_samples,))
                dists.append(dist)

            dists = np.concatenate(dists, axis=0)
            gts = np.array([[target_shape_np[irow, pca_idx], pca_idx] for pca_idx in range(10)])

            all_data = np.concatenate((dists, gts), axis=0)

            colors = np.concatenate((
                np.expand_dims(np.array([0,0,255]), axis=0).repeat((dists.shape[0],), axis=0),
                np.expand_dims(np.array([255,0,0]), axis=0).repeat((gts.shape[0],), axis=0)
                ), axis=0                            
            )
            self.visdom.scatter(
                X=all_data, 
                win=f'shape_dists_{irow}', 
                opts=dict(markersymbol='line-ns-open', markercolor=colors))
