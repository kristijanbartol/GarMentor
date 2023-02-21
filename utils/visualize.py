import torch
import numpy as np

from utils.rigid_transform_utils import aa_rotate_translate_points_pytorch3d


COLORS = {
    'red': (255, 0, 0),                 # 0
    'lime': (0, 255, 0),                # 1
    'blue': (0, 0, 255),                # 2
    'yellow': (255, 255, 0),            # 3
    'cyan': (0, 255, 255),              # 4
    'magenta': (255, 0, 255),           # 5
    'silver': (192, 192, 192),          # 6
    'maroon': (128, 0, 0),              # 7
    'green': (0, 128, 0),               # 8
    'purple': (128, 0, 128),            # 9
    'wheat': (245, 222, 179),           # 10
    'deeppink': (255, 20, 147),         # 11
    'white': (255, 255, 255),           # 12
    'indigo': (75, 0, 130),             # 13
    'midnightblue': (25, 25, 112),      # 14
    'lightskyblue': (135, 206, 250),    # 15
    'orange': (255, 165, 0)             # 16
}


class VisLogger():
    
    _nrow = 4
    _ndist_samples = 200
    
    def __init__(self, visdom, renderer=None):
        self.visdom = visdom
        self.renderer = renderer
    
    def vis_rgb(self, rgb_in, label='rgb_in'):
        self.visdom.images(rgb_in[:self._nrow], nrow=self._nrow, win=label,
                           opts={
                               'title': label
                           })

    def vis_edge(self, edge_in):
        edge_in_rgb = torch.cat((edge_in,) * 3, dim=1).squeeze(dim=2) * 255
        self.visdom.images(edge_in_rgb[:self._nrow], nrow=self._nrow, win='edge_in')

    def vis_j2d_heatmaps(self, j2d_heatmaps, label='j2d_heatmaps'):
        colored_h2d_heatmap = torch.zeros(self._nrow, 3, 256, 256).to('cuda:0')
        for color_idx, color_key in enumerate(COLORS):
            heatmap = torch.stack((j2d_heatmaps[:self._nrow, color_idx],) * 3, dim=1)
            color_tensor = torch.tensor(COLORS[color_key])
            heatmap[:, 0] *= color_tensor[0]
            heatmap[:, 1] *= color_tensor[1]
            heatmap[:, 2] *= color_tensor[2]
            colored_h2d_heatmap += heatmap
        self.visdom.images(colored_h2d_heatmap[:self._nrow], nrow=self._nrow, win=label,
                           opts={
                               'title': label
                           })
        
    def vis_keypoints(self, keypoints, label='keypoints'):
        colored_keypoints = torch.zeros(1, 3, 256, 256).to('cuda:0')
        for color_idx, color_key in enumerate(COLORS):
            x, y = [int(_x) for _x in keypoints[0, color_idx]]
            colored_keypoints[0, :, x, y-1] = torch.tensor(COLORS[color_key])
            colored_keypoints[0, :, x, y] = torch.tensor(COLORS[color_key])
            colored_keypoints[0, :, x, y+1] = torch.tensor(COLORS[color_key])
        self.visdom.images(colored_keypoints[:self._nrow], nrow=self._nrow, win=label,
                           opts={
                               'title': label
                           })
        
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
