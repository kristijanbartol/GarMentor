from typing import Optional
import torch
from torch.distributions import Normal
import numpy as np
from visdom import Visdom

from models.smpl_official import SMPL
from rendering.body import BodyRenderer
from vis.visualizers.keypoints import KeypointsVisualizer
from vis.visualizers.body import BodyVisualizer


class VisLogger():

    """
    A visualization logger class for debugging and verification during training.
    """
    
    _nrow = 4
    _ndist_samples = 200
    
    def __init__(
            self, 
            device: str, 
            visdom: Visdom, 
            smpl_model: SMPL
        ) -> None:
        """
        A visualization logger constructor.

        The Visdom object serves for the visualization utilities.

        The class takes an SMPL model as an initialization parameter to provide
        for both the KeypointsVisualizer and BodyVisualizer class.
        """
        self.device = device
        self.visdom = visdom
        self.kpts_vis = KeypointsVisualizer(
            device=self.device,
            smpl_model=smpl_model
        )
        self.body_vis = BodyVisualizer(
            device=self.device,
            smpl_model=smpl_model
        )
    
    def vis_rgb(
            self, 
            rgb_in: torch.Tensor, 
            label: str = 'rgb_in'
        ) -> None:
        """
        Visualize a batch of RGB images in Visdom.
        """
        self.visdom.images(rgb_in[:self._nrow], nrow=self._nrow, win=label,
                           opts={'title': label})

    def vis_edge(
            self, 
            edge_in: torch.Tensor,
            label: Optional[str] = 'edge_in'
        ) -> None:
        """
        Visualize a batch of edge maps.
        """
        edge_in_rgb = torch.cat((edge_in,) * 3, dim=1).squeeze(dim=2) * 255
        self.visdom.images(edge_in_rgb[:self._nrow], nrow=self._nrow, win=label)

    def vis_heatmaps(
            self, 
            j2d_heatmaps: torch.Tensor, 
            label: Optional[str] = 'j2d_heatmaps'
        ) -> None:
        """
        Visualize a batch of keypoint heatmaps.
        """
        colored_heatmaps = self.kpts_vis.batched_vis_heatmaps(
            heatmaps=j2d_heatmaps,
            num_heatmaps=self._nrow
        )
        self.visdom.images(
            colored_heatmaps, 
            nrow=self._nrow, 
            win=label,
            opts={'title': label}
        )
        
    def vis_poses(
            self, 
            keypoints: torch.Tensor, 
            label: Optional[str] = 'keypoints'
        ) -> None:
        """
        Visualize a batch of poses (keypoints + connections).
        """
        colored_kpts_imgs = self.kpts_vis.batched_vis_pose(
            kpts=keypoints,
            num_maps=self._nrow
        )
        self.visdom.images(
            colored_kpts_imgs, 
            nrow=self._nrow, 
            win=label,
            opts={'title': label}
        )
        
    def vis_pred_rgb(
            self, 
            pred_verts: torch.Tensor, 
            back_img: Optional[torch.Tensor] = None
        ) -> None:
        """
        Project and visualize body vertices with optional background image.
        """
        pred_img, pred_mask = self.body_vis.vis_body_from_verts(pred_verts)
        if back_img is not None:
            pred_img = self.body_vis.add_background(pred_img, pred_mask, back_img)
        
        rgb_out = pred_img  # (N, C, img_wh, img_wh)
        self.visdom.images(rgb_out[:self._nrow], nrow=self._nrow, win='rgb_out')
        
    def vis_shape_dist(
            self, 
            pred_shape_dist: Normal, 
            target_shape: torch.Tensor
        ) -> None:
        """
        Visualize the predicted shape distribution and the target shape parameters.
        """
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
