from abc import abstractmethod
from typing import Union, Tuple, Optional
import torch
import numpy as np
import cv2
from psbody.mesh import Mesh

from configs.const import MEAN_CAM_T
from data.datasets.off_the_fly_train_datasets import SurrealTrainDataset
from data.mesh_managers.textured_garments import TexturedGarmentsMeshManager
from models.parametric_model import ParametricModel
from models.smpl_official import SMPL
from rendering.clothed_renderer import ClothedRenderer
from rendering.body_renderer import BodyRenderer
from utils.garment_classes import GarmentClasses
from utils.image_utils import add_rgb_background
from utils.convert_arrays import to_tensors
from utils.label_conversions import (
    COCO_START_IDXS,
    COCO_END_IDXS,
    COCO_LR
)
from vis.colors import (
    KPT_COLORS,
    LCOLOR,
    RCOLOR
)

from tailornet_for_garmentor.models.smpl4garment_utils import SMPL4GarmentOutput


class Visualizer(object):
    
    @abstractmethod
    def vis(self,
            kpts: Optional[np.ndarray] = None,
            back_img: Optional[np.ndarray] = None,
            skeleton: Optional[bool] = None,
            verts: Optional[Union[np.ndarray, torch.Tensor]] = None,
            smpl_output_dict: Optional[SMPL4GarmentOutput] = None,
            cam_t: Optional[np.ndarray] = None
        ) -> Union[Tuple[Mesh, Mesh, Mesh],                     # Visualizer3D
               Union[Tuple[np.ndarray, np.ndarray],         # BodyVisualizer
                     Tuple[torch.Tensor, torch.Tensor]],    
               Tuple[np.ndarray, np.ndarray],               # ClothedVisualizer
               np.ndarray]: ...                             # KeypointsVisualizer

    @abstractmethod
    def vis_from_params(
            self,
            pose: Union[np.ndarray, torch.Tensor], 
            shape: Union[np.ndarray, torch.Tensor], 
            style_vector: Optional[Union[np.ndarray, torch.Tensor]] = None,
            cam_t: Optional[np.ndarray] = None
    ) -> Union[Tuple[Mesh, Mesh, Mesh],                     # Visualizer3D
               Union[Tuple[np.ndarray, np.ndarray],         # BodyVisualizer
                     Tuple[torch.Tensor, torch.Tensor]],    
               Tuple[np.ndarray, np.ndarray],               # ClothedVisualizer
               np.ndarray]: ...                             # KeypointsVisualizer


class Visualizer2D(Visualizer):

    default_glob_orient = torch.Tensor([0., 0., 0.])
    default_cam_t = np.array(MEAN_CAM_T)

    def __init__(
            self,
            backgrounds_dir_path: str = None
    ) -> None:
        super().__init__()
        self.backgrounds_dir_path = backgrounds_dir_path

        self.background_paths = None
        if backgrounds_dir_path is not None:
            self.backgrounds_paths = SurrealTrainDataset._get_background_paths(
                backgrounds_dir_path=backgrounds_dir_path,
                num_backgrounds=1000
            )

    def add_background(
            self, 
            rgb_img: Union[np.ndarray, torch.Tensor],
            mask: Union[np.ndarray, torch.Tensor],
            back_img: Union[np.ndarray, torch.Tensor]
        ) -> Union[np.ndarray, torch.Tensor]:
        '''Add random 2D background "behind" rendered person based on mask.'''
        if self.background_paths is not None and back_img is not None:
            if back_img is None:
                back_img = SurrealTrainDataset.load_background(
                    backgrounds_paths=self.backgrounds_paths,
                    img_wh=self.img_wh
                ).to(self.device)

            rgb_img = add_rgb_background(
                backgrounds=back_img,
                rgb=rgb_img,
                seg=mask
            )
        else:
            print('WARNING: The background paths not provided.'\
                ' Returning the original image.')
        return rgb_img


class Visualizer3D(Visualizer):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def save_vis(
            self,
            meshes: Tuple[Mesh, Mesh, Mesh],
            rel_path: str
    ) -> None: ...


class KeypointsVisualizer(Visualizer2D):

    def __init__(
            self, 
            device: str, 
            img_wh: int = 256,
            backgrounds_dir_path: str = None
        ) -> None:
        super().__init__(backgrounds_dir_path)

        self.device = device
        self.img_wh = img_wh

    def batched_vis_heatmaps(
            self,
            heatmaps: torch.Tensor, 
            num_heatmaps: int
        ) -> torch.Tensor:
        '''Visualize a number of colored heatmaps, given the batch of keypoints.'''
        colored_heatmaps = torch.zeros(num_heatmaps, 3, self.img_wh, self.img_wh).to(self.device)
        for color_idx, color_key in enumerate(KPT_COLORS):
            heatmaps = torch.stack((heatmaps[:num_heatmaps, color_idx],) * 3, dim=1)
            color_tensor = torch.tensor(KPT_COLORS[color_key])
            heatmaps[:, 0] *= color_tensor[0]
            heatmaps[:, 1] *= color_tensor[1]
            heatmaps[:, 2] *= color_tensor[2]
            colored_heatmaps += heatmaps
        return colored_heatmaps

    def vis_heatmap_torch(
            self,
            heatmap: torch.Tensor
    ) -> torch.Tensor:
        '''Visualize a colored heatmap based on given heatmap in Torch.'''
        colored_heatmap = torch.zeros(3, self.img_wh, self.img_wh).to(self.device)
        for color_idx, color_key in enumerate(KPT_COLORS):
            heatmap = torch.stack((heatmap[color_idx],) * 3, dim=1)
            color_tensor = torch.tensor(KPT_COLORS[color_key])
            heatmap[0] *= color_tensor[0]
            heatmap[1] *= color_tensor[1]
            heatmap[2] *= color_tensor[2]
            colored_heatmaps += heatmap
        return colored_heatmap

    def vis_heatmap_numpy(
            self,
            heatmap: np.ndarray
    ) -> np.ndarray:
        '''Visualize a colored heatmap based on given heatmap in Numpy.'''
        colored_heatmap = np.zeros(3, self.img_wh, self.img_wh)
        for color_idx, color_key in enumerate(KPT_COLORS):
            heatmap = np.stack((heatmap[color_idx],) * 3, dim=1)
            heatmap[0] *= KPT_COLORS[color_key][0]
            heatmap[1] *= KPT_COLORS[color_key][1]
            heatmap[2] *= KPT_COLORS[color_key][2]
            colored_heatmaps += heatmap
        return colored_heatmap
    
    def vis_keypoints(
            self,
            kpts: np.ndarray,
            back_img: Optional[np.ndarray] = None
    ) -> np.ndarray:
        '''Visualize a colored image of keypoints, given coordinates.'''
        if back_img is None:
            back_img = np.zeros(3, self.img_wh, self.img_wh)
            
        for idx, color_key in enumerate(KPT_COLORS):
            kpt = kpts[idx]
            color = KPT_COLORS[color_key]
            cv2.circle(back_img, kpt, 3, color, 3)
        return back_img

    def _add_skeleton(
            self, 
            kpts: np.ndarray,
            pose_img: np.ndarray
    ) -> np.ndarray:
        '''Add line connections between the joints (COCO-specific).'''
        for line_idx, start_idx in COCO_START_IDXS:
            start_kpt = kpts[start_idx]
            end_kpt = kpts[COCO_END_IDXS[line_idx]]
            color = LCOLOR if COCO_LR[line_idx] else RCOLOR
            cv2.line(pose_img, start_kpt, end_kpt, color, 2) 
        return pose_img

    def vis(self,
            kpts: np.ndarray,
            back_img: Optional[np.ndarray] = None,
            skeleton: Optional[bool] = True
    ) -> np.ndarray:
        '''Visualize a colored image of the pose, given coordinates.'''
        pose_img = self.vis_keypoints(kpts, back_img)
        if skeleton:
            self._add_skeleton(kpts, pose_img)
        return pose_img

    def vis_from_params(
            self,
            pose: np.ndarray,
            shape: np.ndarray,
            glob_orient: Optional[np.ndarray] = None,
            cam_t: Optional[np.ndarray] = None
    ):
        if glob_orient is None:
            glob_orient = self.default_glob_orient
        if cam_t is None:
            cam_t = self.default_cam_t

        

    def overlay_pose(
            self,
            kpts: np.ndarray,
            back_img: np.ndarray
    ) -> np.ndarray:
        '''Overlay pose on top of the background image.'''
        return self.vis_pose(kpts, back_img)


class BodyVisualizer(Visualizer2D):

    def __init__(
            self, 
            device: str,
            backgrounds_dir_path: str = None,
            smpl_model: SMPL = None
        ) -> None:
        super().__init__(backgrounds_dir_path)

        self.device = device
        self.renderer = BodyRenderer(
            device=self.device,
            batch_size=1
        )
        self.smpl_model = smpl_model

    def vis(
        self,
        verts: Union[np.ndarray, torch.Tensor],
        cam_t: Optional[np.ndarray] = None
    ) -> Union[Tuple[np.ndarray, np.ndarray],
               Tuple[torch.Tensor, torch.Tensor]]:
        '''Render body using simple rendering strategy + get mask.'''
        body_rgb, body_mask = self.renderer(
            verts=verts,
            cam_t=cam_t
        )
        return body_rgb, body_mask

    def vis_from_params(
            self,
            pose: Union[np.ndarray, torch.Tensor],
            shape: Union[np.ndarray, torch.Tensor],
            glob_orient: Union[np.ndarray, torch.Tensor] = None,
            cam_t: Optional[np.ndarray] = None
    ) -> Union[Tuple[np.ndarray, np.ndarray],
               Tuple[torch.Tensor, torch.Tensor]]:
        '''First run the SMPL body model to get verts and then render.'''
        if glob_orient is None:
            glob_orient = self.default_glob_orient

        pose, shape, glob_orient = to_tensors(
            arrays=[pose, shape, glob_orient]
        )
        body_vertices: np.ndarray = self.smpl_model(
            body_pose=pose,
            global_orient=glob_orient,
            betas=shape,
            pose2rot=False
        ).vertices

        return self.vis(
            verts=body_vertices,
            cam_t=cam_t
        )

    
class ClothedVisualizer(Visualizer2D):

    ''' Visualize a parametric model with clothing.
    
        Note that this class does not support texture mapping because
        then I would need psbody.Mesh object and I am only using the
        pytorch3d.structures.Meshes. I can't render psbody.Mesh using
        my current renderers. The ClothedVisualizer class is a subclass
        of Visualizer2D, therefore, it has a method for adding a
        background.
    '''

    def __init__(
            self, 
            gender: str,
            upper_class: str,
            lower_class: str,
            device: str,
            backgrounds_dir_path: str = None,
            img_wh=256
        ) -> None:
        ''' Initialize the visualizer using parameter specification.'''
        super().__init__(backgrounds_dir_path)

        _garment_classes = GarmentClasses(
            upper_class, 
            lower_class
        )
        self.parametric_model = ParametricModel(
            gender=gender, 
            garment_classes=_garment_classes
        )
        self.device = device
        self.renderer = ClothedRenderer(
            device=self.device,
            batch_size=1
        )
        self.img_wh = img_wh

    def vis(self,
            smpl_output_dict: SMPL4GarmentOutput,
            cam_t: Optional[np.ndarray] = None
        ) -> Tuple[np.ndarray, np.ndarray]:
        ''' Visualize clothed mesh(es), given SMPL4GarmentOutput info.'''
        rgb_img, seg_maps = self.renderer(
            smpl_output_dict,
            self.parametric_model.garment_classes,
            cam_t
        )
        return rgb_img, seg_maps

    def vis_from_params(
            self,
            pose: np.ndarray, 
            shape: np.ndarray, 
            style_vector: np.ndarray,
            cam_t: Optional[np.ndarray] = None
        ) -> Tuple[np.ndarray, np.ndarray]:
        ''' Visualize clothed mesh(es).
        
            First, the parametric model is ran to obtain the verts.
            Then, the renderer renders the clothed meshes. The method
            returns an RGB rendered image (without background) and the
            mask of the person's silhouette. Note that the method 
            always expects Numpy arrays because visualizing TailorNet 
            will not be required in training loop, for now.
        '''
        pose, shape, style_vector = to_tensors(
            arrays=[pose, shape, style_vector]
        )
        smpl_output_dict = self.parametric_model.run(
            pose=pose,
            shape=shape,
            style_vector=style_vector
        )
        rgb_img, seg_maps = self.renderer(
            smpl_output_dict,
            self.parametric_model.garment_classes,
            cam_t
        )
        return rgb_img, seg_maps


class ClothedVisualizer3D(Visualizer3D):

    ''' Visualize 3D clothed parametric mesh (texture-only).
    
        Note that this class can only produce textured meshes and not simply
        single-colored meshes because pytorch3d.structures.Meshes are required
        for single-color meshes, but then you can't easily properly save them
        as obj + texture image. On the other hand, psbody.Mesh can be saved,
        but the current functionalities do not allow creating texture maps for
        single-color meshes. This class is intended to be used in a way that
        it produces 3D meshes at the end which are then stored to the disk.
    '''

    def __init__(
            self,
            gender: str,
            upper_class: str,
            lower_class: str,
        ) -> None:
        '''Prepare texture mesh manager and parametric model.'''
        self.gender = gender
        self.garment_classes = GarmentClasses(
            upper_class, 
            lower_class
        )
        self.textured_mesh_manager = TexturedGarmentsMeshManager(save_maps_to_disk=True)
        self.parametric_model = ParametricModel(
            gender=gender, 
            garment_classes=self.garment_classes
        )

    def vis(self,
            smpl_output_dict: SMPL4GarmentOutput
        ) -> Tuple[Mesh, Mesh, Mesh]:
        ''' Visualize clothed mesh(es), given SMPL4GarmentOutput info.'''
        meshes = self.textured_mesh_manager.create_meshes(
            smpl_output_dict=smpl_output_dict
        )
        meshes = self.textured_mesh_manager.texture_meshes(
            meshes=meshes,
            garment_classes=self.garment_classes
        )
        return (
            meshes[0],  # body mesh
            meshes[1],  # upper garment mesh
            meshes[2]   # lower garment mesh
        )

    def vis_from_params(
            self,
            pose: np.ndarray, 
            shape: np.ndarray, 
            style_vector: np.ndarray,
        ) -> Tuple[Mesh, Mesh, Mesh]:
        ''' Visualize clothed mesh(es), given pose, shape, and style params.'''
        pose, shape, style_vector = to_tensors(
            arrays=[pose, shape, style_vector]
        )
        smpl_output_dict = self.parametric_model.run(
            pose=pose,
            shape=shape,
            style_vector=style_vector
        )
        meshes = self.vis(smpl_output_dict)
        return (
            meshes[0],  # body mesh
            meshes[1],  # upper garment mesh
            meshes[2]   # lower garment mesh
        )

    def save_vis(
            self,
            meshes: Tuple[Mesh, Mesh, Mesh],
            rel_path: str
    ) -> None:
        '''Save the visualization of 3D meshes to disk (only way to observe).'''
        self.textured_mesh_manager.save_meshes(
            meshes=meshes,
            rel_path=rel_path
        )
