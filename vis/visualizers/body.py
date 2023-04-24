from typing import Union, Tuple, Optional
import torch
import numpy as np
from PIL import (
    Image, 
    ImageOps
)

from models.smpl_official import (
    easy_create_smpl_model,
    SMPL
)
from rendering.body import BodyRenderer
from utils.convert_arrays import to_numpy
from vis.visualizers.common import Visualizer2D


class BodyVisualizer(Visualizer2D):

    def __init__(
            self, 
            device: str,
            gender: Optional[str] = None,
            smpl_model: Optional[SMPL] = None,
            backgrounds_dir_path: str = None
        ) -> None:
        """
        Initialize BodyVisualizer class.

        If smpl_model is provided, then gender argument is ignored.
        If smpl_model is not provided, then gender argument, if
        provided, is used to create an SMPL model. Finally, if both
        smpl_model and gender are not provided, the SMPL model is None
        and is expected that will be created in the visualization method.
        """
        super().__init__(backgrounds_dir_path)

        self.device = device
        self.renderer = BodyRenderer(
            device=self.device
        )

        self.smpl_model = None
        model_description = 'None'
        if smpl_model is not None:
            self.smpl_model = smpl_model
            model_description = f'predefined_{smpl_model.gender}'
        else:
            if gender is not None:
                self.smpl_model = easy_create_smpl_model(
                    gender=gender,
                    device=device
                )
                model_description = f'new_{gender}'
        print(f'[BodyVisualizer] Using {model_description} SMPL.')

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
            cam_t: Optional[Union[np.ndarray, torch.Tensor]] = None,
            gender: Optional[str] = None
    ) -> Union[Tuple[np.ndarray, np.ndarray],
               Tuple[torch.Tensor, torch.Tensor]]:
        '''First run the SMPL body model to get verts and then render.'''
        if self.smpl_model is not None:
            smpl_model = self.smpl_model
        else:
            if gender is None:
                print('WARNING: Gender unspecified, setting male.')
                gender = 'male'
            smpl_model = easy_create_smpl_model(
                gender=gender,
                device=self.device
            )

        body_verts: torch.Tensor = self.create_body(
            pose=pose,
            shape=shape,
            smpl_model=smpl_model
        ).vertices

        body_rgb, body_mask = self.vis(
            verts=body_verts,
            cam_t=cam_t
        )
        if type(pose) == np.ndarray:
            body_rgb, body_mask = to_numpy(body_rgb, body_mask)

        return body_rgb, body_mask
    
    def save_vis(
            self,
            rgb_img: np.ndarray,
            save_path: str
    ) -> None:
        rgb_img = ImageOps.flip(Image.fromarray(rgb_img.astype(np.uint8)))
        rgb_img.save(save_path)
        print(f'Saved body image: {save_path}...')
