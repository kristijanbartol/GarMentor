from typing import Dict, Tuple, Optional
import torch.nn as nn
import torch
import numpy as np

from pytorch3d.renderer import (
    FoVOrthographicCameras,
    PointLights,
    RasterizationSettings,
    MeshRasterizer,
    HardPhongShader,
    BlendParams)

from data.const import BODY_FACES


class Renderer(nn.Module):

    def __init__(
            self,
            device: str,
            batch_size: int,
            img_wh: int = 256,
            cam_t: Optional[torch.Tensor] = None,
            cam_R: Optional[torch.Tensor] = None,
            projection_type: str = 'perspective',
            orthographic_scale: float = 0.9,
            blur_radius: float = 0.0,
            faces_per_pixel: int = 1,
            bin_size: int = None,
            max_faces_per_bin: int = None,
            perspective_correct: bool = False,
            cull_backfaces: bool = False,
            clip_barycentric_coords: bool = None,
            light_t: Tuple[float] = ((0.0, 0.0, -2.0),),
            light_ambient_color: Tuple[float] = ((0.5, 0.5, 0.5),),
            light_diffuse_color: Tuple[float] = ((0.3, 0.3, 0.3),),
            light_specular_color: Tuple[float] = ((0.2, 0.2, 0.2),),
            background_color: Tuple[float] = (0.0, 0.0, 0.0)
        ) -> None:
        ''' The body renderer constructor.

            Parameters
            ----------
            :param img_wh: Size of rendered image.
            :param blur_radius: Float distance in the range [0, 2] 
                used to expand the face bounding boxes for rasterization. 
                Setting blur radius results in blurred edges around the 
                shape instead of a hard boundary.
                Set to 0 (no blur) if rendering for visualisation purposes.
            :param faces_per_pixel: Number of faces to save per pixel, 
                returning the nearest faces_per_pixel points along the z-axis.
                Set to 1 if rendering for visualisation purposes.
            :param bin_size: Size of bins to use for coarse-to-fine rasterization 
                (i.e breaking image into tiles with size=bin_size before 
                rasterising?). Setting bin_size=0 uses naive rasterization; 
                setting bin_size=None attempts to set it heuristically based on 
                the shape of the input (i.e. image_size). This should not affect 
                the output, but can affect the speed of the forward pass.
                Heuristic based formula maps image_size -> bin_size as follows:
                    image_size < 64 -> 8
                    16 < image_size < 256 -> 16
                    256 < image_size < 512 -> 32
                    512 < image_size < 1024 -> 64
                    1024 < image_size < 2048 -> 128
            :param max_faces_per_bin: Only applicable when using coarse-to-fine 
                rasterization (bin_size > 0); this is the maxiumum number of 
                faces allowed within each bin. If more than this many faces 
                actually fall into a bin, an error will be raised. This should 
                not affect the output values, but can affect the memory usage in 
                the forward pass. Heuristic used if None value given:
                    max_faces_per_bin = int(max(10000, meshes._F / 5))
            :param perspective_correct: Bool, Whether to apply perspective 
                correction when computing barycentric coordinates for pixels.
            :param cull_backfaces: Bool, Whether to only rasterize mesh faces 
                which are visible to the camera.  This assumes that vertices of
                front-facing triangles are ordered in an anti-clockwise fashion, 
                and triangles that face away from the camera are in a clockwise 
                order relative to the current view direction. 
                NOTE: This will only work if the mesh faces are consistently 
                defined with counter-clockwise ordering when viewed from the 
                outside.
            :param clip_barycentric_coords: By default, turn on 
                clip_barycentric_coords if blur_radius > 0. When blur_radius > 0, 
                a face can be matched to a pixel that is outside the face, 
                resulting in negative barycentric coordinates.
        '''
        super().__init__()
        self.img_wh = img_wh
        self.device = device

        # Cameras - pre-defined here but can be specified in forward 
        # pass if cameras will vary (e.g. random cameras).
        assert(projection_type in ['perspective', 'orthographic'], 
            print('Invalid projection type:', projection_type))
        print('\nRenderer projection type:', projection_type)
        self.projection_type = projection_type
        if cam_R is None:
            # Rotating 180° about z-axis to make pytorch3d camera convention same 
            # as what I've been using so far in my perspective_project_torch/NMR/pyrender.
            # (Actually pyrender also has a rotation defined in the 
            # renderer to make it same as NMR.)
            cam_R = torch.tensor([[-1., 0., 0.],
                                    [0., -1., 0.],
                                    [0., 0., 1.]], device=device).float()
            cam_R = cam_R[None, :, :].expand(batch_size, -1, -1)
        if cam_t is None:
            cam_t = torch.tensor([0., 0.2, 2.5]).float().to(device)
            cam_t = cam_t[None, :].expand(batch_size, -1)
        # Pytorch3D camera is rotated 180° about z-axis to match my 
        # perspective_project_torch/NMR's projection convention.
        # So, need to also rotate the given camera translation 
        # (implemented below as elementwise-mul).
        cam_t = cam_t * torch.tensor([-1., -1., 1.], device=cam_t.device).float()
        
        self.cameras = FoVOrthographicCameras(
            device=device,
            R=cam_R,
            T=cam_t,
            scale_xyz=((
                orthographic_scale,
                orthographic_scale,
                1.0),)
        )

        # Lights for textured RGB render - pre-defined here but can be specified in 
        # forward pass if lights will vary (e.g. random cameras).
        self.lights_rgb_render = PointLights(
            device=device,
            location=light_t,
            ambient_color=light_ambient_color,
            diffuse_color=light_diffuse_color,
            specular_color=light_specular_color
        )
        # Lights for IUV render - don't want lighting to affect the rendered image.
        self.lights_iuv_render = PointLights(
            device=device,
            ambient_color=[[1, 1, 1]],
            diffuse_color=[[0, 0, 0]],
            specular_color=[[0, 0, 0]]
        )

        # Rasterizer
        raster_settings = RasterizationSettings(
            image_size=img_wh,
            blur_radius=blur_radius,
            faces_per_pixel=faces_per_pixel,
            bin_size=bin_size,
            max_faces_per_bin=max_faces_per_bin,
            perspective_correct=perspective_correct,
            cull_backfaces=cull_backfaces,
            clip_barycentric_coords=clip_barycentric_coords
        )
        self.rasterizer = MeshRasterizer(
            cameras=self.cameras, 
            raster_settings=raster_settings
        )  # Specify camera in forward pass

        # Shader for textured RGB output and IUV output
        self.blend_params = BlendParams(background_color=background_color)
        self.rgb_shader = HardPhongShader(
            device=device, 
            cameras=self.cameras,
            lights=self.lights_rgb_render, 
            blend_params=self.blend_params
        )

        self.body_faces_numpy = BODY_FACES
        self.body_faces_torch = torch.from_numpy(BODY_FACES)
        self.to(device)

    def to(self, device):
        '''Move tensors to specified device.'''
        self.rasterizer.to(device)
        self.rgb_shader.to(device)
        self.body_faces_torch.to(device)

    def _update_lights_settings(
            self, 
            new_lights_settings: Dict
        ) -> None:
        '''Update lights settings by directly setting PointLights properties.'''
        self.lights_rgb_render.location = new_lights_settings['location']
        self.lights_rgb_render.ambient_color = new_lights_settings['ambient_color']
        self.lights_rgb_render.diffuse_color = new_lights_settings['diffuse_color']
        self.lights_rgb_render.specular_color = new_lights_settings['specular_color']

    def _process_optional_arguments(
            self,
            cam_t: Optional[np.ndarray] = None,
            orthographic_scale: Optional[float] = None,
            lights_rgb_settings: Optional[Dict[str, Tuple[float]]] = None
        ) -> None:
        '''Update camera translation, focal length, or lights settings.'''
        if cam_t is not None:
            cam_t = torch.from_numpy(cam_t).float().unsqueeze(0).to(self.device)
            self.cameras.T = cam_t * torch.tensor(
                [-1., -1., 1.], device=self.device).float()
        if orthographic_scale is not None and self.projection_type == 'orthographic':
            self.cameras.focal_length = orthographic_scale * (self.img_wh / 2.0)
        if lights_rgb_settings is not None and self.render_rgb:
            self._update_lights_settings(lights_rgb_settings)
