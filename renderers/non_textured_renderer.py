import torch
import torch.nn as nn
import numpy as np
from scipy.io import loadmat

from configs import paths

import matplotlib.pyplot as plt

# Data structures and functions for rendering
from pytorch3d.structures import Meshes, join_meshes_as_scene
from pytorch3d.renderer import (
    FoVOrthographicCameras,
    PointLights,
    RasterizationSettings,
    MeshRasterizer,
    HardPhongShader,
    BlendParams,
    Textures)


class NonTexturedRenderer(nn.Module):
    def __init__(self,
                 img_wh=256,
                 cam_t=None,
                 cam_R=None,
                 orthographic_scale=0.9,
                 blur_radius=0.0,
                 faces_per_pixel=1,
                 bin_size=None,
                 max_faces_per_bin=None,
                 perspective_correct=False,
                 cull_backfaces=False,
                 clip_barycentric_coords=None,
                 light_t=((0.0, 0.0, -2.0),),
                 light_ambient_color=((0.5, 0.5, 0.5),),
                 light_diffuse_color=((0.3, 0.3, 0.3),),
                 light_specular_color=((0.2, 0.2, 0.2),),
                 background_color=(0.0, 0.0, 0.0)):
        """
        :device: CPU or CUDA
        :batch_size: Batch size
        :body_faces: Body mesh faces indices (F, 3)
        :garment_faces: Garment faces indices (F, 3)
        :num_body_verts: Body mesh vertices V=27554
        :num_garment_verts: Garment mesh vertices V=7702
        :param img_wh: Size of rendered image.
        :param blur_radius: Float distance in the range [0, 2] used to expand the face
            bounding boxes for rasterization. Setting blur radius
            results in blurred edges around the shape instead of a
            hard boundary.
            Set to 0 (no blur) if rendering for visualisation purposes.
        :param faces_per_pixel: Number of faces to save per pixel, returning
            the nearest faces_per_pixel points along the z-axis.
            Set to 1 if rendering for visualisation purposes.
        :param bin_size: Size of bins to use for coarse-to-fine rasterization (i.e
            breaking image into tiles with size=bin_size before rasterising?).
            Setting bin_size=0 uses naive rasterization; setting bin_size=None
            attempts to set it heuristically based on the shape of the input (i.e. image_size).
            This should not affect the output, but can affect the speed of the forward pass.
            Heuristic based formula maps image_size -> bin_size as follows:
                image_size < 64 -> 8
                16 < image_size < 256 -> 16
                256 < image_size < 512 -> 32
                512 < image_size < 1024 -> 64
                1024 < image_size < 2048 -> 128
        :param max_faces_per_bin: Only applicable when using coarse-to-fine rasterization
            (bin_size > 0); this is the maxiumum number of faces allowed within each
            bin. If more than this many faces actually fall into a bin, an error
            will be raised. This should not affect the output values, but can affect
            the memory usage in the forward pass.
            Heuristic used if None value given:
                max_faces_per_bin = int(max(10000, meshes._F / 5))
        :param perspective_correct: Bool, Whether to apply perspective correction when computing
            barycentric coordinates for pixels.
        :param cull_backfaces: Bool, Whether to only rasterize mesh faces which are
            visible to the camera.  This assumes that vertices of
            front-facing triangles are ordered in an anti-clockwise
            fashion, and triangles that face away from the camera are
            in a clockwise order relative to the current view
            direction. NOTE: This will only work if the mesh faces are
            consistently defined with counter-clockwise ordering when
            viewed from the outside.
        :param clip_barycentric_coords: By default, turn on clip_barycentric_coords if blur_radius > 0.
        When blur_radius > 0, a face can be matched to a pixel that is outside the face,
        resulting in negative barycentric coordinates.
        """
        super().__init__()
        self.img_wh = img_wh
        self.device = 'cpu'

        self.batch_size = 1
        self.garment_faces_offset = self.garment_faces + self.body_faces.max()

        # Cameras - pre-defined here but can be specified in forward pass if cameras will vary (e.g. random cameras)
        if cam_R is None:
            # Rotating 180° about z-axis to make pytorch3d camera convention same as what I've been using so far in my perspective_project_torch/NMR/pyrender
            # (Actually pyrender also has a rotation defined in the renderer to make it same as NMR.)
            cam_R = torch.tensor([[-1., 0., 0.],
                                  [0., 1., 0.],
                                  [0., 0., -1.]], device=self.device).float()
            cam_R = cam_R[None, :, :].expand(self.batch_size, -1, -1)
        if cam_t is None:
            cam_t = torch.tensor([0., 0.2, 2.5]).float().to(self.device)[None, :].expand(self.batch_size, -1)
        # Pytorch3D camera is rotated 180° about z-axis to match my perspective_project_torch/NMR's projection convention.
        # So, need to also rotate the given camera translation (implemented below as elementwise-mul).
        cam_t = cam_t * torch.tensor([-1., -1., 1.], device=self.device).float()

        self.cameras = FoVOrthographicCameras(device=self.device,
                                            R=cam_R,
                                            T=cam_t,
                                            scale_xyz=((
                                                orthographic_scale,
                                                orthographic_scale,
                                                1.0),))

        # Lights for textured RGB render - pre-defined here but can be specified in forward pass if lights will vary (e.g. random cameras)
        self.lights_rgb_render = PointLights(device=self.device,
                                                location=light_t,
                                                ambient_color=light_ambient_color,
                                                diffuse_color=light_diffuse_color,
                                                specular_color=light_specular_color)

        # Rasterizer
        raster_settings = RasterizationSettings(image_size=img_wh,
                                                blur_radius=blur_radius,
                                                faces_per_pixel=faces_per_pixel,
                                                bin_size=bin_size,
                                                max_faces_per_bin=max_faces_per_bin,
                                                perspective_correct=perspective_correct,
                                                cull_backfaces=cull_backfaces,
                                                clip_barycentric_coords=clip_barycentric_coords)
        self.rasterizer = MeshRasterizer(cameras=self.cameras, raster_settings=raster_settings)  # Specify camera in forward pass

        # Shader for textured RGB output and IUV output
        blend_params = BlendParams(background_color=background_color)
        self.rgb_shader = HardPhongShader(device=self.device, cameras=self.cameras,
                                            lights=self.lights_rgb_render, blend_params=blend_params)

        self.to(self.device)

    def to(self, device):
        # Rasterizer and shader have submodules which are not of type nn.Module
        self.rasterizer.to(device)
        self.rgb_shader.to(device)

    '''
    def forward(self, body_vertices, garment_vertices, cam_t=None, orthographic_scale=None, 
                lights_rgb_settings=None):
        """
        Render a batch of textured RGB images and IUV images from a batch of meshes.

        Fragments output from rasterizer:
        pix_to_face:
          LongTensor of shape (B, image_size, image_size, faces_per_pixel)
          specifying the indices of the faces (in the packed faces) which overlap each pixel in the image.
        zbuf:
          FloatTensor of shape (B, image_size, image_size, faces_per_pixel)
          giving the z-coordinates of the nearest faces at each pixel in world coordinates, sorted in ascending z-order.
        bary_coords:
          FloatTensor of shape (B, image_size, image_size, faces_per_pixel, 3)
          giving the barycentric coordinates in NDC units of the nearest faces at each pixel, sorted in ascending z-order.
        pix_dists:
          FloatTensor of shape (B, image_size, image_size, faces_per_pixel)
          giving the signed Euclidean distance (in NDC units) in the x/y plane of each point closest to the pixel.

        :param body_vertices: (B, N, 3), N=27554
        :param garment_vertices: (B, N, 3), N=7702
        :param cam_t: (B, 3)
        :param orthographic_scale: (B, 2)
        :param lights_rgb_settings: dict of lighting settings with location, ambient_color, diffuse_color and specular_color.
        :returns rgb_images: (B, img_wh, img_wh, 3) rendered bodies and garments
        :returns iuv_images: (B, img_wh, img_wh, 3) silhouettes
        :returns depth_images: (B, img_wh, img_wh)
        """
        if cam_t is not None:
            # Pytorch3D camera is rotated 180° about z-axis to match my perspective_project_torch/NMR's projection convention.
            # So, need to also rotate the given camera translation (implemented below as elementwise-mul).
            self.cameras.T = cam_t * torch.tensor([-1., -1., 1.], device=cam_t.device).float()
        if orthographic_scale is not None and self.projection_type == 'orthographic':
            self.cameras.focal_length = orthographic_scale * (self.img_wh / 2.0)

        if lights_rgb_settings is not None:
            self.lights_rgb_render.location = lights_rgb_settings['location']
            self.lights_rgb_render.ambient_color = lights_rgb_settings['ambient_color']
            self.lights_rgb_render.diffuse_color = lights_rgb_settings['diffuse_color']
            self.lights_rgb_render.specular_color = lights_rgb_settings['specular_color']
        
        union_textures = Textures(
            verts_rgb=torch.cat([self.body_verts_rgb_colors, self.garment_verts_rgb_colors], dim=1))

        union_meshes = Meshes(
            verts=torch.cat([body_vertices, garment_vertices], dim=1),
            faces=torch.cat([self.body_faces, self.garment_faces_offset], dim=1),
            textures=union_textures
        )

        # Rasterize
        fragments = self.rasterizer(union_meshes, cameras=self.cameras)
        zbuffers = fragments.zbuf[:, :, :, 0]

        # Render RGB output.
        output = {}
        rgb_images = self.rgb_shader(fragments, union_meshes, lights=self.lights_rgb_render)[:, :, :, :3]
        output['rgb_images'] = torch.clamp(rgb_images, max=1.0)
        output['iuv_images'] = torch.where(output['rgb_images'] > 0., 1., 0.)

        # Get depth image
        output['depth_images'] = zbuffers

        return output
    '''

    def forward(self, body_verts, body_faces, upper_garment_verts, 
                upper_garment_faces, lower_garment_verts, lower_garment_faces, 
                cam_t=None, orthographic_scale=None, 
                lights_rgb_settings=None):

        if cam_t is not None:
            # Pytorch3D camera is rotated 180° about z-axis to match my perspective_project_torch/NMR's projection convention.
            # So, need to also rotate the given camera translation (implemented below as elementwise-mul).
            self.cameras.T = cam_t * torch.tensor([-1., -1., 1.], device=cam_t.device).float()
        if orthographic_scale is not None and self.projection_type == 'orthographic':
            self.cameras.focal_length = orthographic_scale * (self.img_wh / 2.0)

        if lights_rgb_settings is not None:
            self.lights_rgb_render.location = lights_rgb_settings['location']
            self.lights_rgb_render.ambient_color = lights_rgb_settings['ambient_color']
            self.lights_rgb_render.diffuse_color = lights_rgb_settings['diffuse_color']
            self.lights_rgb_render.specular_color = lights_rgb_settings['specular_color']

        num_body_verts = body_verts.shape[0]
        num_upper_garment_verts = upper_garment_verts.shape[0] if upper_garment_verts is not None else 0
        num_lower_garment_verts = lower_garment_verts.shape[0] if lower_garment_verts is not None else 0

        total_num_verts = num_body_verts + num_upper_garment_verts + num_lower_garment_verts

        if not torch.is_tensor(body_verts):
            body_verts = torch.from_numpy(body_verts).unsqueeze(0).to(self.device)
            body_faces = torch.from_numpy(body_faces).unsqueeze(0).to(self.device)
            if upper_garment_verts is not None:
                upper_garment_verts = torch.from_numpy(upper_garment_verts).unsqueeze(0).to(self.device)
                upper_garment_faces = torch.from_numpy(upper_garment_faces).unsqueeze(0).to(self.device)
            if lower_garment_verts is not None:
                lower_garment_verts = torch.from_numpy(lower_garment_verts).unsqueeze(0).to(self.device)
                lower_garment_faces = torch.from_numpy(lower_garment_faces).unsqueeze(0).to(self.device)

        self.garment_faces_offset = self.garment_faces + self.body_faces.max()

        union_verts = body_verts
        union_faces = body_faces
        faces_offset = body_faces.max()
        if upper_garment_verts is not None:
            union_verts = torch.cat([union_verts, upper_garment_verts], dim=1)
            union_faces = torch.cat([union_faces, upper_garment_faces + faces_offset])
            faces_offset += upper_garment_faces.max()
        if lower_garment_verts is not None:
            union_verts = torch.cat([union_verts, lower_garment_verts], dim=1)
            union_faces = torch.cat([union_faces, lower_garment_faces + faces_offset])

        union_verts_white_color = torch.tensor([1., 1., 1.], device=self.device).repeat(1, total_num_verts, 1)
        upper_verts_white_color = torch.tensor([1., 1., 1.], device=self.device).repeat(1, num_upper_garment_verts, 1)
        lower_verts_white_color = torch.tensor([1., 1., 1.], device=self.device).repeat(1, num_lower_garment_verts, 1)
        
        union_textures = Textures(verts_rgb=union_verts_white_color)

        union_mesh = Meshes(
            verts=union_verts,
            faces=union_faces,
            textures=union_textures
        )

        upper_mesh = Meshes(
            verts=upper_garment_verts,
            faces=upper_garment_faces,
            textures=Textures(verts_rgb=upper_verts_white_color)
        )

        lower_mesh = Meshes(
            verts=upper_garment_verts,
            faces=upper_garment_faces,
            textures=Textures(verts_rgb=lower_verts_white_color)
        )

        seg_maps = []
        for mesh_to_render in [union_mesh, upper_mesh, lower_mesh]:
            # Rasterize
            fragments = self.rasterizer(mesh_to_render, cameras=self.cameras)

            # Render cloth segmentations.
            seg_maps.append(self.rgb_shader(fragments, mesh_to_render, lights=self.lights_rgb_render)[:, :, :, :3])

        return seg_maps
