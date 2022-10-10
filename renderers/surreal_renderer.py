import os
import random
from typing import Dict, List, Tuple
import numpy as np

_DEVICE = 'CPU' if os.environ['PYOPENGL_PLATFORM'] == 'osmesa' \
    else 'GPU'
print(f'NOTE: Using {_DEVICE} for rendering...')
import trimesh
import pyrender

from tailornet_for_garmentor.models.smpl4garment_utils import SMPL4GarmentOutput

from utils.colors import Colors, NoColors, BodyColors, GarmentColors, A
from utils.garment_classes import GarmentClasses


class SurrealRenderer:
    
    def __init__(self, 
                 img_wh: int = 256,
                 intensity: float = 3.0):
        
        self.img_wh = img_wh

        self.body_materials = self._color_to_material_pallete(BodyColors)
        self.garment_materials = self._color_to_material_pallete(GarmentColors)
        
        self.light = pyrender.SpotLight(
            color=np.ones(3), 
            intensity=intensity,
            innerConeAngle=np.pi/16.0,
            outerConeAngle=np.pi/6.0)
        
        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=img_wh,
            viewport_height=img_wh)
            
    @staticmethod
    def _color_to_material_pallete(pallete: Colors
                                   ) -> List[pyrender.Material]:
        return [pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.0,
                alphaMode='OPAQUE',
                baseColorFactor=A(color)
            ) for color in pallete]
        
    @staticmethod
    def _select_random_material(pallete: pyrender.Material
                                ) -> pyrender.Material:
        return pallete[random.randint(0, len(pallete) - 1)]
        
    def _render_scene(self, 
                      meshes: List[trimesh.Trimesh], 
                      materials: List[pyrender.Material],
                      camera: pyrender.camera.Camera, 
                      camera_pose: np.ndarray, 
                      light: pyrender.light.Light,
                      light_pose: np.ndarray) -> np.ndarray:
        scene = pyrender.Scene()
        for mesh_idx, mesh in enumerate(meshes):
            if mesh is not None:
                pyrender_mesh = pyrender.Mesh.from_trimesh(
                    mesh,
                    material=materials[mesh_idx]
                )
                scene.add(pyrender_mesh, 'mesh')
        scene.add(camera, pose=camera_pose)
        scene.add(light, pose=light_pose)
        
        self.renderer.render(scene)[0]
        
    def _extract_segmentation_maps(self, 
                                   render_imgs: Dict[str, np.ndarray]
                                   ) -> Dict[str, np.ndarray]:
        seg_maps = {}
        for img_key in render_imgs:
            seg_maps[img_key] = np.zeros((self.img_wh, self.img_wh), 
                                          dtype=np.bool)
            img = render_imgs[img_key]
            background_pixels = np.all(img == NoColors.WHITE, axis=2)
            seg_maps[img_key][background_pixels] = 1
        return seg_maps
    
    @staticmethod
    def _smpl_to_trimesh(smpl_output_dict: Dict[str, SMPL4GarmentOutput]
                         ) -> Tuple[trimesh.Trimesh, Dict[str, trimesh.Trimesh]]:
        body_mesh = trimesh.Trimesh(
            vertices=smpl_output_dict['upper'].body_verts,
            faces=smpl_output_dict['upper'].body_faces
        )
        garment_meshes = {}
        for garment_part in ['upper', 'lower']:
            garment_meshes[garment_part] = trimesh.Trimesh(
                vertices=smpl_output_dict[garment_part].garment_verts,
                faces=smpl_output_dict[garment_part].garment_faces
            )
        return body_mesh, garment_meshes
    
    def _seg_maps_to_features(self,
                               seg_maps: Dict[str, np.ndarray], 
                               garment_classes: GarmentClasses) -> np.ndarray:
        seg_maps_features = np.zeros(
            (GarmentClasses.NUM_CLASSES + 1, self.img_wh, self.img_wh), 
            dtype=np.bool)
        for garment_part, label in garment_classes.labels.items():
            if label is not None:
                seg_maps_features[label] = seg_maps[garment_part]
        seg_maps_features[-1] = seg_maps['whole']
        return seg_maps_features
        
    def render(self, 
               smpl_output_dict: Dict[str, SMPL4GarmentOutput],
               garment_classes: GarmentClasses
               ) -> Tuple[np.ndarray[np.float32], np.ndarray[np.bool]]:
        
        body_mesh, garment_meshes = self._smpl_to_trimesh(smpl_output_dict)
        
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
        s = np.sqrt(2) / 2
        camera_pose = np.array([
            [0.0, -s,   s,   1.3],
            [1.0,  0.0, 0.0, -0.2],
            [0.0,  s,   s,   1.35],
            [0.0,  0.0, 0.0, 1.0],
        ])
        
        render_imgs = {}
        garment_materials = {}
        for garment_part in ['upper', 'lower']:
            garment_mesh = garment_meshes[garment_part]
            garment_materials[garment_part] = self._select_random_material(
                self.garment_materials)
            render_imgs[garment_part] = self._render_scene(
                meshes=[garment_mesh],
                materials=[garment_materials[-1]],
                camera=camera,
                camera_pose=camera_pose,
                light=self.light,       # TODO: Augment light.
                light_pose=camera_pose
            )
                
        render_imgs['whole'] = self._render_scene(
            meshes=[body_mesh, 
                    garment_meshes['upper'], 
                    garment_meshes['lower']],
            materials=[self._select_random_material(self.body_materials), 
                       garment_materials['upper'], 
                       garment_materials['lower']],
            camera=camera,
            camera_pose=camera_pose,
            light=self.light,
            light_pose=camera_pose
        )
        seg_maps = self._extract_segmentation_maps(render_imgs)
        seg_maps = self._seg_maps_to_features(seg_maps, garment_classes)
        
        return render_imgs['whole'], seg_maps
