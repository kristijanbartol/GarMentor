import os
import random
from typing import Dict, List, Tuple
import numpy as np

_DEVICE = 'CPU' if os.environ['PYOPENGL_PLATFORM'] == 'osmesa' \
    else 'GPU'
print(f'NOTE: Using {_DEVICE} for rendering...')
import trimesh
import pyrender

from utils.exceptions import NakedGarmentorException
from utils.colors import NoColors, BodyColors, GarmentColors, A


class TrimeshRenderer:
    
    def __init__(self, 
                 img_wh: int = 256,
                 intensity: float = 3.0):
        
        self.img_wh = img_wh

        self.body_material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode='OPAQUE',
            baseColorFactor=self.COLOR_PALLETE['grey'])
        self.garment_material_pallete = self._get_garment_material_pallete()
        
        self.light = pyrender.SpotLight(
            color=np.ones(3), 
            intensity=intensity,
            innerConeAngle=np.pi/16.0,
            outerConeAngle=np.pi/6.0)
        
        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=img_wh,
            viewport_height=img_wh)
            
    def _get_garment_material_pallete(self) -> List:
        return [pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.0,
                alphaMode='OPAQUE',
                baseColorFactor=A(color)
            ) for color in GarmentColors]
        
    def _get_random_garment_material(self):
        return self.garment_material_pallete[
            random.randint(0, len(self.garment_material_pallete) - 1)
        ]
        
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
        
    def render(self, 
               body_mesh: trimesh.Trimesh, 
               garment_meshes: Dict[trimesh.Trimesh]
               ) -> Tuple[Dict[str, np.ndarray[np.float32]], 
                          Dict[str, np.ndarray[np.bool]]]:
        if garment_meshes['upper'] is None and garment_meshes['lower'] is None:
            raise NakedGarmentorException()
        
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
        s = np.sqrt(2)/2
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
            garment_materials[garment_part] = self._get_random_garment_material()
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
            materials=[self.body_material, 
                       garment_materials['upper'], 
                       garment_materials['lower']],
            camera=camera,
            camera_pose=camera_pose,
            light=self.light,
            light_pose=camera_pose
        )
        seg_maps = self._extract_segmentation_maps(render_imgs)
        
        return render_imgs, seg_maps
