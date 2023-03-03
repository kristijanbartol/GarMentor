from typing import List, Dict, Tuple
from os.path import join
from glob import glob
from psbody.mesh import Mesh
from PIL import Image
import numpy as np
import os
from tqdm import tqdm

from data.const import (
    MGN_CLASSES,
    GARMENT_CLASSES,
    MGN_DATASET,
    UV_MAPS_PATH
)
from data.mesh_managers.common import (
    MeshManager,
    create_psbody_meshes
)
from utils.garment_classes import GarmentClasses

from tailornet_for_garmentor.models.smpl4garment_utils import SMPL4GarmentOutput


class TexturedGarmentsMeshManager(MeshManager):

    def __init__(
            self, 
            save_maps_to_disk: bool = False
        ) -> None:
        self.garment_dict = {}
        for garment in MGN_CLASSES:
            self.garment_dict[garment] = glob(join(MGN_DATASET, '*', garment + '.obj'))
        
        self.save_maps_to_disk = save_maps_to_disk
        self._prepare_texture_maps()

    def _prepare_texture_maps(self) -> None:
        if not os.path.exists(UV_MAPS_PATH) and self.save_maps_to_disk:
            os.makedirs(UV_MAPS_PATH)
        
        self.vt_map = None
        self.garment_ft_maps = []
        self.body_ft_map = None
        self.texture_images = []

        self._create_vt_map(self.garment_dict, UV_MAPS_PATH)
        self._create_ft_maps(self.garment_dict, UV_MAPS_PATH)
        self._create_body_ft_map(UV_MAPS_PATH)

        self.texture_dirpaths = [
            os.path.join(MGN_DATASET, x) for x in os.listdir(MGN_DATASET)
        ]

        _all_textures = sorted(glob(f'{MGN_DATASET}/*/multi_tex.jpg'))
        _all_segmentations = sorted(glob(f'{MGN_DATASET}/*/segmentation.png'))
        self._generate_body_texture_from_texture(
            texture_pth=_all_textures,
            segmentation_pth=_all_segmentations
        )

    def _create_vt_map(self, garment_dict: dict, save_to: str):
        '''
        Create vt map for all the garments. Creates one vt map because
        all the garments share the same vt-s
        Arguments:
            save_to: save created vt map to this path
        Returns:
            saves general_vt.npy to disk
        '''
        random_mesh_name = garment_dict['TShirtNoCoat'][0]
        loaded_mesh =  Mesh(filename=random_mesh_name)

        vt_map = loaded_mesh.vt
        if self.save_maps_to_disk:
            np.save(f'{save_to}/general_vt.npy', vt_map)
        else:
            self.vt_map = vt_map

    def _create_ft_maps(
            self, 
            garment_dict: Dict[str, str], 
            save_to: str
        ) -> None:
        '''
        Create ft map for specific garments saved as garment_ft.npy
        Arguments:
            garment_name: one name from the garment_classes
            save_to: save created vt map to this path
        Returns:
            saves garment_ft.npy to disk
        '''
        for garment_name in MGN_CLASSES:
            random_mesh_name = garment_dict[garment_name][0]
            loaded_mesh =  Mesh(filename=random_mesh_name)

            index = MGN_CLASSES.index(garment_name)
            garment_tailornet_name = GARMENT_CLASSES[index]

            ft_map = loaded_mesh.ft
            if self.save_maps_to_disk:
                np.save(f'{save_to}/{garment_tailornet_name}_ft.npy', ft_map)
            else:
                self.garment_ft_maps.append(ft_map)

    def _create_body_ft_map(
            self, 
            save_to: str
        ) -> None:
        '''
        Create ft maps for the body using smpl fits from smpl_registered.obj
        All the maps are the same across the examples so only using one
        Arguments:
            save_to: save created vt,ft map to this path
        Returns:
            saves body_ft.npy to disk
        '''
        
        all_bodies = glob(join(MGN_DATASET, '*', 'smpl_registered.obj'))

        random_mesh_name = all_bodies[0]
        loaded_mesh =  Mesh(filename=random_mesh_name)

        body_ft_map = loaded_mesh.ft
        if self.save_maps_to_disk:
            np.save(f'{save_to}/body_ft.npy', body_ft_map)
        else:
            self.body_ft_map = body_ft_map
        
    def _create_texture_images(
            self, 
            texture_paths: List[str], 
            segmentation_paths: List[str], 
            skin_color_pixel: List[float]
        ) -> None:
        '''
        From a mgn texture multi_tex.jpg, set all pixels that are not related to the 
        hair, shoes and skin to a skin color chosen from the hand pixel - set ad hoc to the 
        pixel 1430,1920 -- right bottom corner where hands are in texture
        Arguments:
            texture_pth: path to texture from a mgn dataset
            segmentation_pth: path to segmentation of the texture from a mgn dataset
            skin_color_pixel: pixel location from texture map that correspondns to a skin color
                            value
            show_fig: show the figure in matplotlib
        Returns:
            saves a body_tex.jpg to disk in same folder as texture_pth
        '''
        for i in tqdm(range(len(texture_paths))):
            save_to = texture_paths[i].split('/multi_tex.jpg')[0]
            background_color = [0,0,0] # black

            # load texture and segmentation
            texture_img = Image.open(texture_paths[i])
            texture_img_array = np.array(texture_img)
            segmentation_img = Image.open(segmentation_paths[i])

            H,W,_ = texture_img_array.shape
            if skin_color_pixel is None:
                # sample a few pre-defined points around the nose for the skin color 
                pixel_coords = [(620,470),(600,440),(640,440),(620,410)]
                skin_colors = [texture_img.getpixel(px_coords) for px_coords in pixel_coords]
                skin_colors_r_mean = np.mean([rgb[0] for rgb in skin_colors])
                skin_colors_g_mean = np.mean([rgb[1] for rgb in skin_colors])
                skin_colors_b_mean = np.mean([rgb[2] for rgb in skin_colors])
                skin_color = (skin_colors_r_mean,skin_colors_g_mean,skin_colors_b_mean)
            else:
                skin_color = texture_img.getpixel(skin_color_pixel)
                
            skin_color = np.array(skin_color).reshape(-1,3)

            # resize segmentation to texture size
            resized_segmentation_for_body_texture = segmentation_img.resize((2048,2048))
            resized_segmentation_for_body_texture_array = np.array(resized_segmentation_for_body_texture)

            # create mask to set everything to background segmentation color (black) 
            # except hair, shoes and skin  -- these have in common that their sum is = to 255
            mask = np.ones((H,W))
            mask[(np.sum(resized_segmentation_for_body_texture_array,axis=2) == 255)] = 0 # where sum is 255
            # set it to the background color
            resized_segmentation_for_body_texture_array[mask == 1] = background_color

            # use segmentation map to set everything to skin color on the texture map
            #  except the skin, hair and shoes
            background_pixels = np.sum(resized_segmentation_for_body_texture_array,axis=2) == 0
            texture_img_array[background_pixels] = skin_color

            texture_image = Image.fromarray(texture_img_array)
            if self.save_maps_to_disk:
                texture_image.save(f'{save_to}/body_tex.jpg')
            else:
                self.texture_images.append(texture_image)

    def _get_random_texture_paths(self) -> Tuple[str, str, str]:
        '''
        Get a random texture dirpath from the MGN dataset.
        '''
        texture_dirpath = self.texture_dirpaths[
            np.random.randint(0,len(self.texture_dirpaths))
        ]
        return (
            f'{texture_dirpath}/body_tex.jpg',
            f'{texture_dirpath}/multi_tex.jpg',
            f'{texture_dirpath}/multi_tex.jpg'
        )
    
    @staticmethod
    def create_meshes(smpl_output_dict: SMPL4GarmentOutput
                       ) -> Tuple[Mesh, Mesh, Mesh]:
        return create_psbody_meshes(smpl_output_dict)

    def texture_meshes(
            self,
            smpl_output_dict: SMPL4GarmentOutput,
            garment_classes: GarmentClasses
        ) -> List[Mesh]:
        '''
        Texture the [body, upper garment, lower garment] meshes from list meshes
        with textures from texture_paths -- mesh can be None if no mesh is available
        
        Parameters:
            smpl_output_dict: SMPL4GarmentOutput class with body, upper, and lower garment
                              mesh information
            garment_classes: a GarmentClasses object with the garment classes for the upper 
                    and lower garment
        Returns:
            textured_meshes: list of 3 textured meshes from the input
        '''
        meshes = self._create_meshes(smpl_output_dict)

        # Load pre-defined uv maps and random texture paths.
        general_vt = np.load(f'{UV_MAPS_PATH}/general_vt.npy')
        texture_paths = self._get_random_texture_paths()

        mesh_labels_list = [
            'body',
            garment_classes.upper_class,
            garment_classes.lower_class
        ]
        for mesh_idx, mesh_label in enumerate(mesh_labels_list):
            if meshes[mesh_idx] is not None:
                meshes[mesh_idx].ft = np.load(f'{UV_MAPS_PATH}/{mesh_label}_ft.npy')
                meshes[mesh_idx].vt = general_vt
                meshes[mesh_idx].set_texture_image(texture_paths[mesh_idx])

        return meshes

    def postprocess_meshes(
            self,
            rel_path: str
    ) -> None:
        ''' Modify obj files to support material in Blender.

            By default, psbody mesh's write_obj() function does not utilize
            the `usemtl` keyword, which results in blender not showing the
            material.
        '''
        for mesh_type in ["body", "upper", "lower"]:
            obj_fpath = f"{rel_path}-{mesh_type}.obj"
            content = ""
            with open(obj_fpath, "r") as obj_file:
                first_face = True
                usemtl_encountered = False
                for line in obj_file:
                    if not usemtl_encountered and \
                        line.strip().split(' ')[0] == 'usemtl':
                        usemtl_encountered = True
                    if first_face and line.strip().split(' ')[0] == 'f':
                        first_face = False
                        if not usemtl_encountered:
                            content += \
                                f"usemtl {rel_path}-{mesh_type}\n"
                            usemtl_encountered = True
                    content += line
            with open(obj_fpath, "w") as obj_file:
                obj_file.write(content)


if __name__ == '__main__':
    TexturedGarmentsMeshManager(save_maps_to_disk=True)
