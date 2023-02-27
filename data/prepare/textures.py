
from typing import List, Dict
from os.path import join
from glob import glob
from psbody.mesh import Mesh
from PIL import Image
import numpy as np
import os
import sys
from tqdm import tqdm

sys.path.append('/garmentor/')

from data.const import (
    MGN_CLASSES,
    GARMENT_CLASSES,
    MGN_DATASET,
    UV_MAPS_PATH
)


def texture_meshes(
        self,
        meshes: list, 
        texture_paths: list, 
        garment_tag: list, 
        uv_maps_pth: str
    ) -> Tuple[Mesh]:
    '''
    Texture the [body, upper garment, lower garment] meshes from list meshes
    with textures from texture_paths -- mesh can be None if no mesh is available
    Arguments:
        meshes: list of 3 psbody Mesh classes for the body, upper garment and lower garment.
                a mesh can be None if you don't want to texture that part of mesh
        texture_paths: list of 3 paths to texture images for body, upper garment and lower garment
        garment_tag: list of 2 garment labels for the upper garment and lower garment
        uv_maps_pth: folder path where the uv maps are stored for each garment type
    Returns:
        textured_meshes: list of 3 textured meshes from the input
    '''
    body_mesh = meshes[0]
    ug_mesh = meshes[1]
    lg_mesh = meshes[2]

    # load pre-defined uv maps
    general_vt = np.load(f'{uv_maps_pth}/general_vt.npy')

    # texture body
    if body_mesh is not None:
        body_ft = np.load(f'{uv_maps_pth}/body_ft.npy')
        
        body_mesh.vt = general_vt
        body_mesh.ft = body_ft
        body_mesh.set_texture_image(texture_paths[0])

    # texture upper garment
    if ug_mesh is not None:
        
        ug_tag = garment_tag[0]
        ug_ft = np.load(f'{uv_maps_pth}/{ug_tag}_ft.npy')

        ug_mesh.vt = general_vt
        ug_mesh.ft = ug_ft
        ug_mesh.set_texture_image(texture_paths[1])

    # texture lower garment
    if lg_mesh is not None:

        lg_tag = garment_tag[1]
        lg_ft = np.load(f'{uv_maps_pth}/{lg_tag}_ft.npy')

        lg_mesh.vt = general_vt
        lg_mesh.ft = lg_ft
        lg_mesh.set_texture_image(texture_paths[2])

    return [body_mesh, ug_mesh, lg_mesh]


class TextureManager:

    def __init__(
            self, 
            save_to_disk: bool = False
        ) -> None:
        garment_dict = {}
        for garment in MGN_CLASSES:
            garment_dict[garment] = glob(join(MGN_DATASET, '*', garment + '.obj'))
        
        self.save_to_disk = save_to_disk
        if not os.path.exists(UV_MAPS_PATH) and save_to_disk:
            os.makedirs(UV_MAPS_PATH)
        
        self.vt_map = None
        self.garment_ft_maps = []
        self.body_ft_map = None
        self.texture_images = []

        self._create_vt_map(garment_dict, UV_MAPS_PATH)
        self._create_ft_maps(garment_dict, UV_MAPS_PATH)
        self._create_body_ft_map(UV_MAPS_PATH)
        
        all_textures = sorted(glob(f'{MGN_DATASET}/*/multi_tex.jpg'))
        all_segmentations = sorted(glob(f'{MGN_DATASET}/*/segmentation.png'))

        self._generate_body_texture_from_texture(
            texture_pth=all_textures,
            segmentation_pth=all_segmentations,
            save_to_disk=True
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
        if self.save_to_disk:
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
            if self.save_to_disk:
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
        if self.save_to_disk:
            np.save(f'{save_to}/body_ft.npy', body_ft_map)
        else:
            self.body_ft_map = body_ft_map
        
    def _create_texture_images(
            self, 
            texture_paths: List[str], 
            segmentation_paths: List[str], 
            skin_color_pixel: List[float],
            save_to_disk: bool = False
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
            save_to_disk: save generated body texture to same folder as texture_pth
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
            if save_to_disk:
                texture_image.save(f'{save_to}/body_tex.jpg')
            else:
                self.texture_images.append(texture_image)


if __name__ == '__main__':
    TextureManager()
