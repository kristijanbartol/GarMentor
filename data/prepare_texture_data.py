
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


def create_vt_map(garment_dict: dict, save_to: str):
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

    np.save(f'{save_to}/general_vt.npy',loaded_mesh.vt)

def create_ft_map(garment_dict: dict, garment_name: str, save_to:str):
    '''
    Create ft map for specific garments saved as garment_ft.npy
    Arguments:
        garment_name: one name from the garment_classes
        save_to: save created vt map to this path
    Returns:
        saves garment_ft.npy to disk
    '''
    
    random_mesh_name = garment_dict[garment_name][0]
    loaded_mesh =  Mesh(filename=random_mesh_name)

    index = MGN_CLASSES.index(garment_name)
    garment_tailornet_name = GARMENT_CLASSES[index]

    np.save(f'{save_to}/{garment_tailornet_name}_ft.npy',loaded_mesh.ft)

def create_body_ft_map(save_to: str):
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

    np.save(f'{save_to}/body_ft.npy',loaded_mesh.ft)
    

def generate_body_texture_from_texture(texture_pth: str, segmentation_pth:str, skin_color_pixel=None,
                                        save_to_disk=False):
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

    save_to = texture_pth.split('/multi_tex.jpg')[0]
    background_color = [0,0,0] # black

    # load texture and segmentation
    texture_img = Image.open(texture_pth)
    texture_img_array = np.array(texture_img)
    segmentation_img = Image.open(segmentation_pth)

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

    if save_to_disk:
        body_texture_image = Image.fromarray(texture_img_array)
        body_texture_image.save(f'{save_to}/body_tex.jpg')


if __name__ == '__main__':
    garment_dict = {}
    for garment in MGN_CLASSES:
        garment_dict[garment] = glob(join(MGN_DATASET, '*', garment + '.obj'))
    
    if not os.path.exists(UV_MAPS_PATH):
        os.makedirs(UV_MAPS_PATH)
    create_vt_map(garment_dict, UV_MAPS_PATH)
    for garment_name in MGN_CLASSES:
        create_ft_map(garment_dict, garment_name, UV_MAPS_PATH)
    create_body_ft_map(UV_MAPS_PATH)
    
    all_textures = sorted(glob(f'{MGN_DATASET}/*/multi_tex.jpg'))
    all_segmentations = sorted(glob(f'{MGN_DATASET}/*/segmentation.png'))
    nr_files = len(all_textures)
    for i in tqdm(range(nr_files)):
        generate_body_texture_from_texture(texture_pth=all_textures[i],
                                           segmentation_pth=all_segmentations[i],
                                           save_to_disk=True)
