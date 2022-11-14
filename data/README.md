## Generate AGORA meshes

1. Download the MGN texture dataset from [MGN GitHub](https://github.com/bharat-b7/MultiGarmentNetwork#dress-smpl-body-model-with-our-digital-wardrobe)
   Part-1 and Part-2 and unpack them in `/data/mgn/`.
2. Download the SMPL and SMPL-X fits from the [Agora Website](https://agora.is.tue.mpg.de/) and extract them to `/data/agora`
3. Run `prepare_texture_data.py` (only once).
4. Run `generate_subjects_meshes.py` to do the following:
   - i. Load the metadata/parameters of SMPL-X fits to 3D scans used by AGORA.
   - ii. Create `beta`, `theta`, and apply body rotation (`global_orient`).
   - iii. Convert SMPL-X to SMPL parameters.
   - iv. Randomly sample style.
   - v. Run parametric model to obtain cloth meshes (set your garment combination in the `__main__` part of the script).
   - vi. Apply texture.
   - vii. Save meshes (body, upper, lower).
   - You do it only once for all original scan/SMPL-X model. We can later update to generate more subjects (different than AGORA's). The script can be updated so that different clothing class combinations are also genered (shirt, short-pant).
5. Run `generate_scene_meshes.py` to do the following:
   - i. Load camera information from AGORA dataset.
   - ii. Apply specified translations (rotations are already applied).
   - iii. Store transformed meshes for individual scene.
   - The script can be updated to store more information. It can also be used only to extract camera information and mesh translation and then apply these transformation in Blender directly to avoid storing meshes for particular scenes.
