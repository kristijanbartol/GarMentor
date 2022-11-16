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
5. Run `generate_scene_meshes.py -s brushifygrasslands brushifyforest archviz construction flowers -n X` to do the following:
   - i. Load camera information from AGORA dataset and sample the amount of scene configurations as specified by the `-n` parameter
      - Provide `-1` to sample all possible configurations
      - Sampling is done randomly, so multiple runs will result in different configurations being sampled
      - Currently: Multiple runs can overwrite existing samples (if the same configuration is sampled)
         - Planned for the future: `-n` defines how many configurations should be sampled _after_ the script finishes, resulting in noop if the given number of configurations has already been sampled in a previous run
      - Giving a single value to `-n` chooses this value for all scenes, but you can also give a different value for each scene
   - ii. Apply translations as specified by the camera information from AGORA (rotations are already applied).
   - iii. Store transformed meshes for each sampled scene configurations
      - Each scene configuration is stored in a subfolder according to its image name
      - Each stored `.obj` file will contain three textured meshes (the body mesh as well as upper and lower garment meshes)
      - Additionally, for each `.obj` file, there will be a `.mtl` material definition file as well as three `.jpg` texture files 
   - The `-s` flag specifies for which scenes the script should sample configurations
