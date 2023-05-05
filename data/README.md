# Generate AGORA Meshes

0. The following scripts are to be run in the docker container built in [this](../README.md) guide.
   - On Linux, adapt the paths in the else clause of `garmentor/docker/run.sh` and run the script to enter the docker container
   - On Windows, adapt and use `garmentor/docker/run.bat`
1. In `garmentor/data/const.py` adapt the `DATA_DIR` as well as the other variables to match your local setup
   - For this guide, we assume `DATA_DIR = '/data/'` and all other directories as subdirectories therein
2. Download the MGN texture dataset from [MGN GitHub](https://github.com/bharat-b7/MultiGarmentNetwork#dress-smpl-body-model-with-our-digital-wardrobe)
   Part-1 and Part-2 and unpack them in `/data/mgn/`.
3. Download the SMPL and SMPL-X fits from the [Agora Website](https://agora.is.tue.mpg.de/) and extract them to `/data/agora`
4. Run `generate/characters.py` to do the following:
   - i. Load the metadata/parameters of SMPL-X fits to 3D scans used by AGORA.
   - ii. Create `beta`, `theta`, and apply body rotation (`global_orient`).
   - iii. Convert SMPL-X to SMPL parameters.
     - This will take a lot of time when run for the first time
     - Following garment combinations will reuse the previously converted parameters
     - If you want to save time, we provide our conversion results as a `.npz` file in `garmentor/assets/conv_results.npz`
       - Copy this file to `<garmentor/data/const.py::GARMENTOR_DIR>/conversion_output/`
   - iv. Randomly sample style.
   - v. Run parametric model to obtain cloth meshes
   - vi. Apply texture.
   - vii. Save meshes (body, upper, lower).
   - The script needs to be run once for each garment combination (e.g. t-shirt & short-pant)
     - You can change the garment combination by changing the values of `UPPER_GARMENT_TYPE` and `LOWER_GARMENT_TYPE` inside of the script to values specified in `garmentor/data/const.py::GARMENT_CLASSES`
5. Run `generate/scene_meshes.py -s brushifyforest archviz construction flowers -n X` to do the following:
   - i. Load camera information from AGORA dataset and sample the amount of scene configurations as specified by the `-n` parameter
      - Provide `-1` to sample all possible configurations
      - Sampling is done randomly, so multiple runs will result in different configurations being sampled
        - **Attention**: Multiple runs can overwrite existing samples (if the same configuration is sampled)
      - Giving a single value to `-n` chooses this value for all scenes, but you can also give a different value for each scene
   - ii. Store the meshes for each sampled scene configuration
      - Each scene configuration is stored in a subfolder according to its image name
      - Each stored `.obj` file will contain three textured meshes (the body mesh as well as upper and lower garment meshes)
      - Additionally, for each `.obj` file, there will be a `.mtl` material definition file as well as three `.jpg` texture files 
   - The `-s` flag specifies for which scenes the script should sample configurations
     - We omitted the brushifygrasslands scene, as we could not obtain a version of the scene that is compatible with AGORA's ground truth information
   - The script will automatically try to evenly sample from all available garment combinations that were generated in step 4

# Send Generated AGORA Meshes to Unreal Engine

1. At the top of the `blender_import.py` script, set the variables according to your needs
2. Open blender and the desired Unreal Engine instance, remove any meshes from the blender scene, and execute the `blender_import.py` script inside of blender
   - This will import the generated AGORA meshes into blender and transfer them to the opened Unreal Engine editor
   - If you are unable to transfer all meshes at once, repeatedly execute the script while increasing the `BATCHES_ALREADY_PROCESSED` variable by one each time the script is run

# Render Images in Unreal Engine

1. Once all meshes have been successfully transferred to Unreal Engine, set up the project structure inside of the engine, see [here](unreal_engine_setup/README.md)

2. Set the variables at the top of the `unreal_render.py` script according to your needs

3. Make sure the correct scene is loaded inside of the engine

3. Execute the `unreal_render.py` script inside of the Unreal Engine editor
   - `File` -> `Execute Python Script`
   - If necessary (depends on batch size) repeat multiple times while increasing the `BATCHES_ALREADY_PROCESSED` variable
