# HierProb3D Data Preparation
## Setup Data for Inference
1) Create a directory that should contain the data for HierProb3D (from now on referred to as `<hierprob3d_data_root>`)
2) Extract the downloaded SMPL male, female and neutral model `*.pkl` files to `<hierprob3d_data_root>`
3) Download the [SMPL-X model](https://smpl-x.is.tue.mpg.de/) and extract the `smplx` folder into `<hierprob3d_data_root>`
4) Download the SMPL <-> SMPL-X [model correspondences](https://smpl-x.is.tue.mpg.de/) and extract them to `<hierprob3d_data_root>/transfer`
5) Run
    ```bash
    garmentor/setup_scripts/setup_hierprob3d_inference.sh /path/to/garmentor/directory <hierprob3d_data_root>
    ```
    * This converts chumpy objects in these models to numpy arrays and copies necessary files to your `<hierprob3d_data_root>` directory
6) Remove the original SMPL model `*.pkl` files from `<hierprob3d_data_root>`
7) Download pre-trained checkpoints from [here](https://drive.google.com/drive/folders/1WHdbAaPM8-FpnwMuCdVEchskgKab3gel) and extract them to `<hierprob3d_data_root>`

Your `<hierprob3d_data_root>` should now look like this:
```
<hierprob3d_data_root>/
├── cocoplus_regressor.npy
├── J_regressor_extra.npy
├── J_regressor_h36m.npy
├── pose_hrnet_w48_384x288.pth
├── poseMF_shapeGaussian_net_weights_female.tar
├── poseMF_shapeGaussian_net_weights_male.tar
├── poseMF_shapeGaussian_net_weights.tar
├── smpl
│   ├── SMPL_FEMALE.pkl
│   ├── SMPL_MALE.pkl
│   └── SMPL_NEUTRAL.pkl
├── smplx
│   ├── SMPLX_FEMALE.npz
│   ├── SMPLX_FEMALE.pkl
│   ├── SMPLX_MALE.npz
│   ├── SMPLX_MALE.pkl
│   ├── SMPLX_NEUTRAL.npz
│   ├── SMPLX_NEUTRAL.pkl
│   ├── smplx_npz.zip
│   └── version.txt
├── transfer
│   ├── smpl2smplh_def_transfer.pkl
│   ├── smpl2smplx_deftrafo_setup.pkl
│   ├── smplh2smpl_def_transfer.pkl
│   ├── smplh2smplx_deftrafo_setup.pkl
│   ├── smplx_mask_ids.npy
│   ├── smplx_to_smpl.pkl
│   ├── smplx2smpl_deftrafo_setup.pkl
│   └── smplx2smplh_deftrafo_setup.pkl
└── UV_Processed.mat
```
## Setup Data for Training
Please note: You will need to download roughly 168 GB of data.
1) Create the directory `<hierprob3d_data_root>/training`
2) Download train and validation body poses and textures from [here](https://drive.google.com/drive/folders/1lvxwKcqi4HaxTLQlEicPhN5Q3L-aWjYN) and move them to `<hierprob3d_data_root>/training`
3) In the following, these data locations will be referenced:
    * `<lsun_zipped>`: Directory where the zipped LSUN scenes are located
    * `<lsun_unzipped>`: Directory where the unzipped LSUN databases are located
    * `<lsun_images>`: Directory where the extracted LSUN images are located
4) Download the [LSUN](https://www.yf.io/p/lsun) datasets to `<lsun_zipped>`
    * Option 1:
        * Create a new (or use an existing) python environment and install (e.g. with `pip`) the following packages:
            * `numpy`
            * `opencv-python`
            * `lmdb`
            * `tqdm`
        * From within `garmentor/lsun_for_garmentor`, run
            ```bash
            python3 download.py -o <lsun_zipped>
            ```
    * Option 2:
        * Manually download the dataset from [this](http://dl.yf.io/lsun/scenes/) website to `<lsun_zipped>`
5) Extract the downloaded `*.zip` files to `<lsun_unzipped>`
    * Do **not** extract the test images
6) To extract the images from the unzipped databases, you can
    * Option 1:
        * Use the previously created Python environment and run, from within `garmentor/lsun_for_garmentor`:
            ```
            python data.py export <list all extracted folders from inside <lsun_unzipped> > --out_dir <lsun_images> [--max_extract X]
            ```
            * The `--max_extract X` option allows you to only extract a maximum of `X` images from each database
    * Option 2:
        * Run
            ```bash
            garmentor/setup_scripts/setup_hierprob3d_data.sh /path/to/garmentor/directory <lsun_unzipped> <lsun_images> [X]
            ```
            * Specifying `X` is optional and lets you set the maximum number of images that should be extracted from each database
    * `lsun_images` should now look like this:
        ```bash
        <lsun_images>/
        ├── bedroom_train_lmdb_images
        ├── bedroom_val_lmdb_images
        ├── bridge_train_lmdb_images
        ├── bridge_val_lmdb_images
        ├── church_outdoor_train_lmdb_images
        ├── church_outdoor_val_lmdb_images
        ├── classroom_train_lmdb_images
        ├── classroom_val_lmdb_images
        ├── conference_room_train_lmdb_images
        ├── conference_room_val_lmdb_images
        ├── dining_room_train_lmdb_images
        ├── dining_room_val_lmdb_images
        ├── kitchen_train_lmdb_images
        ├── kitchen_val_lmdb_images
        ├── living_room_train_lmdb_images
        ├── living_room_val_lmdb_images
        ├── restaurant_train_lmdb_images
        ├── restaurant_val_lmdb_images
        ├── tower_train_lmdb_images
        └── tower_val_lmdb_images
        ```
5) Run
    ```bash
    garmentor/setup_scripts/setup_hierprob3d_training.sh /path/to/garmentor/directory <hierprob3d_data_root> <lsun_images>
    ```

Your `<hierprob3d_data_root>` directory should now look like this:
```bash
<hierprob3d_data_root>/
├── cocoplus_regressor.npy
├── J_regressor_extra.npy
├── J_regressor_h36m.npy
├── pose_hrnet_w48_384x288.pth
├── poseMF_shapeGaussian_net_weights_female.tar
├── poseMF_shapeGaussian_net_weights_male.tar
├── poseMF_shapeGaussian_net_weights.tar
├── smpl
│   ├── SMPL_FEMALE.pkl
│   ├── SMPL_MALE.pkl
│   └── SMPL_NEUTRAL.pkl
├── training
│   ├── lsun_backgrounds
│   │   ├── train
│   │   └── val
│   ├── smpl_train_poses.npz
│   ├── smpl_train_textures.npz
│   ├── smpl_val_poses.npz
│   └── smpl_val_textures.npz
└── UV_Processed.mat

```

