# HierProb3D Data Preparation
## Setup Data for Inference
1) Create a directory that should contain the data for HierProb3D (from now on referred to as `<hierprob3d_data_root>`)
2) Extract the downloaded SMPL male, female and neutral model `*.pkl` files to `<hierprob3d_data_root>`
3) Run
    ```bash
    garmentor/setup_scripts/setup_hierprob3d_inference.sh /path/to/garmentor/directory <hierprob3d_data_root>
    ```
    * This converts chumpy objects in these models to numpy arrays and copies necessary files to your `<hierprob3d_data_root>` directory
4) Remove the original SMPL model `*.pkl` files from `<hierprob3d_data_root>`
5) Download pre-trained checkpoints from [here](https://drive.google.com/drive/folders/1WHdbAaPM8-FpnwMuCdVEchskgKab3gel) and extract them to `<hierprob3d_data_root>`

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
        * Clone [this](https://github.com/jufi2112/lsun_for_garmentor) repository 
        * From within this repository, run
            ```bash
            python3 download.py -o <lsun_zipped>
            ```
    * Option 2:
        * Manually download the dataset from [this](http://dl.yf.io/lsun/scenes/) website to `<lsun_zipped>`
5) Extract the downloaded `*.zip` files to `<lsun_unzipped>`
    * It's not a problem if you're unable to unzip the files that contain the test images 
6) To extract the images from the unzipped databases, you can
    * Option 1:
        * Use the previously created Python environment and run, from within the cloned `lsun_for_garmentor` repository,
            ```
            python data.py export <list all extracted folders from inside <lsun_unzipped> > --out_dir <lsun_images>
            ```
    * Option 2:
        * Run
            ```bash
            garmentor/setup_scripts/setup_hierprob3d_data.sh /path/to/garmentor/directory <lsun_unzipped> <lsun_images>
            ```
5) Run
    ```bash
    garmentor/setup_scripts/setup_hierprob3d_training.sh /path/to/garmentor/directory <hierprob3d_data_root> <lsun_images>
    ```

Your `<hierprob3d_data_root>` should now look like this:
```bash
```

