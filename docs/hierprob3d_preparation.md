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
Please note: You will need to download roughly 157 GB of data
1) Create the directory `<hierprob3d_data_root>/training`
2) Download train and validation body poses and textures from [here](https://drive.google.com/drive/folders/1lvxwKcqi4HaxTLQlEicPhN5Q3L-aWjYN) and move them to `<hierprob3d_data_root>/training`
3) Download the [LSUN](https://www.yf.io/p/lsun) dataset to a directory `<lsun_original_data>`
    * Clone [this](https://github.com/fyu/lsun) repository
    * From within this repository, run
        ```bash
        python3 download.py -o <lsun_original_data>
        ```
        * The download can take a while, depending on your internet connection (~157 GB)
4) Run
    ```bash
    garmentor/setup_scripts/setup_hierprob3d_training.sh /path/to/garmentor/directory <hierprob3d_data_root> <lsun_original_data>
    ```

Your `<hierprob3d_data_root>` should now look like this:
```bash
```

