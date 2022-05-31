# HierProb3D Data Preparation
1) Create a directory that should contain the data for HierProb3D (from now on referred to as `<hierprob3d_data_root>`)
2) Extract the downloaded SMPL male, female and neutral model `*.pkl` files to `<hierprob3d_data_root>`
3) Run
    ```bash
    garmentor/setup_scripts/setup_hierprob3d.sh path/to/garmentor/directory <hierprob3d_data_root>
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
├── smpl
│   ├── SMPL_FEMALE.pkl
│   ├── SMPL_MALE.pkl
│   └── SMPL_NEUTRAL.pkl
└── UV_Processed.mat

```