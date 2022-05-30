# TailorNet Data Preparation
1) Download the SMPL models (Female, Male and Neutral) from [this](https://smpl.is.tue.mpg.de/) link (requires registration)
2) Unzip the file and put the 3 `*.pkl` files from `SMPL_python_v.1.1.0/smpl/models/` in a directory `<tailornet_data_root>/smpl`
    * `<tailornet_data_root>` can be choosen arbitrarily
3) Clone [this](https://github.com/zycliao/TailorNet_dataset) repository
4) Change into the cloned directory and modify the `global_var.py` script:
    * Set `ROOT=<tailornet_data_root>`
    * TODO: Downloaded male and female models are v.1.1.0, but every script references them as v.1.0.0, why? Do we have to rename these files, or can we safely change all references to v.1.1.0?
    * For `SMPL_PATH_MALE` and `SMPL_PATH_FEMALE` change the filename (the last argument in the `os.path.join` call) to the corresponding filename that you downloaded previously
5) From the `TailorNet_dataset` directory, run
    ```bash
    python -m smpl_lib.convert_smpl_models
    ```
6) Download TailorNet dataset from [here](https://nextcloud.mpi-klsb.mpg.de/index.php/s/W7a57iXRG9Yms6P)
    * Download `dataset_meta.zip` (dataset meta data)
    * Download one or more sub-datasets (each represents a different garment type):
        * t-shirt_female(6.9 GB)
        * t-shirt_male(7.2 GB)
        * old-t-shirt_female(10 GB)
        * t-shirt_female_sample(19 MB)
        * shirt_female(12.7 GB)
        * shirt_male(13.5 GB)
        * pant_female(3.3 GB)
        * pant_male(3.4 GB)
        * short-pant_female(1.9 GB)
        * short-pant_male(2 GB)
        * skirt_female(5 GB)
7) Unzip the downloaded files to `<tailornet_data_root>`
8) For the garment types you downloaded, download the pre-trained weights from [here](https://nextcloud.mpi-klsb.mpg.de/index.php/s/LTWJPcRt7gsgoss)
9) Unzip the downloaded weights to a directory `<tailornet_data_root>/weights`
    * TODO: change SMPL model version in filename to 1.1.0? Or do we keep it? In `TailorNet/global_var.py`

Your `<tailornet_data_root>` should now look similar to this (depending on the different garment types that you downloaded):

```
.
├── apose.npy
├── apose.pkl
├── garment_class_info.pkl
├── garment_class_info_py2.pkl
├── pant_upper_boundary.npy
├── shirt_left_boundary.npy
├── shirt_right_boundary.npy
├── shirt_upper_boundary.npy
├── short-pant_upper_boundary.npy
├── skirt_upper_boundary.npy
├── skirt_weight.npz
├── smpl
│   ├── basicModel_f_lbs_10_207_0_v1.0.0.pkl
│   ├── basicmodel_m_lbs_10_207_0_v1.0.0.pkl
│   ├── basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl
│   ├── smpl_female.npz
│   ├── smpl_hres_female.npz
│   ├── smpl_hres_male.npz
│   ├── smpl_male.npz
├── split_static_pose_shape.npz
├── t-shirt_male
│   ├── avail.txt
│   ├── pivots.txt
│   ├── pose
│   ├── shape
│   ├── style
│   ├── style_model.npz
│   ├── style_shape
│   └── test.txt
├── t-shirt_upper_boundary.npy
└── weights
    └── t-shirt_male_weights
```
<!---
SMPL_PATH_NEUTRAL = '/data/tailornet/smpl/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl'
SMPL_PATH_MALE = '/data/tailornet/smpl/basicmodel_m_lbs_10_207_0_v1.0.0.pkl'
SMPL_PATH_FEMALE = '/data/tailornet/smpl/basicModel_f_lbs_10_207_0_v1.0.0.pkl' 
--->