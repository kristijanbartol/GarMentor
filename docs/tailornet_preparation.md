# TailorNet Data Preparation
1) Download the SMPL models (Female, Male and Neutral) from [this](https://smpl.is.tue.mpg.de/) link (requires registration)
2) Unzip the file and put the three `*.pkl` files from `SMPL_python_v.1.1.0/smpl/models/` in a directory `<tailornet_data_root>/smpl`
    * `<tailornet_data_root>` can be choosen arbitrarily
3) In `garmentor/tailornet_dataset_for_garmentor/global_var.py` , make sure that the filenames for `SMPL_PATH_MALE` and `SMPL_PATH_FEMALE` are the same as the filenames of the `*.pkl` files that you unzipped in step 2
    * The filenames are the last arguments in the `os.path.join()` calls
    * It's ok that no neutral model is listed here
4) In `garmentor/tailornet_for_garmentor/global_var.py` , make sure that the **filenames** for `SMPL_PATH_[FEMALE, MALE, NEUTRAL]` correspond to the filenames of the `*.pkl` files that you unzipped in step 2
    * Only modify the filenames but keep the paths as they are
5) Run
    ```bash
    garmentor/setup_scripts/setup_tailornet.sh /path/to/garmentor/directory <tailornet_data_root>
    ```
    * This converts the SMPL models for use in TailorNet
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
9) Unzip the downloaded weights to `<tailornet_data_root>/weights`

Your `<tailornet_data_root>` should now look similar to this (depending on the different garment types that you downloaded):

```
<tailornet_data_root>/
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
│   ├── basicmodel_f_lbs_10_207_0_v1.1.0.pkl
│   ├── basicmodel_m_lbs_10_207_0_v1.1.0.pkl
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