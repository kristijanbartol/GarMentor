# 3DPW Dataset Setup
1) Download the 3DPW dataset from [here](https://virtualhumans.mpi-inf.mpg.de/3DPW/)
2) Create a directory `<3dpw_root>`
3) Extract `imageFiles.zip` and `sequenceFiles.zip` to `<3dpw_root>`
4) Run
    ```bash
    garmentor/setup_scripts/setup_3dpw.sh /path/to/garmentor/directory <3dpw_root> <hierprob3d_data_root>
    ```
5) Download HRNet 2D joint detections on 3DPW from [here](https://drive.google.com/drive/folders/1GnVukI3Z1h0fq9GeD40RI8z35EfKWEda) and copy them to `<3dpw_root>/test`

Your `<3dpw_root>` should now look like this:
```
3dpw_test/
├── imageFiles
│   ├── courtyard_arguing_00
│   ├── courtyard_backpack_00
│   ├── ...
│   └── outdoors_slalom_01
├── sequenceFiles
│   ├── test
│   ├── train
│   └── validation
└── test
    ├── 3dpw_test.npz
    ├── cropped_frames
    └── hrnet_results_centred.npy
```