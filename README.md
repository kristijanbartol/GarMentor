# Garmentor: Parametric Pose, Shape, and Clothes In-The-Wild

Work-in-progress

## Installation

### Requirements
- Linux or macOS
- Python ≥ 3.6

### Setup

1) Install Docker and the NVIDIA Container Toolkit (see e.g. [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker))
2) Build our docker image (from within the `garmentor/` directory) with:
    ```
    docker build -t <username>-garmentor docker
    ```
    or execute the `build.sh` script **from within** the `garmentor/docker/` directory.
3) Setup data required by TailorNet and HierProb3D:
    * TailorNet: Please refer to [this document](docs/tailornet_preparation.md)
    * HierProb3D: Please refer to [this document](docs/hierprob3d_preparation.md)
4) Setup 3DPW and SSP-3D datasets:
    * 3DPW: Please refer to [this document](docs/3dpw_preparation.md)
    * SSP-3D: Please refer to [this document](docs/ssp-3d_preparation.md)
5) To run the container, use
    ```
    docker/run.sh
    ```

### Model files
You will need to download the SMPL model. The [neutral model](http://smplify.is.tue.mpg.de) is required for training and running the demo code. If you want to evaluate the model on datasets with gendered SMPL labels (such as 3DPW and SSP-3D), the male and female models are available [here](http://smpl.is.tue.mpg.de). You will need to convert the SMPL model files to be compatible with python3 by removing any chumpy objects. To do so, please follow the instructions [here](https://github.com/vchoutas/smplx/tree/master/tools).

Download pre-trained model checkpoints for our 3D Shape/Pose network, as well as for 2D Pose [HRNet-W48](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch) from [here](https://drive.google.com/drive/folders/1WHdbAaPM8-FpnwMuCdVEchskgKab3gel?usp=sharing). In addition to the neutral-gender prediction network presented in the paper, we provide pre-trained checkpoints for male and female prediction networks, which are trained with male/female SMPL shape respectively. Download these checkpoints if you wish to do gendered shape inference.

Place the SMPL model files and network checkpoints in the `model_files` directory, which should have the following structure. If the files are placed elsewhere, you will need to update `configs/paths.py` accordingly.

    HierarchicalProbabilistic3DHuman
    ├── model_files                                  # Folder with model files
    │   ├── smpl
    │   │   ├── SMPL_NEUTRAL.pkl                     # Gender-neutral SMPL model
    │   │   ├── SMPL_MALE.pkl                        # Male SMPL model
    │   │   ├── SMPL_FEMALE.pkl                      # Female SMPL model
    │   ├── poseMF_shapeGaussian_net_weights.tar     # Pose/Shape distribution predictor checkpoint
    │   ├── pose_hrnet_w48_384x288.pth               # Pose2D HRNet checkpoint
    │   ├── cocoplus_regressor.npy                   # Cocoplus joints regressor
    │   ├── J_regressor_h36m.npy                     # Human3.6M joints regressor
    │   ├── J_regressor_extra.npy                    # Extra joints regressor
    │   └── UV_Processed.mat                         # DensePose UV coordinates for SMPL mesh             
    └── ...
 
## Inference
`run_predict.py` is used to run inference on a given folder of input images. For example, to run inference on the demo folder, do:
```
python run_predict.py --image_dir ./demo/ --save_dir ./output/ --visualise_samples --visualise_uncropped
```
This will first detect human bounding boxes in the input images using Mask-RCNN. If your input images are already cropped and centred around the subject of interest, you may skip this step using `--cropped_images` as an option. The 3D Shape/Pose network is somewhat sensitive to cropping and centering - this is a good place to start troubleshooting in case of poor results.

If the gender of the subject is known, you may wish to carry out gendered inference using the provided male/female model weights. This can be done by modifying the above command as follows:
```
python run_predict.py --gender male --pose_shape_weights model_files/poseMF_shapeGaussian_net_weights_male.tar --image_dir ./demo/ --save_dir ./output_male/ --visualise_samples --visualise_uncropped
```
(similar for the female model). Using gendered models for inference may result in better body shape estimates, as it serves as a prior over 3D shape.

Inference can be slow due to the rejection sampling procedure used to estimate per-vertex 3D uncertainty. If you are not interested in per-vertex uncertainty, you may modify `predict/predict_poseMF_shapeGaussian_net.py` by commenting out code related to sampling, and use a plain texture to render meshes for visualisation (this will be cleaned up and added as an option to in the `run_predict.py` future).

## Evaluation
`run_evaluate.py` is used to evaluate our method on the 3DPW and SSP-3D datasets. A description of the metrics used to measure performance is given in `metrics/eval_metrics_tracker.py`.

Download SSP-3D from [here](https://github.com/akashsengupta1997/SSP-3D). Update `configs/paths.py` with the path pointing to the un-zipped SSP-3D directory. Evaluate on SSP-3D with:
```
python run_evaluate.py -D ssp3d
```

Download 3DPW from [here](https://virtualhumans.mpi-inf.mpg.de/3DPW/). You will need to preprocess the dataset first, to extract centred+cropped images and SMPL labels (adapted from [SPIN](https://github.com/nkolot/SPIN/tree/master/datasets/preprocess)):
```
python -m data.pw3d_preprocess --dataset_path $3DPW_DIR_PATH
```
This should create a subdirectory with preprocessed files, such that the 3DPW directory has the following structure:
```
$3DPW_DIR_PATH
      ├── test                                  
      │   ├── 3dpw_test.npz    
      │   ├── cropped_frames   
      ├── imageFiles
      └── sequenceFiles
```
Additionally, download HRNet 2D joint detections on 3DPW from [here](https://drive.google.com/drive/folders/1GnVukI3Z1h0fq9GeD40RI8z35EfKWEda?usp=sharing), and place this in `$3DPW_DIR_PATH/test`. Update `configs/paths.py` with the path pointing to `$3DPW_DIR_PATH/test`. Evaluate on 3DPW with:
```
python run_evaluate.py -D 3dpw
```
The number of samples used to evaluate sample-related metrics can be changed using the `--num_samples` option (default is 10).

## Training
`run_train.py` is used to train our method using random synthetic training data (rendered on-the-fly during training). 

Download .npz files containing SMPL training/validation body poses and textures from [here](https://drive.google.com/drive/folders/1lvxwKcqi4HaxTLQlEicPhN5Q3L-aWjYN?usp=sharing). Place these files in a `./train_files` directory, or update the appropriate variables in `configs/paths.py` with paths pointing to the these files. Note that the SMPL textures are from [SURREAL](https://github.com/gulvarol/surreal) and [MultiGarmentNet](https://github.com/bharat-b7/MultiGarmentNetwork).

We use images from [LSUN](https://github.com/fyu/lsun) as random backgrounds for our synthetic training data. Specifically, images from the 10 scene categories are used. Instructions to download and extract these images are provided [here](https://github.com/fyu/lsun). The `copy_lsun_images_to_train_files_dir.py` script can be used to copy LSUN background images to the `./train_files` directory, which should have the following structure:
```
train_files
      ├── lsun_backgrounds
          ├── train
          ├── val
      ├── smpl_train_poses.npz
      ├── smpl_train_textures.npz                                  
      ├── smpl_val_poses.npz                                  
      └── smpl_val_textures.npz                                  
```

Finally, start training with:
```
python run_train.py -E experiments/exp_001
```
As a sanity check, the script should find 91106 training poses, 125 + 792 training textures, 397582 training backgrounds, 33347 validation poses, 32 + 76 validation textures and 3000 validation backgrounds.

## Weaknesses and Future Research
The following aspects of our method may be the subject of future research:
- Mesh interpenetrations: this occurs occasionally amongst 3D mesh samples drawn from shape and pose distribution predictions. A sample inter-penetratation penalty may be useful.
- Sample diversity / distribution expressiviness: since the predicted distributions are uni-modal, sample diversity may be limited.
- Sampling speed: rejection sampling from a matrix-Fisher distribution is currently slow.
- Non-tight clothing: body shape prediction accuracy suffers when subjects are wearing non-tight clothing, since the synthetic training data does not model clothing in 3D (only uses clothing textures). Perhaps better synthetic data (e.g. [AGORA](https://agora.is.tue.mpg.de)) will alleviate this issue.

## Acknowledgments
Code was adapted from/influenced by the following repos - thanks to the authors!

- [HMR](https://github.com/akanazawa/hmr)
- [SPIN](https://github.com/nkolot/SPIN)
- [VIBE](https://github.com/mkocabas/VIBE)
- [HRNet](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch)
- [PyTorch3D](https://github.com/facebookresearch/pytorch3d)
- [Probabilistic Orientation Estimation with Matrix Fisher Distributions](https://github.com/Davmo049/Public_prob_orientation_estimation_with_matrix_fisher_distributions)
- [CannyEdgePytorch](https://github.com/DCurro/CannyEdgePytorch)
- [Matrix-Fisher-Distribution](https://github.com/tylee-fdcl/Matrix-Fisher-Distribution)
- [SURREAL](https://github.com/gulvarol/surreal)
- [MultiGarmnetNet](https://github.com/bharat-b7/MultiGarmentNetwork)
- [LSUN](https://github.com/fyu/lsun)
