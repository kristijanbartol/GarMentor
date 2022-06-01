#!/bin/bash

# arg 1 - path to garmentor
# arg 2 - path to <hierprob3d_data_root>
# arg 3 - Path to downloaded lsun dataset

if [ "$#" -ne 3 ]; then
    echo "1st argument: Path to garmentor repository"
    echo "2nd argument: Path to <hierprob3d_data_root>"
    echo "3rd argument: Path to <lsun_original_data>"
else
    docker run -it \
        --rm \
        --runtime=nvidia \
        -v $1:/garmentor \
        -v $2:/hierprob3d_data \
        -v $3:/lsun_data \
        -w /garmentor/data \
        $USER-garmentor /bin/bash -c '
        python copy_lsun_images_to_train_files_dir.py --lsun_dir /lsun_data --train_files_dir /hierprob3d_data/training
        chmod -R a+rw /data/hierprob3d/training
        '
    