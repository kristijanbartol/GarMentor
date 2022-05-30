#!/bin/bash

# arg 1 - path to garmentor directory
# arg 2 - path to <tailornet_root_dir>

if [ "$#" -ne 2 ]; then
    echo "1st argument: Path to garmentor directory"
    echo "2nd argument: Path to <tailornet_root_dir>"
else
    docker run -it \
        --rm \
        --runtime=nvidia \
        -v $1:/garmentor \
        -v $2:/tailornet_data \
        -w /garmentor/tailornet_dataset-for-garmentor \
        $USER-garmentor \
        python -m smpl_lib.convert_smpl_models
fi


