#!/bin/bash

# arg 1 - path to garmentor directory
# arg 2 - path to <tailornet_data_root>

if [ "$#" -ne 2 ]; then
    echo "1st argument: Path to garmentor directory"
    echo "2nd argument: Path to <tailornet_data_root>"
else
    echo $username
    docker run -it \
        --rm \
        --runtime=nvidia \
        -v $1:/garmentor \
        -v $2:/tailornet_data \
        -w /garmentor/tailornet_dataset_for_garmentor \
        $USER-garmentor /bin/bash -c '
        python -m smpl_lib.convert_smpl_models
        for file in /tailornet_data/smpl/*
        do
            if [[ $file == *"smpl_"* ]]; then
                chmod a+rwx $file
            fi
        done
        '
fi


