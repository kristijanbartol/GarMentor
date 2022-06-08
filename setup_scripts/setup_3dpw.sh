#!/bin/bash

# arg 1 - path to garmentor directory
# arg 2 - path to <3dpw_root>
# arg 3 - path to <hierprob3d_data_root>

if [ "$#" -ne 3 ]; then
    echo "1st argument: Path to garmentor directory"
    echo "2nd argument: Path to <3dpw_root>"
    echo "3rd argument: Path to <hierprob3d_data_root>"
else
    docker run -it \
        --rm \
        --runtime=nvidia \
        -v $1:/garmentor \
        -v $2:/data/3DPW \
        -v $3:/data/hierprob3d \
        -w /garmentor \
        $USER-garmentor /bin/bash -c '
        python -m data.pw3d_preprocess --dataset_path /data/3DPW
        chmod -R a+rwx /data/3DPW/test
        '
fi