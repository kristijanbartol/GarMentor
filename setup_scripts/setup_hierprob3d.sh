#!/bin/bash

# arg 1 - path to <hierprob3d_data_root>

if [ "$#" -ne 1 ]; then
    echo "1st argument: Path to <hierprob3d_data_root>"
else
    docker run -it \
        --rm \
        --runtime=nvidia \
        -v $1:/hierprob3d_data \
        -w /git/smplx/tools \
        $USER-garmentor /bin/bash -c '
        for file in /hierprob3d_data/*.pkl
        do
            python clean_ch.py --input-models $file --output-folder /hierprob3d_data/smpl
        done
        '
fi