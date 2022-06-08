#!/bin/bash

# arg 1 - path to garmentor
# arg 2 - path to <hierprob3d_data_root>

if [ "$#" -ne 2 ]; then
    echo "1st argument: Path to garmentor directory"
    echo "2nd argument: Path to <hierprob3d_data_root>"
else
    for file in $1/model_files/*
    do
        cp $file $2
    done
    docker run -it \
        --rm \
        --runtime=nvidia \
        -v $1:/garmentor \
        -v $2:/hierprob3d_data \
        -w /git/smplx/tools \
        $USER-garmentor /bin/bash -c '
        for file in /hierprob3d_data/*.pkl
        do
            python clean_ch.py --input-models $file --output-folder /hierprob3d_data/smpl
        done
        chmod -R a+rwx /hierprob3d_data/smpl
        for file in /hierprob3d_data/smpl/*
        do
            if [[ $file == *"_f_"* ]]; then
                mv $file /hierprob3d_data/smpl/SMPL_FEMALE.pkl
            elif [[ $file == *"_m_"* ]]; then
                mv $file /hierprob3d_data/smpl/SMPL_MALE.pkl
            elif [[ $file == *"_neutral_"* ]]; then
                mv $file /hierprob3d_data/smpl/SMPL_NEUTRAL.pkl
            else
                echo "$file does not follow the assumed naming convention, please manually change the filename according to the associated gender"
            fi
        done
        '
fi