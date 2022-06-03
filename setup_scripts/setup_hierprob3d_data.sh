#!/bin/bash

# arg 1 - path to garmentor
# arg 1 - path to <lsun_unzipped>
# arg 2 - path to <lsun_images>

if [ "$#" -ne 3 ]; then
    echo "1st argument: Path to garmentor"
    echo "2nd argument: Path to <lsun_unzipped>"
    echo "3rd argument: Path to <lsun_images>"
else
    docker run -it \
        --rm \
        --runtime=nvidia \
        -v $1:/garmentor \
        -v $2:/lsun_unzipped \
        -v $3:/lsun_images \
        -w /garmentor/lsun_for_garmentor \
        $USER-garmentor /bin/bash -c '
        DIRECTORIES=
        for dir in /lsun_unzipped/*/
        do
            DIRECTORIES+=$dir" "
        done
        DIRECTORIES=$DIRECTORIES | xargs
        python data.py export $DIRECTORIES --out_dir /lsun_images
        for dir in /lsun_images/*/
        do
            chmod -R a+rw $dir
        done
        '
fi