#!/bin/bash

# arg 1 - path to garmentor
# arg 2 - path to <lsun_unzipped>
# arg 3 - path to <lsun_images>
# arg 4 - Optional: Max number of images to extract from each database

if [ "$#" -eq 4 -o "$#" -eq 3 ]; then
    if [ "$#" -eq 4 ]; then
        re='^[0-9]+$'
        if ! [[ $4 =~ $re ]]; then
            echo "4th argument $4 is not a valid integer"
            exit 1
        fi
        MAX_EXTRACT=$4
    else
        MAX_EXTRACT=-1
    fi
    docker run -it \
        --rm \
        --runtime=nvidia \
        -v $1:/garmentor \
        -v $2:/lsun_unzipped \
        -v $3:/lsun_images \
        -w /garmentor/lsun_for_garmentor \
        -e MAX_EXTRACT=$MAX_EXTRACT \
        $USER-garmentor /bin/bash -c '
        DIRECTORIES=
        for dir in /lsun_unzipped/*/
        do
            DIRECTORIES+=$dir" "
        done
        DIRECTORIES=$DIRECTORIES | xargs
        python data.py export $DIRECTORIES --out_dir /lsun_images --max_extract $MAX_EXTRACT
        for dir in /lsun_images/*/
        do
            chmod -R a+rw $dir
        done
        '
else
    echo "1st argument: Path to garmentor"
    echo "2nd argument: Path to <lsun_unzipped>"
    echo "3rd argument: Path to <lsun_images>"
    echo "4th argument: [Optional]: Maximum number of images to extract from each database"
fi