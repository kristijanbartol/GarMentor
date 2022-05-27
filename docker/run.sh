#!/bin/bash

if [[ "$USER" == "kristijan" ]]; then
	REPO_DIR=/media/kristijan/kristijan-hdd-ex/garmentor/
	BASE_DATA_DIR=/media/kristijan/kristijan-hdd-ex/data/
elif [[ "$USER" == "dbojanic" ]]; then
	REPO_DIR=/home/dbojanic/garmentor/
	BASE_DATA_DIR=/home/dbojanic/data/
elif [[ "$USER" == "julien" ]]; then
	REPO_DIR=/home/julien/git/garmentor/
	BASE_DATA_DIR=/home/julien/data/
else
	REPO_DIR=/home/$USER/garmentor/
	BASE_DATA_DIR=/home/$USER/data/
fi

docker run -it --rm --runtime=nvidia --gpus all --shm-size=8gb \
	-v ${REPO_DIR}:/garmentor \
	-v ${BASE_DATA_DIR}/3dpw/:/data/3dpw/ \
	-v ${BASE_DATA_DIR}:/SSP-3D/ \
	$USER-garmentor
