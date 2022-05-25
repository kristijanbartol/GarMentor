#!/bin/sh

if [[ "$USER" == "kristijan" ]]; then
	REPO_DIR=/media/kristijan/kristijan-hdd-ex/garmentor/
	BASE_DATA_DIR=/media/kristijan/kristijan-hdd-ex/data/
elif [[ "$USER" == "dbojanic" ]]; then
	REPO_DIR=/home/dbojanic/garmentor/
	BASE_DATA_DIR=/home/dbojanic/data/
else
	REPO_DIR=/home/$USER/garmentor/
	BASE_DATA_DIR=/home/$USER/data/
fi

docker run --rm --gpus all --shm-size=8gb --name $USER-garmentor -it \
	-v ${REPO_DIR}:/garmentor \
	-v ${BASE_DATA_DIR}/3dpw/:/data/3dpw/ \
	-v ${BASE_DATA_DIR}:/SSP-3D/ $USER-garmentor
