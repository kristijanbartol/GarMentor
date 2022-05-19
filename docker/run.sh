#!/bin/sh

if [[ "$(whoami)" == "kristijan" ]]; then
	REPO_DIR=/media/kristijan/kristijan-hdd-ex/meshurer/
	BASE_DATA_DIR=/media/kristijan/kristijan-hdd-ex/data/
		else
	REPO_DIR=/home/dbojanic/meshurer/
	BASE_DATA_DIR=/home/dbojanic/data/
fi

docker run --rm --gpus all --shm-size=8gb --name kbartol-meshurer -it \
	-v ${REPO_DIR}:/meshurer \
	-v ${BASE_DATA_DIR}/3dpw/:/data/3dpw/ \
	-v ${BASE_DATA_DIR}:/SSP-3D/ kbartol-meshurer

