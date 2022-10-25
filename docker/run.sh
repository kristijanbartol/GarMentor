#!/bin/bash

if [[ "$USER" == "kristijan" ]]; then
	REPO_DIR=/media/kristijan/kristijan-hdd-ex/garmentor/
	BASE_DATA_DIR=/media/kristijan/kristijan-hdd-ex/datasets/
	PW3D_DIR=${BASE_DATA_DIR}/3dpw/
	SSP3D_DIR=${BASE_DATA_DIR}/SSP-3D/
	TAILORNET_DATA_DIR=${BASE_DATA_DIR}/tailornet
	HIERPROB3D_DATA_DIR=${BASE_DATA_DIR}/hierprob3d
	GARMENTOR_DATA_DIR=${BASE_DATA_DIR}/garmentor
	MGN_DATA_DIR=${BASE_DATA_DIR}/mgn
elif [[ "$USER" == "dbojanic" ]]; then
	REPO_DIR=/home/dbojanic/garmentor/
	BASE_DATA_DIR=/home/dbojanic/data/
	PW3D_DIR=${BASE_DATA_DIR}/3dpw/
	SSP3D_DIR=${BASE_DATA_DIR}/SSP-3D/
	TAILORNET_DATA_DIR=${BASE_DATA_DIR}/tailornet/
	HIERPROB3D_DATA_DIR=${BASE_DATA_DIR}/hierprob3d/
	GARMENTOR_DATA_DIR=${BASE_DATA_DIR}/garmentor/
	MGN_DATA_DIR=${BASE_DATA_DIR}/mgn
	USER="kbartol"
elif [[ "$USER" == "julien" ]]; then
	REPO_DIR=/home/julien/git/garmentor
	PW3D_DIR=/home/julien/data/3DPW
	SSP3D_DIR=/home/julien/data/SSP-3D
	TAILORNET_DATA_DIR=/home/julien/data/TailorNet
	HIERPROB3D_DATA_DIR=/home/julien/data/HierProb3D
	GARMENTOR_DATA_DIR=/home/julien/data/garmentor
	MGN_DATA_DIR=/home/julien/data/mgn
else
	REPO_DIR=/path/to/garmentor/repository
	PW3D_DIR=/path/to/3DPW/dataset
	SSP3D_DIR=/path/to/SSP-3D/dataset
	TAILORNET_DATA_DIR=/path/to/tailornet/data
	HIERPROB3D_DATA_DIR=/path/to/hierprob3d/data
	GARMENTOR_DATA_DIR=/path/to/garmentor/folder
	MGN_DATA_DIR=/path/to/mgn/folder
fi

docker run -it \
	-p 8888:8888 \
	--rm \
	--gpus all \
	--shm-size=8gb \
	--name $USER-garmentor \
	-v ${REPO_DIR}:/garmentor \
	-v ${PW3D_DIR}:/data/3DPW/ \
	-v ${SSP3D_DIR}:/data/SSP-3D/ \
	-v ${TAILORNET_DATA_DIR}:/data/tailornet/ \
	-v ${HIERPROB3D_DATA_DIR}:/data/hierprob3d/ \
	-v ${GARMENTOR_DATA_DIR}:/data/garmentor/ \
	-v ${MGN_DATA_DIR}:/data/mgn/ \
	$USER-garmentor
