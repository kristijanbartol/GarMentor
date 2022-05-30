#!/bin/bash

if [[ "$USER" == "kristijan" ]]; then
	REPO_DIR=/media/kristijan/kristijan-hdd-ex/garmentor/
	ThreeDPW_DIR=TODO
	SSP3D_DIR=TODO
	TAILORNET_DATA_DIR=TODO
	HIERPROB3D_DATA_DIR=TODO
elif [[ "$USER" == "dbojanic" ]]; then
	REPO_DIR=/home/dbojanic/garmentor/
	ThreeDPW_DIR=TODO
	SSP3D_DIR=TODO
	TAILORNET_DATA_DIR=TODO
	HIERPROB3D_DATA_DIR=TODO
elif [[ "$USER" == "julien" ]]; then
	REPO_DIR=/home/julien/git/garmentor
	ThreeDPW_DIR=/home/julien/data/3DPW
	SSP3D_DIR=/home/julien/data/SSP-3D
	TAILORNET_DATA_DIR=/home/julien/body_models
	HIERPROB3D_DATA_DIR=/home/julien/git/HierarchicalProbabilistic3DHuman/model_files
else
	REPO_DIR=/path/to/garmentor/repository
	ThreeDPW_DIR=/path/to/3DPW/dataset
	SSP3D_DIR=/path/to/SSP-3D/dataset
	TAILORNET_DATA_DIR=/path/to/tailornet/data
	HIERPROB3D_DATA_DIR=/path/to/hierprob3d/data
fi

docker run --rm --gpus all --shm-size=8gb --name $USER-garmentor -it \
	-v ${REPO_DIR}:/garmentor \
	-v ${ThreeDPW_DIR}:/data/3DPW/ \
	-v ${SSP3D_DIR}:/data/SSP-3D/ \
	-v ${TAILORNET_DATA_DIR}:/data/tailornet/ \
	-v ${HIERPROB3D_DATA_DIR}:/data/hierprob3d/ \
	$USER-garmentor
