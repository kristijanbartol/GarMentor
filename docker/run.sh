#!/bin/bash

if [[ "$USER" == "kristijan" ]]; then
	REPO_DIR=/media/kristijan/GarMentor/
	BASE_DATA_DIR=/media/kristijan/data/
	DRAPENET_REPO=/media/kristijan/DrapeNet/
	DRAPENET_DATA=/media/kristijan/data/drapenet/
	PW3D_DIR=${BASE_DATA_DIR}/3dpw/
	SSP3D_DIR=${BASE_DATA_DIR}/SSP-3D/
	TAILORNET_DATA_DIR=${BASE_DATA_DIR}/tailornet
	HIERPROB3D_DATA_DIR=${BASE_DATA_DIR}/hierprob3d
	CAT_DATA_DIR=${BASE_DATA_DIR}/cat
	MGN_DATA_DIR=${BASE_DATA_DIR}/mgn
	AGORA_DATA_DIR=${BASE_DATA_DIR}/agora
	FRANKMOCAP_DATA_DIR=${BASE_DATA_DIR}/frankmocap
elif [[ "$USER" == "dbojanic" ]]; then
	REPO_DIR=/home/dbojanic/GarMentor/
	BASE_DATA_DIR=/home/dbojanic/data/
	DRAPENET_REPO=/media/kristijan/DrapeNet/
	DRAPENET_DATA=/home/dbojanic/data/drapenet/
	PW3D_DIR=${BASE_DATA_DIR}/3dpw/
	SSP3D_DIR=${BASE_DATA_DIR}/SSP-3D/
	TAILORNET_DATA_DIR=${BASE_DATA_DIR}/tailornet/
	HIERPROB3D_DATA_DIR=${BASE_DATA_DIR}/hierprob3d/
	CAT_DATA_DIR=${BASE_DATA_DIR}/cat/
	MGN_DATA_DIR=${BASE_DATA_DIR}/mgn
	AGORA_DATA_DIR=${BASE_DATA_DIR}/agora
	FRANKMOCAP_DATA_DIR=${BASE_DATA_DIR}/frankmocap
	USER="kbartol"
elif [[ "$USER" == "julien" ]]; then
	REPO_DIR=/home/julien/git/GarMentor
	BASE_DATA_DIR=/home/julien/data/
	DRAPENET_REPO=/home/julien/DrapeNet/
	DRAPENET_DATA=/home/julien/data/drapenet/
	PW3D_DIR=/home/julien/data/3DPW
	SSP3D_DIR=/home/julien/data/SSP-3D
	TAILORNET_DATA_DIR=/home/julien/data/TailorNet
	HIERPROB3D_DATA_DIR=/home/julien/data/HierProb3D
	CAT_DATA_DIR=/home/julien/data/cat
	MGN_DATA_DIR=/home/julien/data/mgn
	AGORA_DATA_DIR=/home/julien/data/agora
	FRANKMOCAP_DATA_DIR=${BASE_DATA_DIR}/frankmocap
else
	REPO_DIR=/path/to/GarMentor/repository
	BASE_DATA_DIR=/path/to/data/dir/
	DRAPENET_REPO=/path/to/drapenet/repo/
	DRAPENET_DATA=/path/to/drapenet/data/dir/
	PW3D_DIR=/path/to/3DPW/dataset
	SSP3D_DIR=/path/to/SSP-3D/dataset
	TAILORNET_DATA_DIR=/path/to/tailornet/data
	HIERPROB3D_DATA_DIR=/path/to/hierprob3d/data
	CAT_DATA_DIR=/path/to/cat/folder
	MGN_DATA_DIR=/path/to/mgn/folder
	AGORA_DATA_DIR=/path/to/agora/folder
	FRANKMOCAP_DATA_DIR=${BASE_DATA_DIR}/frankmocap
fi

docker run -it \
	-p 8888:8888 \
	--rm \
	--gpus all \
	--shm-size=8gb \
	--name $USER-garmentor \
	-v ${REPO_DIR}:/GarMentor \
	-v ${DRAPENET_REPO}:/GarMentor/DrapeNet/ \
	-v ${DRAPENET_DATA}:/data/drapenet/ \
	-v ${PW3D_DIR}:/data/3DPW/ \
	-v ${SSP3D_DIR}:/data/SSP-3D/ \
	-v ${TAILORNET_DATA_DIR}:/data/tailornet/ \
	-v ${HIERPROB3D_DATA_DIR}:/data/hierprob3d/ \
	-v ${CAT_DATA_DIR}:/data/cat/ \
	-v ${MGN_DATA_DIR}:/data/mgn/ \
	-v ${AGORA_DATA_DIR}:/data/agora/ \
	-v ${FRANKMOCAP_DATA_DIR}:/data/frankmocap/ \
	$USER-garmentor
