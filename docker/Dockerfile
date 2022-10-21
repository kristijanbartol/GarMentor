
#FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel
#FROM pytorch/pytorch:1.11.0-cuda10.1-cudnn8-devel
#FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel

# needed to allow apt update
#RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC

#RUN apt update && apt install -y git \
#	ninja-build \
#	libglib2.0-0 \
#	libsm6 \
#	libxrender-dev \
#	libxext6 \
#	libgl1-mesa-glx \
#	vim \
#	curl \
#	libboost-dev \
#	tmux \
#	freeglut3-dev

#ENV BOOST_INCLUDE_DIRS=/usr/include/boost

# install python dependencies
#RUN pip install --ignore-installed PyYAML \
#	opencv-python \
#	tqdm \
#	matplotlib \
#	dataclasses \
#	#pyyaml \
#	yacs \
#	fvcore==0.1.2.post20201213 \
#	scikit-image==0.17.2 \
#	scipy==1.5.4 \
#	smplx==0.1.26 \
#	trimesh \
#	scikit-learn \
#	tensorboard \
#	chumpy \
#	tensorboardX \
#	"protobuf<4.21.0" \
#	lmdb \
#	visdom \
#	notebook \
#	pyrender \
#	pytorch3d==0.3.0

# pytorch3d prebuilt wheel
#RUN pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu113_pyt1110/download.html

# psbody.mesh installation
#RUN mkdir /git && cd /git && git clone https://github.com/MPI-IS/mesh && cd mesh && make all

# clone SMPL-X code and make tools/clean_ch.py usable with Python 3
#RUN cd /git && git clone https://github.com/vchoutas/smplx && cd smplx && git reset --hard 5fa20519735cceda19afed0beeabd53caef711cd && cd tools && \
#	sed -i "/body_data = / s/pickle.load(body_file)/pickle.load(body_file, encoding='latin1')/" clean_ch.py && \
#	sed -i "s/iteritems/items/g" clean_ch.py

#RUN mkdir -p /garmentor

#WORKDIR /garmentor

# Original Dockerfile content:

FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel

ENV PATH=$PATH:/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/cuda/bin/
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64/

ENV CUDA_HOME=/usr/local/cuda/
ENV CUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

ENV C_INCLUDE_PATH=$C_INCLUDE_PATH:$CUDA_HOME/targets/
ENV CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:$CUDA_HOME/targets/
ENV C_INCLUDE_PATH=$C_INCLUDE_PATH:$CUDA_HOME/targets/x86_64-linux/include/
ENV CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:$CUDA_HOME/targets/x86_64-linux/include/

ENV BOOST_INCLUDE_DIRS=/usr/include/boost

ENV FORCE_CUDA="1"

COPY . /garmentor/

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC

RUN apt-get update && apt-get install -y git \
	ninja-build \
	libglib2.0-0 \
 	libsm6 \
 	libxrender-dev \
 	libxext6 \
 	libgl1-mesa-glx \
	libboost-dev \
 	python3-pip \
 	vim \
 	curl \
     && apt-get clean \
     && rm -rf /var/lib/apt/lists/*

RUN python3.7 -m pip install --ignore-installed PyYAML \
 	torch==1.6.0 \
 	torchvision==0.7.0 \
 	opencv-python \
 	tqdm \
 	matplotlib \
 	dataclasses \
 	pyyaml \
 	yacs \
 	fvcore==0.1.2.post20201213 \
 	scikit-image==0.17.2 \
 	scipy==1.5.4 \
 	smplx==0.1.26 \
 	trimesh \
	tensorboardX \
 	scikit-learn \
 	tensorboard \
	pytorch3d==0.3.0 \
	ipython \
	markupsafe==2.0.1 \
	traitlets==5.1.1 \
	pygments==2.4.1 \
	jupyter \
	chumpy \
	visdom \
	open3d \
	loguru \
	omegaconf

RUN python3.7 -m pip install --upgrade ipykernel

# psbody.mesh installation
RUN mkdir /git && cd /git && git clone https://github.com/MPI-IS/mesh && cd mesh && make all

# clone SMPL-X code and make tools/clean_ch.py usable with Python 3
RUN cd /git && git clone https://github.com/vchoutas/smplx && cd smplx && git reset --hard 5fa20519735cceda19afed0beeabd53caef711cd && cd tools && \
	sed -i "/body_data = / s/pickle.load(body_file)/pickle.load(body_file, encoding='latin1')/" clean_ch.py && \
	sed -i "s/iteritems/items/g" clean_ch.py

WORKDIR /

RUN curl -LO https://github.com/NVIDIA/cub/archive/1.10.0.tar.gz && \
 	tar xzf 1.10.0.tar.gz 

ENV CUB_HOME=/cub-1.10.0

#RUN python3.7 -m pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.3.0"

WORKDIR /garmentor
