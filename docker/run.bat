@echo off
docker run -it -p 8888:8888 --rm --gpus all --shm-size=8gb --name julien-garmentor -v C:\Users\Julien\git\garmentor:/garmentor -v D:\data\3DPW:/data/3DPW -v D:\data\SSP-3D:/data/SSP-3D -v D:\data\TailorNet:/data/tailornet -v D:\data\HierProb3D:/data/hierprob3d -v D:\data\garmentor:/data/garmentor -v D:\data\mgn:/data/mgn -v D:\data\agora:/data/agora julien-garmentor