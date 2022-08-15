# MAFNet-PyTorch
The official pytorch implementation of MAFNet: Segmentation of Road Potholes with Multi-modal Attention Fusion Network for Autonomous Vehicles. 

We test our code in Python 3.7, CUDA 11.1, cuDNN 8, and PyTorch 1.7.1. We provide `Dockerfile` to build the docker image we used. You can modify the `Dockerfile` as you want.  

# Introduction

# Dataset

# Pretrained weights

# Usage
* Clone this repo
```
$ git clone https://github.com/lab-sun/MAFNet.git
```
* Build docker image
```
$ cd ~/MAFNet
$ docker build -t docker_image_mafnet .
```
* Download the dataset
```
$ (You are in the MAFNet folder)
$ mkdir ./dataset
$ cd ./dataset
$ (download our preprocessed dataset.zip in this folder)
$ unzip -d .. dataset.zip
```
* To reproduce our results, you need to download our pretrained weights.
```
$ (You are in the MAFNet folder)
$ mkdir ./weights_backup/MAFNet
$ cd ./weights_backup/MAFNet
$ (download our preprocessed dataset.zip in this folder)
$ unzip -d .. dataset.zip
```





# acknowledgement
Some of the codes are borrowed from [RTFNet](https://github.com/yuxiangsun/RTFNet), [AARTFNet](https://github.com/hlwang1124/AAFramework) and [TransUNet](https://github.com/Beckschen/TransUNet)



Contact: yx.sun@polyu.edu.hk

Website: https://yuxiangsun.github.io/
