# # Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.

FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.13.1-gpu-py39-cu117-ubuntu20.04-sagemaker

LABEL authors="henanwan@amazon.com"

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    FORCE_CUDA="1"

# Avoid Public GPG key error
# https://github.com/NVIDIA/nvidia-docker/issues/1631
RUN rm -rf /etc/apt/sources.list.d/cuda.list \
    && rm -rf /etc/apt/sources.list.d/nvidia-ml.list \
    && apt-key del 7fa2af80 \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/7fa2af80.pub

# (Optional, use Mirror to speed up downloads)
# RUN sed -i 's/http:\/\/archive.ubuntu.com\/ubuntu\//http:\/\/opentuna.cn\/ubuntu\//g' /etc/apt/sources.list && \
#    pip config set global.index-url https://opentuna.cn/pypi/web/simple

# Install the required packages
RUN apt-get update \
    && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install MMEngine, MMCV and MMDetection
RUN pip install openmim && \
    mim install mmcv-full mmdet mmsegmentation

# Install MMDetection3D
RUN conda clean --all \
    && git clone https://github.com/open-mmlab/mmdetection3d.git -b dev-1.x /mmdetection3d \
    && cd /mmdetection3d \
    && pip install --no-cache-dir -e .

RUN mkdir /opt/ml/ && mkdir /opt/ml/code

COPY train.py /opt/ml/code/train.py

RUN wget https://download.openmmlab.com/mmdetection3d/v1.0.0_models/3dssd/3dssd_4x4_kitti-3d-car/3dssd_4x4_kitti-3d-car_20210818_203828-b89c8fc4.pth -O /opt/ml/code/3dssd_4x4_kitti-3d-car_20210818_203828-b89c8fc4.pth

COPY 3dssd_4x4_a2d2-3d-car.py /mmdetection3d/configs/3dssd/
COPY a2d2-3d-car.py /mmdetection3d/configs/_base_/datasets/
COPY a2d2_dataset.py /mmdetection3d/mmdet3d/datasets/
COPY custom_3d.py /mmdetection3d/mmdet3d/datasets/
RUN sed -i 's/import KittiDataset/import KittiDataset\nfrom .a2d2_dataset import A2D2Dataset/' /mmdetection3d/mmdet3d/datasets/__init__.py

ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code
ENV SAGEMAKER_PROGRAM /opt/ml/code/train.py

RUN export MASTER_ADDR=algo-1
RUN export MASTER_PORT=12345

WORKDIR /mmdetection3d

# Starts PyTorch distributed framework
ENTRYPOINT ["bash", "-m", "start_with_right_hostname.sh"]