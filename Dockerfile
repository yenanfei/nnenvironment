ARG PYTORCH="1.9.1"
ARG CUDA="11.1"
ARG CUDNN="8"
#ARG MMDET="2.24.0"
#ARG MMSEG="0.20.0"
#ARG MMCV="1.7.0"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
ENV FORCE_CUDA="1"

RUN sed -i "s@http://.*archive.ubuntu.com@https://mirrors.tuna.tsinghua.edu.cn@g" /etc/apt/sources.list
RUN sed -i "s@http://.*security.ubuntu.com@https://mirrors.tuna.tsinghua.edu.cn@g" /etc/apt/sources.list

# To fix GPG key error when running apt-get update
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 cmake libxrender-dev libxext6 htop psmisc nfs-common\
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

RUN pip install --no-cache-dir --upgrade pip setuptools==59.5.0 wheel setuptools tensorboardx nuscenes-devkit openmim mmdet mmsegmentation torch-scatter efficientnet_pytorch
RUN min install mmcv-full==1.5.0

RUN git clone https://github.com/open-mmlab/mmdetection3d.git
WORKDIR mmdetection3d
RUN pip install --no-cache-dir -e .