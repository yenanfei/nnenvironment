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

RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime
RUN echo 'Asia/Shanghai' >/etc/timezone

RUN sed -i "s@http://.*archive.ubuntu.com@https://mirrors.tuna.tsinghua.edu.cn@g" /etc/apt/sources.list
RUN sed -i "s@http://.*security.ubuntu.com@https://mirrors.tuna.tsinghua.edu.cn@g" /etc/apt/sources.list

# To fix GPG key error when running apt-get update
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update && apt-get install -y ffmpeg curl vim libglfw3 tmux libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 cmake libxrender-dev libxext6 htop psmisc nfs-common\
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

#RUN curl https://get.docker.com | sh
RUN pip config set global.index-url https://mirror.baidu.com/pypi/simple

COPY libstdc++.so.6.0.28  /lib/x86_64-linux-gnu/libstdc++.so.6.0.28
COPY onnxruntime_gpu-1.13.1-cp37-cp37m-linux_x86_64.whl .
COPY install.sh .
COPY libm-2.31.so /lib/x86_64-linux-gnu/libm.so.6
RUN pip install --no-cache-dir --upgrade pip pycuda scipy mmcv-full==1.4.0 trimesh scikit-image pyopengl geojson lyft-dataset-sdk numba==0.48.0 glumpy PyOpenGL_accelerate imgaug setuptools==59.5.0 wheel setuptools pymap3d rtree numpy-quaternion tensorboardx nuscenes-devkit openmim mmdet==2.14.0 mmsegmentation==0.14.1 torch-scatter efficientnet_pytorch

COPY mmdetection3d .
RUN python setup.py develop

RUN bash install.sh

RUN pip install onnxruntime_gpu-1.13.1-cp37-cp37m-linux_x86_64.whl

CMD [ "code-server","serve-local", "--bind-addr", "0.0.0.0:8888","--disable-telemetry","--auth","none"]