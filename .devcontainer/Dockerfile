FROM pytorch/pytorch:2.10.0-cuda13.0-cudnn9-devel

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive\
    TORCH_CUDA_ARCH_LIST="12.0"

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl htop libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace