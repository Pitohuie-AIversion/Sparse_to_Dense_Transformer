# VIVTransformer Dockerfile
# 多阶段构建，支持开发和生产环境

# 基础镜像阶段
FROM python:3.9-slim as base

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# 创建非root用户
RUN groupadd --gid 1000 vivuser && \
    useradd --uid 1000 --gid vivuser --shell /bin/bash --create-home vivuser

# 设置工作目录
WORKDIR /app

# 复制requirements文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# 开发环境阶段
FROM base as development

# 安装开发依赖
COPY requirements-dev.txt* ./
RUN if [ -f requirements-dev.txt ]; then pip install -r requirements-dev.txt; fi

# 安装Jupyter和其他开发工具
RUN pip install jupyter ipywidgets plotly

# 复制项目文件
COPY --chown=vivuser:vivuser . .

# 切换到非root用户
USER vivuser

# 暴露端口
EXPOSE 8888 6006

# 开发环境启动命令
CMD ["bash"]

# 生产环境阶段
FROM base as production

# 复制项目文件
COPY --chown=vivuser:vivuser . .

# 安装项目
RUN pip install -e .

# 切换到非root用户
USER vivuser

# 健康检查
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "from models.attention import ATTENTION_REGISTRY; print(f'{len(ATTENTION_REGISTRY)} attention mechanisms available')" || exit 1

# 生产环境启动命令
CMD ["python", "main.py"]

# GPU支持阶段
FROM nvidia/cuda:11.8-devel-ubuntu20.04 as gpu-base

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda

# 安装Python和系统依赖
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3-pip \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# 创建python符号链接
RUN ln -s /usr/bin/python3.9 /usr/bin/python

# 创建非root用户
RUN groupadd --gid 1000 vivuser && \
    useradd --uid 1000 --gid vivuser --shell /bin/bash --create-home vivuser

# 设置工作目录
WORKDIR /app

# 复制requirements文件
COPY requirements.txt .

# 安装PyTorch GPU版本
RUN pip install --upgrade pip && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
    pip install -r requirements.txt

# GPU生产环境
FROM gpu-base as gpu-production

# 复制项目文件
COPY --chown=vivuser:vivuser . .

# 安装项目
RUN pip install -e .

# 切换到非root用户
USER vivuser

# 健康检查
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); from models.attention import ATTENTION_REGISTRY; print(f'{len(ATTENTION_REGISTRY)} attention mechanisms available')" || exit 1

# GPU环境启动命令
CMD ["python", "main.py"]

# 最终阶段选择
FROM production as final