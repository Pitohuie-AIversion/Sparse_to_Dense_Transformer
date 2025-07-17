#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VIVTransformer Setup Script

A comprehensive framework for testing and comparing 38 different attention mechanisms
in transformer architectures.
"""

import os
import sys
from pathlib import Path
from setuptools import setup, find_packages

# 确保Python版本兼容性
if sys.version_info < (3, 8):
    raise RuntimeError("VIVTransformer requires Python 3.8 or higher")

# 获取项目根目录
HERE = Path(__file__).parent.absolute()

# 读取README文件
def read_readme():
    """读取README.md文件内容"""
    readme_path = HERE / "README.md"
    if readme_path.exists():
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return "A comprehensive framework for testing attention mechanisms in transformers."

# 读取requirements文件
def read_requirements(filename="requirements.txt"):
    """读取requirements文件"""
    req_path = HERE / filename
    if req_path.exists():
        with open(req_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return []

# 获取版本信息
def get_version():
    """从__init__.py文件中获取版本信息"""
    version_file = HERE / "vivtransformer" / "__init__.py"
    if version_file.exists():
        with open(version_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("__version__"):
                    return line.split("=")[1].strip().strip('"').strip("'")
    return "1.0.0"

# 项目元数据
NAME = "vivtransformer"
DESCRIPTION = "A comprehensive framework for testing and comparing 38 different attention mechanisms in transformer architectures"
URL = "https://github.com/yourusername/VIVTransformer"
EMAIL = "your.email@example.com"
AUTHOR = "VIVTransformer Contributors"
REQUIRES_PYTHON = ">=3.8.0"
VERSION = get_version()

# 必需的依赖
REQUIRED = [
    "torch>=1.9.0",
    "numpy>=1.21.0",
    "pyyaml>=5.4.0",
    "matplotlib>=3.3.0",
    "fightingcv_attention>=0.0.9",
    "tqdm>=4.62.0",
    "tensorboard>=2.7.0",
    "scikit-learn>=1.0.0",
    "pandas>=1.3.0",
    "seaborn>=0.11.0",
]

# 可选依赖
EXTRAS = {
    "dev": [
        "pytest>=6.2.0",
        "pytest-cov>=2.12.0",
        "black>=21.0.0",
        "isort>=5.9.0",
        "flake8>=3.9.0",
        "mypy>=0.910",
        "pre-commit>=2.15.0",
    ],
    "docs": [
        "sphinx>=4.0.0",
        "sphinx-rtd-theme>=0.5.0",
        "sphinx-autodoc-typehints>=1.12.0",
        "myst-parser>=0.15.0",
    ],
    "jupyter": [
        "jupyter>=1.0.0",
        "ipywidgets>=7.6.0",
        "plotly>=5.0.0",
    ],
    "wandb": [
        "wandb>=0.12.0",
    ],
    "mlflow": [
        "mlflow>=1.20.0",
    ],
    "optuna": [
        "optuna>=2.10.0",
    ],
}

# 添加"all"选项，包含所有可选依赖
EXTRAS["all"] = list(set(sum(EXTRAS.values(), [])))

# 分类器
CLASSIFIERS = [
    # 开发状态
    "Development Status :: 4 - Beta",
    
    # 目标受众
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "Intended Audience :: Education",
    
    # 主题
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Topic :: Education",
    
    # 许可证
    "License :: OSI Approved :: MIT License",
    
    # Python版本支持
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3 :: Only",
    
    # 操作系统
    "Operating System :: OS Independent",
    "Operating System :: POSIX",
    "Operating System :: Microsoft :: Windows",
    "Operating System :: MacOS",
    
    # 环境
    "Environment :: Console",
    "Environment :: GPU :: NVIDIA CUDA",
]

# 关键词
KEYWORDS = [
    "transformer",
    "attention",
    "deep learning",
    "machine learning",
    "pytorch",
    "neural networks",
    "artificial intelligence",
    "research",
    "benchmark",
    "comparison",
]

# 项目URL
PROJECT_URLS = {
    "Homepage": URL,
    "Documentation": f"{URL}/wiki",
    "Source": URL,
    "Tracker": f"{URL}/issues",
    "Changelog": f"{URL}/blob/main/CHANGELOG.md",
    "Contributing": f"{URL}/blob/main/CONTRIBUTING.md",
}

# 入口点
ENTRY_POINTS = {
    "console_scripts": [
        "vivtransformer=main:main",
        "viv-train=main:main",
        "viv-test=scripts.test:main",
        "viv-benchmark=scripts.benchmark:main",
    ],
}

# 包数据
PACKAGE_DATA = {
    "vivtransformer": [
        "configs/*.yaml",
        "configs/loss_configs/*.yaml",
        "data/samples/*",
        "templates/*",
    ],
}

# 数据文件
DATA_FILES = [
    ("configs", ["configs/config.yaml"]),
    ("docs", ["README.md", "CHANGELOG.md", "CONTRIBUTING.md", "LICENSE"]),
]

if __name__ == "__main__":
    setup(
        name=NAME,
        version=VERSION,
        description=DESCRIPTION,
        long_description=read_readme(),
        long_description_content_type="text/markdown",
        author=AUTHOR,
        author_email=EMAIL,
        python_requires=REQUIRES_PYTHON,
        url=URL,
        project_urls=PROJECT_URLS,
        
        # 包信息
        packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
        package_data=PACKAGE_DATA,
        data_files=DATA_FILES,
        include_package_data=True,
        
        # 依赖
        install_requires=REQUIRED,
        extras_require=EXTRAS,
        
        # 入口点
        entry_points=ENTRY_POINTS,
        
        # 元数据
        classifiers=CLASSIFIERS,
        keywords=" ".join(KEYWORDS),
        license="MIT",
        
        # 其他选项
        zip_safe=False,
        platforms=["any"],
        
        # 测试
        test_suite="tests",
        tests_require=EXTRAS["dev"],
        
        # 命令类
        cmdclass={},
    )