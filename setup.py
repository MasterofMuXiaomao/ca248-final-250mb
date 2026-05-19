#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CA-248 250MB最终版 - 安装配置"""

from setuptools import setup, find_packages
import os

# 读取README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# 读取requirements
with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="ca248-final-250mb",
    version="1.0.0",
    author="MasterofMuXiaomao (沐小卯)",
    author_email="ca248@openclaw.ai",
    description="CA-248 250MB最终版 - 移动优化的248维智能实体架构",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MasterofMuXiaomao/ca248-final-250mb",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "mobile": [
            "onnxruntime>=1.15.0",
            "onnx>=1.14.0",
        ],
        "full": [
            "torch>=1.13.0",
            "transformers>=4.25.0",
            "numpy>=1.21.0",
            "scipy>=1.7.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ca248-demo=ca248_final:demo_main",
            "ca248-benchmark=ca248_final:benchmark_main",
        ],
    },
    include_package_data=True,
    keywords=[
        "ai",
        "artificial-intelligence",
        "cognitive-architecture",
        "mobile-ai",
        "248-dimensional",
        "e8-symmetry",
        "categorical-attention",
    ],
    project_urls={
        "Bug Reports": "https://github.com/MasterofMuXiaomao/ca248-final-250mb/issues",
        "Source": "https://github.com/MasterofMuXiaomao/ca248-final-250mb",
        "Documentation": "https://github.com/MasterofMuXiaomao/ca248-final-250mb#readme",
    },
)