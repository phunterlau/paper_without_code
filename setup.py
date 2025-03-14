#!/usr/bin/env python3
"""
Setup script for PaperWoCode.
"""
from setuptools import setup, find_packages

setup(
    name="paperwocode",
    version="0.1.0",
    description="A tool to generate code prototypes from research papers",
    author="PaperWoCode Team",
    author_email="example@example.com",
    url="https://github.com/yourusername/paperwocode",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "markitdown[all]~=0.1.0a1",
        "requests",
        "anthropic",
        "openai",
        "PyPDF2",
        "tqdm",
        "numpy",
        "pandas",
        "matplotlib",
        "scikit-learn",
        "torch",
        "torchvision",
        "torchaudio",
        "seaborn",
        "pyyaml",
        "tiktoken",
    ],
    entry_points={
        "console_scripts": [
            "paperwocode=main:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.10",
)
