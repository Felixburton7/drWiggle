#!/usr/bin/env python3

from setuptools import setup, find_packages

setup(
    name="mdcath-processor",
    version="0.1.0",
    description="Processing pipeline for mdCATH molecular dynamics dataset",
    author="Felix Burton",
    author_email="info@drfelix.org",
    url="https://github.com/drfelix/mdcath-processor",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "mdcath=mdcath.cli:main",
        ],
    },
    install_requires=[
        "h5py>=3.7.0",
        "numpy>=1.22.0",
        "pandas>=1.5.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.12.0",
        "pyyaml>=6.0",
        "tqdm>=4.64.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Chemistry",
    ],
    python_requires=">=3.9",
)