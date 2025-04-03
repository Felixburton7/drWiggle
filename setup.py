from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
readme_path = os.path.join(this_directory, 'README.md')
long_description = ""
if os.path.exists(readme_path):
    with open(readme_path, encoding='utf-8') as f:
        long_description = f.read()
else:
    print(f"Warning: README.md not found at {readme_path}")


setup(
    name="drwiggle",
    version="1.0.0", # Updated version
    author="AI Assistant (via Prompt)", # Please Change
    author_email="your.email@example.com", # Please Change
    description="Protein Flexibility Classification Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/drwiggle", # Please Change
    # find_packages() will find the 'drwiggle' source directory inside the current dir
    packages=find_packages(where='.'),
    package_dir={'': '.'}, # The 'drwiggle' package is in the current dir
    include_package_data=True, # To include non-code files like default_config.yaml
    install_requires=[
        "numpy>=1.20",
        "pandas>=1.3",
        "scikit-learn>=1.1", # Ensure version supports weighted kappa etc.
        "pyyaml>=6.0",
        "click>=8.0",
        "matplotlib>=3.5",
        "seaborn>=0.11",
        "joblib>=1.0",
        "tqdm>=4.60",
        "torch>=1.10", # Specify version compatible with your CUDA/MPS if needed
        "optuna>=2.10",
        "biopython>=1.79",
    ],
    entry_points={
        "console_scripts": [
            "drwiggle=drwiggle.cli:cli",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License", # Choose your license
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.8',
    package_data={
        # Tells setuptools to include default_config.yaml when the package is installed
        'drwiggle': ['default_config.yaml'],
    },
)
