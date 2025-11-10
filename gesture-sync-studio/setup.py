"""
Setup script for Gesture Sync Studio training framework.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="gesture-sync-studio",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Audio-to-Gesture Animation System for Blender",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/gesture-sync-studio",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Multimedia :: Graphics :: 3D Modeling",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "librosa>=0.9.0",
        "scipy>=1.7.0",
        "torch>=2.0.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "training": [
            "tensorboard>=2.13.0",
            "onnx>=1.14.0",
            "onnxruntime>=1.15.0",
        ],
        "dev": [
            "pytest>=7.3.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "gesture-train=training.train:main",
            "gesture-export=training.export:main",
        ],
    },
    include_package_data=True,
    package_data={
        "blender_addon": ["*.json"],
        "training": ["*.yaml"],
    },
)
