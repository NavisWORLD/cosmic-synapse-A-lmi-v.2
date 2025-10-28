"""
Setup script for HRCS
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="hrcs",
    version="1.0.0",
    author="Cory Shane Davis",
    author_email="",
    description="Harmonic Resonance Communication System - Infrastructure-free mesh communication",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-repo/hrcs",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: System Administrators",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Communications",
        "Topic :: System :: Networking",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "sounddevice>=0.4.4",
        "cryptography>=3.4.8",
        "pyyaml>=5.4.1",
        "click>=8.0.0",
        "psutil>=5.8.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "sdr": [
            "soapysdr",
        ],
    },
    entry_points={
        "console_scripts": [
            "hrcs=hrcs.application.cli:cli",
        ],
    },
)

