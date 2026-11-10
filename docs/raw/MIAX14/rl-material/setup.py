"""
Local installation (alternative to the Dev Container).

From the repo root, inside a Python 3.9-3.11 virtual environment (3.11 recommended):

    pip install -e .

This installs the dependencies below and makes everything under `src/` importable
from any notebook, e.g. `from src.utils.plots import plot_training_results`.
"""
from setuptools import find_packages, setup

setup(
    name="rl-material",
    version="1.0.0",
    description="RL material for BME Institute (MIAX)",
    author="Jesús Sanz del Real",
    # TensorFlow 2.15 only ships wheels for Python 3.9-3.11
    python_requires=">=3.9,<3.12",
    packages=find_packages(include=["src", "src.*"]),
    install_requires=[
        # Notebooks use the old gym API: reset() -> obs, step() -> (obs, r, done, info)
        "gym==0.23.1",
        # LunarLander physics. Not using gym[box2d] because box2d-py needs SWIG + a C++
        # compiler; Box2D has precompiled wheels for Windows, macOS and Linux
        "Box2D==2.3.10",
        # Rendering of CartPole / LunarLander in gym 0.23.1
        "pygame==2.6.1",
        # Last TF with Keras 2. Keras 3 (TF >= 2.16) breaks the dueling/rainbow networks
        "tensorflow==2.15.1",
        "numpy==1.26.4",
        "pandas==2.2.3",
        "matplotlib==3.9.4",
        "requests",
        "jupyterlab>=4,<5",
    ],
)
