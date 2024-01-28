from setuptools import setup

requirements = [
    "numpy>=1.16",
    "scipy>=1.3",
    "torch>=2.1.1",
    "gpytorch>=1.11",
    "botorch>=0.8",
    "wandb>=0.16",
    "matplotlib>=3.7",
    "tqdm>=4.0",
    "pandas>=1.0",
    "notebook>=6.0"
    "ipywidgets>=8.1.1"
]

setup(
    name="pandora_bayesopt",
    version="1.0",
    description="Experiments for Bayesian optimization with Pandora's box",
    author="Qian Xie and collaborators",
    python_requires='>=3.11',
    packages=["pandora_bayesopt"],
    install_requires=requirements
)
