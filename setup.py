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
    "scikit-learn>=1.1",
    "pandas>=2.2",
    "ConfigSpace<=0.6.1",
    "hpobench @ git+https://github.com/automl/HPOBench@0.0.10",
]

setup(
    name="pandora_bayesopt",
    version="1.0",
    description="Experiments for Bayesian optimization with Pandora's box",
    author="Qian Xie and collaborators",
    python_requires='>=3.9',
    packages=["pandora_bayesopt"],
    install_requires=requirements
)
