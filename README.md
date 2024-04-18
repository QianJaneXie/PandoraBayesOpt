# Gittins Policy for Bayesian Optimization
This repository contains the implementation Gittins policy and its variants as well as the comparison against traditional policies such as Random Search, Thompson Sampling, Expected Improvement (EI), Expected Improvement Per Unit Cost (EIPC), Knowledge Gradient, (Budgeted) Multi-step Lookahead EI in the context of homogeneous-cost and heterogeneous-cost Bayesian Optimization.

## Setup
#### 1. Clone the GitHub repository
```
git clone https://github.com/QianJaneXie/PandoraBayesOpt.git
```

#### 2. Create a conda/homebrew virtual enviornment
```
conda create --name pandorabayesopt_env python=3.9
```
or
```
python3.9 -m venv pandorabayesopt_env
```

#### 3. Install required packages
```
conda activate pandorabayesopt_env
pip install -e .
```
or
```
source pandorabayesopt_env/bin/activate
pip install -e .
```
