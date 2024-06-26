# Cost-aware Bayesian optimization via the Pandora's box Gittins index
This repository contains the implementation of the Pandora's box Gittins index (PBGI) policy and its variant PBGI-D. The policies are compared against various baselines in the context of uniform-cost and cost-aware Bayesian Optimization.

## Baselines
- Uniform-cost Bayesian optimization:
  - Random Search (RS)
  - Thompson Sampling (TS)
  - Upper Confidence Bound (UCB)
  - Expected Improvement (EI)
  - Knowledge Gradient (KG)
  - Multi-step Lookahead EI (MSEI)
- Cost-aware Bayesian Optimization
  - Expected Improvement Per Unit Cost (EIPC)
  - Budgeted Multi-step Lookahead EI (BMSEI)
 
## Contexts
- Experiments
  - Bayesian regret (fixed-amplitude)
  - Synthetic benchmark
  - Empirical objective
  - Timing
- Illustrations
  - Bayesian regret (variable-amplitude)
  - Impact of Lambda

## Setup
#### 1. Clone the GitHub repository
```
git clone https://github.com/QianJaneXie/PandoraBayesOpt.git
```

#### 2. Create a conda/homebrew virtual environment
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
