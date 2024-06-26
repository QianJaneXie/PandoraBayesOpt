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
- **Experiments**
  - Bayesian regret (fixed-amplitude)
  - Synthetic benchmark
  - Empirical
  - Timing
- **Illustrations**
  - EI/EIPC poor performance (via Bayesian-regret variable-amplitude)
  - Impact of lambda
 
## Instruction
All experiments are run via Weights & Biases (wandb) sweeps. Follow these steps to execute the experiments:

1. **Prepare configuration file**: Each experiment requires a configuration file. These files can be found in the folder `scripts/config`. Select the appropriate configuration file for the experiment you want to run.
   
2. **Launch wandb agent**: The wandb agent is used to execute the sweeps. Launch the agent using the following command:
   ```sh
   wandb agent <sweep_id>

3. **Run Bayesian optimization**: The Bayesian optimization process is executed with the hyperparameters specified in the configuration file. This is handled by the `run_bayesopt_experiment` function found in each script within the `scripts` folder. Make sure to review and understand the specific script you are running.

4. **Log cata to wandb**: During the experiment, all relevant data and metrics are logged to wandb for tracking and analysis.

5. **Review results**: After the experiments are complete, review the results logged in your wandb dashboard.

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
