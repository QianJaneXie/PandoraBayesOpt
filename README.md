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

The implementation of PBGI, EIPC, MSEI, and BMSEI can be found in `pandora_bayesopt/acquisition`.
 
## Contexts
- **Experiments**
  - Bayesian regret (fixed-amplitude)
  - Synthetic benchmark
  - Empirical
  - Timing
- **Illustrations**
  - EI/EIPC poor performance (via Bayesian-regret variable-amplitude)
  - Impact of lambda
 
The empirical objective functions we used in our experiments can be found in `pandora_bayesopt/test_functions`.
 
## Execution
All our experiments were run using Weights & Biases (wandb) for tracking and analysis. Alternatively, you can run the experiments using Python scripts without wandb. Follow these steps to execute the experiments:

1. **Prepare configuration file**: Each of our experiment takes a wandb configuration file as an input. These configuration files can be found in the folder scripts/config, which lists all the hyperparameter choices for a wandb sweep. This configuration file is not necessary if you choose to run the experiments directly using Python scripts.
   
2. **Run Experiment**: You have two options to run the experiments:
   - **Using wandb sweep**: If you wish to use wandb for tracking and analysis, launch the wandb agent using the following command to run a sweep of hyperparameter choices:
     ```sh
     wandb agent <sweep_id>
     ```
   - **Using Python Scripts Directly**: Execute the experiments by running the appropriate Python script. Each script can be found in the `scripts` folder and includes a function named `run_bayesopt_experiment`. If you choose to run the Python scripts directly, replace or comment out the lines involving wandb operations. For example:
     ```sh
     python scripts/<script_name>.py
     ```

3. **Handle Hyperparameters**: 
   The Bayesian optimization process is executed with the hyperparameters specified in the configuration file. This is handled by the `run_bayesopt_experiment` function found in each script within the `scripts` folder. If you choose to run the experiments using Python scripts directly, change the input of `run_bayesopt_experiment` from a configuration to a set of hyperparameters.

3. **Log data**: Whether using wandb or just Python scripts, ensure that all relevant data and metrics are logged for analysis.

4. **Review results**: After the experiments are complete, review the results either in your local environment or on your wandb dashboard if you used wandb.

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
