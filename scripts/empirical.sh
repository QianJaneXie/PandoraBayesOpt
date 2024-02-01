#!/bin/bash
#SBATCH -J empirical                         # Job name
#SBATCH -o empirical_%j.out                  # Output file (%j expands to jobID)
#SBATCH -e empirical_%j.err                  # Error log file (%j expands to jobID)
#SBATCH --mail-type=ALL                      # Request status by email 
#SBATCH --mail-user=qx66@cornell.edu         # Email address to send results to
#SBATCH -N 1                                 # Total number of nodes requested
#SBATCH -n 1                                 # Total number of cores requested
#SBATCH --array=0-220                        # Number of jobs
#SBATCH --mem-per-cpu=8000M                  # Server memory requested (per node)
#SBATCH -t 12:00:00                           # Time limit (hh:mm:ss)
#SBATCH --partition=default_partition        # Request partition
#SBATCH --ntasks-per-node=1                  # Number of tasks per node

source /share/apps/anaconda3/2021.05/etc/profile.d/conda.sh
conda activate myenv
wandb login
wandb agent 'ziv-scully-group/Gittins for Bayesian Optimization/dksankho' --count 1
conda deactivate
