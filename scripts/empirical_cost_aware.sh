#!/bin/bash
#SBATCH -J empirical_robot_pushing_14d_cost_aware                # Job name
#SBATCH -o empirical_robot_pushing_14d_cost_aware_%j.out         # Output file (%j expands to jobID)
#SBATCH -e empirical_robot_pushing_14d_cost_aware_%j.err         # Error log file (%j expands to jobID)
#SBATCH --mail-type=ALL                      # Request status by email 
#SBATCH --mail-user=qx66@cornell.edu         # Email address to send results to
#SBATCH -N 1                                 # Total number of nodes requested
#SBATCH -n 1                                 # Total number of cores requested
#SBATCH --array=0-79                        # Number of jobs
#SBATCH --mem-per-cpu=14G                  # Server memory requested (per node)
#SBATCH -t 280:00:00                           # Time limit (hh:mm:ss)
#SBATCH --partition=default_partition        # Request partition
#SBATCH --ntasks-per-node=1                  # Number of tasks per node

source /share/apps/anaconda3/2021.05/etc/profile.d/conda.sh
conda activate myenv
wandb login
wandb agent 'ziv-scully-group/PandoraBayesOPT/ue7d92uf' --count 1
conda deactivate
