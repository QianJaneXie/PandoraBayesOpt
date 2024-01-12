#!/bin/bash
#SBATCH -J test_cost_Matern                          # Job name
#SBATCH -o test_cost_Matern_%j.out                   # Output file (%j expands to jobID)
#SBATCH -e test_cost_Matern_%j.err                   # Error log file (%j expands to jobID)
#SBATCH --mail-type=ALL                      # Request status by email 
#SBATCH --mail-user=qx66@cornell.edu        # Email address to send results to
#SBATCH -N 1                                 # Total number of nodes requested
#SBATCH -n 1                                 # Total number of cores requested
#SBATCH --array=0-99
#SBATCH --mem-per-cpu=1000M                           # Server memory requested (per node)
#SBATCH -t 4:00:00                           # Time limit (hh:mm:ss)
#SBATCH --partition=default_partition        # Request partition
#SBATCH --ntasks-per-node=1                  # Number of tasks per node
#SBATCH --cpus-per-task=2                           # Number of cpus needed by each task (if task is "make -j3" number should be 3)

source /share/apps/anaconda3/2021.05/etc/profile.d/conda.sh
conda activate gittinsbayesopt_env_39
export MY_SEED=${SLURM_ARRAY_TASK_ID}
python /home/qx66/PandoraBayesOPT/test_cost_Matern.py                      # Path to your script
conda deactivate
