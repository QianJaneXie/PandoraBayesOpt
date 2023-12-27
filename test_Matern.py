#!/usr/bin/env python
# coding: utf-8

import os
seed = int(os.getenv('MY_SEED', '42'))  # Default to 42 if MY_SEED is not set
print(seed)

import torch

# use a GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set default tensor type to float64
torch.set_default_dtype(torch.float64)

# Set the seed for reproducibility
torch.manual_seed(seed)


from utils import create_objective_function, find_global_optimum
from botorch.acquisition import ExpectedImprovement
from acquisition import GittinsIndex
from bayesianoptimizer import BayesianOptimizer

import toml  # or use 'import json'

# Example Usage for 1D: 

# Create the objective model
dim = 2
nu = 0.5
lengthscale = 1.0
outputscale = 1.0
num_rff_features = 1280

maximize = False
num_iterations = 100  # Time budget

lmbda_values = [0.05, 0.01, 0.005, 0.001]
GI_best_history = {}
GI_regret_history = {}

objective = create_objective_function(dim, nu, lengthscale, outputscale, num_rff_features, seed=seed)
global_optimum = find_global_optimum(objective, dim, maximize)

# Run trial with ExpectedImprovement
EI_optimizer = BayesianOptimizer(objective=objective, dim=dim, maximize=maximize, seed=seed, nu=nu, lengthscale=lengthscale, outputscale=outputscale)
EI_optimizer.run(num_iterations=num_iterations, acquisition_function_class=ExpectedImprovement)
EI_best_history = EI_optimizer.get_best_history()
EI_regret_history = EI_optimizer.get_regret_history(global_optimum)

for lmbda in lmbda_values:
    print("lmbda:", lmbda)

    # Run trial with GittinsIndex for the current lambda
    GI_optimizer = BayesianOptimizer(objective=objective, dim=dim, maximize=maximize, seed=seed, nu=nu, lengthscale=lengthscale, outputscale=outputscale)
    GI_optimizer.run(num_iterations=num_iterations, acquisition_function_class=GittinsIndex, lmbda=lmbda)
    GI_best_history[str(lmbda)] = GI_optimizer.get_best_history()
    GI_regret_history[str(lmbda)] = GI_optimizer.get_regret_history(global_optimum)
    print()
print()

print("EI regret history:", EI_regret_history)
print("GI regret history:", GI_regret_history)

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the results directory relative to the script directory
results_dir = os.path.join(script_dir, 'results/Matern{}2/ls={}/{}D'.format(int(nu*2), lengthscale, dim))

# Create the results directory if it doesn't exist
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

result_filename = f'trial_{seed}.toml'
result_filepath = os.path.join(results_dir, result_filename)

def save_results_to_file(data, filepath):
    with open(filepath, 'w') as file:
        toml.dump(data, file)  # or use 'json.dump(data, file, indent=4)'

# Data to be saved
result_data = {
    'problem': {
        'type': 'Matern',
        'dim': dim,
        'nu': nu,
        'lengthscale': lengthscale,
        'outputscale': outputscale
    },
    'trial': seed,
    'global_optimum:': global_optimum,
    'num_iterations': num_iterations,
    'best_history': {
        'EI': EI_best_history,
        'Gittins': GI_best_history
    },
    'regret_history': {
        'EI': EI_regret_history,
        'Gittins': GI_regret_history
    }
}

# Save to file
save_results_to_file(result_data, result_filepath)