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

from src.utils import create_objective_function, find_global_optimum
from botorch.acquisition import ExpectedImprovement
from src.acquisition import GittinsIndex
from src.bayesianoptimizer import BayesianOptimizer

import toml  # or use 'import json'

# Example Usage for 1D: 

# Create the objective model
dim = 3
nu = 1.5
lengthscale = 1/16
outputscale = 1.0
num_rff_features = 1280

maximize = False
num_iterations = 120  # Time budget

lmbda_values = [0.005, 0.001, 0.0005]
GI_best_history = {}
GI_regret_history = {}

import time
start = time.time()
objective = create_objective_function(dim, nu, lengthscale, outputscale, num_rff_features, seed=seed)
end = time.time()
print("creating objective time:", end-start)

start = time.time()
global_optimum = find_global_optimum(objective, dim, maximize)
print("global optimum:", global_optimum)
end = time.time()
print("finding global optimum time:", end-start)

# Run trial with ExpectedImprovement
print("EI")
start = time.time()
EI_optimizer = BayesianOptimizer(objective=objective, dim=dim, maximize=maximize, seed=seed, nu=nu, lengthscale=lengthscale, outputscale=outputscale)
EI_optimizer.run(num_iterations=num_iterations, acquisition_function_class=ExpectedImprovement)
EI_best_history = EI_optimizer.get_best_history()
EI_regret_history = EI_optimizer.get_regret_history(global_optimum)
end = time.time()
print("EI time:", end-start)
print("EI regret history:", EI_regret_history)


# Run trial with GittinsIndex for specified lambda
print("GI with lmbda")
for lmbda in lmbda_values:
    print("lmbda:", lmbda)
    start = time.time()
    # Run trial with GittinsIndex for the current lambda
    GI_optimizer = BayesianOptimizer(objective=objective, dim=dim, maximize=maximize, seed=seed, nu=nu, lengthscale=lengthscale, outputscale=outputscale)
    GI_optimizer.run(num_iterations=num_iterations, acquisition_function_class=GittinsIndex, lmbda=lmbda)
    GI_best_history[str(lmbda)] = GI_optimizer.get_best_history()
    GI_regret_history[str(lmbda)] = GI_optimizer.get_regret_history(global_optimum)
    end = time.time()
    print("GI time:", end - start)
    print("GI regret history:", GI_regret_history[str(lmbda)])
    print()
print()

# Run trial with Hyperparameter-free GittinsIndex
print("Hyperparameter-free GI")
start = time.time()
GI_optimizer = BayesianOptimizer(objective=objective, dim=dim, maximize=maximize, seed=seed, nu=nu, lengthscale=lengthscale, outputscale=outputscale)
GI_optimizer.run(num_iterations=num_iterations, acquisition_function_class=GittinsIndex)
GI_best_history["EIpu_max/2"] = GI_optimizer.get_best_history()
GI_regret_history["EIpu_max/2"] = GI_optimizer.get_regret_history(global_optimum)
end = time.time()
print("GI time:", end - start)
print("GI regret history:", GI_regret_history["EIpu_max/2"])
print()


# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the results directory relative to the script directory
results_dir = os.path.join(script_dir, 'results/homo/Matern{}2/ls={}/{}D'.format(int(nu*2), lengthscale, dim))

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