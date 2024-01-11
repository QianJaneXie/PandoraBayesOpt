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

from pandora_bayesopt.utils import create_objective_function, find_global_optimum
from botorch.acquisition import ExpectedImprovement
from pandora_bayesopt.acquisition import GittinsIndex, ExpectedImprovementWithCost
from pandora_bayesopt.bayesianoptimizer import BayesianOptimizer

import json # temporary

# Example Usage for 1D: 

# Create the objective model
dim = 2
nu = 0.5
lengthscale = 1/16
outputscale = 1.0
num_rff_features = 1280

maximize = False
budget = 80  # Cost budget

import time
start = time.time()
objective = create_objective_function(dim, nu, lengthscale, outputscale, num_rff_features)
end = time.time()
print("creating objective time:", end-start)

start = time.time()
global_optimum = find_global_optimum(objective, dim, maximize)
print("global optimum:", global_optimum)
end = time.time()
print("finding global optimum time:", end-start)

def cost_function(x):
    return 0.1+x.sum(dim=-1)

# Run trial with ExpectedImprovement
print("EI")
start = time.time()
EI_optimizer = BayesianOptimizer(objective=objective, dim=dim, maximize=maximize, cost=cost_function, nu=nu, lengthscale=lengthscale, outputscale=outputscale)
EI_optimizer.run_until_budget(budget=budget, acquisition_function_class=ExpectedImprovement)
EI_cost_history = EI_optimizer.get_cost_history()
EI_best_history = EI_optimizer.get_best_history()
EI_regret_history = EI_optimizer.get_regret_history(global_optimum)
end = time.time()
print("EI time:", end-start)

print("EI cost history:", EI_cost_history)
print("EI regret history:", EI_regret_history)
print()

# Run trial with ExpectedImprovementWithCost
print("EIpu")
start = time.time()
EIpu_optimizer = BayesianOptimizer(objective=objective, dim=dim, maximize=maximize, cost=cost_function, nu=nu, lengthscale=lengthscale, outputscale=outputscale)
EIpu_optimizer.run_until_budget(budget=budget, acquisition_function_class=ExpectedImprovementWithCost)
EIpu_cost_history = EIpu_optimizer.get_cost_history()
EIpu_best_history = EIpu_optimizer.get_best_history()
EIpu_regret_history = EIpu_optimizer.get_regret_history(global_optimum)
end = time.time()
print("EIpu time:", end-start)

print("EIpu cost history:", EIpu_cost_history)
print("EIpu regret history:", EIpu_regret_history)
print()

# Run trial with GittinsIndex for the current lambda
print("GI with lmbda")
lmbda_values = [0.05, 0.01, 0.005, 0.001]
GI_best_history = {}
GI_cost_history = {}
GI_regret_history = {}

for lmbda in lmbda_values:
    print("lmbda:", lmbda)
    # Run trial with GittinsIndex for the current lambda
    GI_optimizer = BayesianOptimizer(objective=objective, dim=dim, maximize=maximize, cost=cost_function, nu=nu, lengthscale=lengthscale, outputscale=outputscale)
    GI_optimizer.run_until_budget(budget=budget, acquisition_function_class=GittinsIndex, lmbda=lmbda)
    GI_cost_history[str(lmbda)] = GI_optimizer.get_cost_history()
    GI_best_history[str(lmbda)] = GI_optimizer.get_best_history()
    GI_regret_history[str(lmbda)] = GI_optimizer.get_regret_history(global_optimum)
    print("GI cost history:", GI_cost_history[str(lmbda)])
    print("GI regret history:", GI_regret_history[str(lmbda)])
    print()
print()

# Run trial with Hyperparameter-free GittinsIndex
print("Hyperparameter-free GI")
start = time.time()
GI_optimizer = BayesianOptimizer(objective=objective, dim=dim, maximize=maximize, cost=cost_function, nu=nu, lengthscale=lengthscale, outputscale=outputscale)
GI_optimizer.run_until_budget(budget=budget, acquisition_function_class=GittinsIndex)
GI_cost_history["EIpu_max/2"] = GI_optimizer.get_cost_history()
GI_best_history["EIpu_max/2"] = GI_optimizer.get_best_history()
GI_regret_history["EIpu_max/2"] = GI_optimizer.get_regret_history(global_optimum)
end = time.time()
print()

print("GI time:", end - start)
print()

print("GI cost history:", GI_cost_history["EIpu_max/2"])
print("GI regret history:", GI_regret_history["EIpu_max/2"])

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the results directory relative to the script directory
results_dir = os.path.join(script_dir, 'results/hetero/linear/Matern{}2/ls={}/{}D'.format(int(nu*2), lengthscale, dim))

# Create the results directory if it doesn't exist
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

result_filename = f'trial_{seed}.toml'
result_filepath = os.path.join(results_dir, result_filename)

def save_results_to_file(data, filepath):
    with open(filepath, 'w') as file:
        json.dump(data, file, indent=4)

# Data to be saved
result_data = {
    'problem': {
        'objective': 'Matern',
        'dim': dim,
        'nu': nu,
        'lengthscale': lengthscale,
        'outputscale': outputscale,
        "cost": '0.1+x'
    },
    'trial': seed,
    'global_optimum': global_optimum,
    'budget': budget,
    'cost_history': {
        'EI': EI_cost_history,
        'EIpu': EIpu_cost_history,
        'Gittins': GI_cost_history
    },
    'best_history': {
        'EI': EI_best_history,
        'EIpu': EIpu_best_history,
        'Gittins': GI_best_history
    },
    'regret_history': {
        'EI': EI_regret_history,
        'EIpu': EIpu_regret_history,
        'Gittins': GI_regret_history
    }
}

# Save to file
save_results_to_file(result_data, result_filepath)