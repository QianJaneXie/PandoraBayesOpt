#!/usr/bin/env python
# coding: utf-8

# # BayesOPT Example for showing Gittins >> EIpu
# Extension of the numerical examples presented in Theorem 1 of Raul and Peter's paper which aims to show the limitation of EIpu and EI. The experiment extends the scope from Pandora's box (discrete finite points) to Bayesian optimization (continuous domain) and compares Gittins with EIpu/EI.

import torch
from pandora_bayesopt.utils import fit_gp_model, create_objective_function, find_global_optimum
from gpytorch.kernels import MaternKernel
from botorch.utils.gp_sampling import get_deterministic_model
from pandora_bayesopt.kernel import VariableAmplitudeKernel
from botorch.acquisition import ExpectedImprovement
from pandora_bayesopt.acquisition import ExpectedImprovementWithCost, GittinsIndex
from pandora_bayesopt.bayesianoptimizer import BayesianOptimizer

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

# use a GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set default tensor type to float64
torch.set_default_dtype(torch.float64)


# ## Define the amplitude function and the cost function 
# The continuous amplitude function and the continuous cost function are constructed based on the variances and costs of the discrete finite points provided in the original example

# Define typical small values for epsilon and delta, and a moderate value for K
epsilon = 0.1
delta = 0.05
K = 100  # Number of points excluding the central point

# Define the functions for the amplitude and the cost
def amplitude_function(x):
    width = 1.0 / K  # Width of the bump to cover only the central point
    amplitude = torch.exp(-((x - 0.5)**2) / (2 * width**2)) * (1 - epsilon**2) + epsilon**2
    return amplitude.squeeze(-1)

def cost_function(x):
    width = 1.0 / K  # Width of the bump to cover only the central point
    peak_height = 1 + delta - epsilon
    cost = torch.exp(-((x - 0.5)**2) / (2 * width**2)) * peak_height + epsilon
    return cost.squeeze(-1)

# ## Define the objective function
# The objective functions are constructed as sample paths drawn from the Matern kernel multiplied by the amplitude function

# Create the objective model
dim = 1
nu = 0.5
lengthscale = 0.01
outputscale = 1.0
num_rff_features = 1280
seed = 42
torch.manual_seed(seed)
print("seed:", seed)

# Create the objective function
matern_sample = create_objective_function(
    dim=dim, 
    nu=nu, 
    lengthscale=lengthscale,
    outputscale=outputscale,
    num_rff_features=num_rff_features
)

def objective_function(x):
    return matern_sample(x) * amplitude_function(x)

# ## Test performance of different policies

maximize = True
dim = 1
budget = 3.0

global_optimum_point, global_optimum_value = find_global_optimum(objective=objective_function, dim=dim, maximize=maximize)
print("global_optimum", global_optimum_point, global_optimum_value)
print()

# Plot for scaled objective function
test_x = torch.linspace(0, 1, 3001, dtype=torch.float64, device=device)
plt.plot(test_x.cpu().numpy(), objective_function(test_x.view(-1,1)).numpy(), color='tab:grey', label="Scaled objective function", alpha=0.6)
plt.plot(test_x.cpu().numpy(), cost_function(test_x.view(-1,1)).numpy(), label="Cost function", alpha=0.6)
plt.plot(global_optimum_point.cpu().numpy(), global_optimum_value, 'r*', label="global_optimum", alpha=0.8)
plt.title(f"Scaled objective function and cost function")
plt.xlabel("x")
plt.grid(True)
plt.show()
plt.close()

# init_x = torch.zeros(dim).unsqueeze(1)

# # Set up the kernel
# base_kernel = MaternKernel(nu=nu).double()
# base_kernel.lengthscale = torch.tensor([[lengthscale]], dtype=torch.float64)
# kernel = VariableAmplitudeKernel(base_kernel, amplitude_function)

# # Test EI policy
# print("EI")
# EI_optimizer = BayesianOptimizer(objective=objective_function, dim=dim, maximize=maximize, initial_points=init_x, kernel=kernel, cost=cost_function)
# EI_optimizer.run_until_budget(budget=budget, acquisition_function_class=ExpectedImprovement)
# EI_cost_history = EI_optimizer.get_cost_history()
# EI_best_history = EI_optimizer.get_best_history()
# EI_regret_history = EI_optimizer.get_regret_history(global_optimum)

# print("EI cost history:", EI_cost_history)
# print("EI best history:", EI_best_history)
# print("EI regret history:", EI_regret_history)
# print()


# # Test EI per unit cost policy
# print("EIpu")
# EIpu_optimizer = BayesianOptimizer(objective=objective_function, dim=dim, maximize=maximize, initial_points=init_x, kernel=kernel, cost=cost_function)
# EIpu_optimizer.run_until_budget(budget=budget, acquisition_function_class=ExpectedImprovementWithCost)
# EIpu_cost_history = EIpu_optimizer.get_cost_history()
# EIpu_best_history = EIpu_optimizer.get_best_history()
# EIpu_regret_history = EIpu_optimizer.get_regret_history(global_optimum)

# print("EIpu cost history:", EIpu_cost_history)
# print("EIpu best history:", EIpu_best_history)
# print("EIpu regret history:", EIpu_regret_history)
# print()

# Test EI with cost-cooling policy
# print("EIpu")
# EIpu_optimizer = BayesianOptimizer(objective=objective_function, dim=dim, maximize=maximize, initial_points=init_x, kernel=kernel, cost=cost_function)
# EIpu_optimizer.run_until_budget(budget=budget, acquisition_function_class=ExpectedImprovementWithCost, cost_cooling=True)
# EIpu_cost_history = EIpu_optimizer.get_cost_history()
# EIpu_best_history = EIpu_optimizer.get_best_history()
# # EIpu_regret_history = EIpu_optimizer.get_regret_history(global_optimum)

# common_cost_points = np.linspace(0, budget, num=int(10*budget))
# interp_func_best = interp1d(EIpu_cost_history, EIpu_best_history, kind='linear', bounds_error=False, fill_value="extrapolate")
# interp_best = interp_func_best(common_cost_points)

# print("EIpu cost history:", EIpu_cost_history)
# print("EIpu best history:", EIpu_best_history)
# print("EIpu regret history:", EIpu_regret_history)
# print("Interporlated best observed values:", interp_best)
# print()

# # Test Hyperparameter-free Gittins policy
# print("Hyperparameter-free GI")
# GI_optimizer = BayesianOptimizer(objective=objective_function, dim=dim, maximize=maximize, initial_points=init_x, kernel=kernel, cost=cost_function)
# GI_optimizer.run_until_budget(budget=budget, acquisition_function_class=GittinsIndex)
# GI_cost_history = GI_optimizer.get_cost_history()
# GI_best_history = GI_optimizer.get_best_history()
# GI_regret_history = GI_optimizer.get_regret_history(global_optimum)
# GI_lmbda_history = GI_optimizer.get_lmbda_history()

# print("GI cost history:", GI_cost_history)
# print("GI best history:", GI_best_history)
# print("GI regret history:", GI_regret_history)
# print("GI lmbda history:", GI_lmbda_history)
# print()

# # Test Gittins policy with small constant lambda
# print("GI with small constant lambda")
# GI_optimizer = BayesianOptimizer(objective=objective_function, dim=dim, maximize=maximize, initial_points=init_x, kernel=kernel, cost=cost_function)
# GI_optimizer.run_until_budget(budget=budget, acquisition_function_class=GittinsIndex, lmbda=0.0001)
# GI_cost_history = GI_optimizer.get_cost_history()
# GI_best_history = GI_optimizer.get_best_history()
# GI_regret_history = GI_optimizer.get_regret_history(global_optimum)

# print("GI cost history:", GI_cost_history)
# print("GI best history:", GI_best_history)
# print("GI regret history:", GI_regret_history)
# print()
