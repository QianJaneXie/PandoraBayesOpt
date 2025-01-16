#!/usr/bin/env python3

# BayesOPT Example for showing Gittins >> EIpu
# Extension of the numerical examples presented in Theorem 1 of Raul and Peter's paper which aims to show the limitation of EIpu and EI. The experiment extends the scope from Pandora's box (discrete finite points) to Bayesian optimization (continuous domain) and compares Gittins with EIpu/EI.

import torch
from pandora_bayesopt.utils import create_objective_function
from gpytorch.kernels import MaternKernel
from pandora_bayesopt.kernel import VariableAmplitudeKernel
from botorch.acquisition import ExpectedImprovement, LogExpectedImprovement
from pandora_bayesopt.acquisition.ei_puc import ExpectedImprovementWithCost
from pandora_bayesopt.acquisition.log_ei_puc import LogExpectedImprovementWithCost
from pandora_bayesopt.acquisition.gittins import GittinsIndex
from pandora_bayesopt.bayesianoptimizer import BayesianOptimizer
import numpy as np
import matplotlib.pyplot as plt
import wandb
from scipy.interpolate import interp1d


# use a GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set default tensor type to float64
torch.set_default_dtype(torch.float64)

def run_bayesopt_experiment(config):
    print(config)

    dim = config['dim']
    if config['kernel'] == 'Matern12':
        nu = 0.5
    elif config['kernel'] == 'Matern32':
        nu = 1.5
    elif config['kernel'] == 'Matern52':
        nu = 2.5  
    lengthscale = config['lengthscale']
    outputscale = config['amplitude']
    num_rff_features = config['num_rff_features']
    problem = config['problem']
    
    cost_function_epsilon = config['cost_function_epsilon']
    cost_function_delta = config['cost_function_delta']
    amplitude_function_width = config['amplitude_function_width']
    cost_function_width = config['cost_function_width']
    budget = config['budget']

    seed = config['seed']
    torch.manual_seed(seed)
    output_standardize = config['output_standardize']
    policy = config['policy']
    print("policy:", policy)
    maximize = True

    amplitude_function_sigma = amplitude_function_width / (2*np.sqrt(-2 * np.log(cost_function_epsilon**2)))
    cost_function_sigma = cost_function_width / (2*np.sqrt(-2 * np.log(cost_function_epsilon**2)))

    if problem == 'hard_for_eipc':

        def squared_euclidean_distance(x, center):
            # Calculate the squared Euclidean distance
            return torch.sum((x - center) ** 2, dim=-1)

        def amplitude_function(x):
            center = torch.full_like(x, 0.5)  # Center at [0.5, 0.5, ...]
            dist_squared = squared_euclidean_distance(x, center)

            amplitude = torch.exp(-dist_squared / (2 * amplitude_function_sigma**2)) * (1 - cost_function_epsilon**2) + cost_function_epsilon**2
            return amplitude

        def cost_function(x):
            center = torch.full_like(x, 0.5)  # Center at [0.5, 0.5, ...]
            peak_height = 1 + cost_function_delta - cost_function_epsilon
            dist_squared = squared_euclidean_distance(x, center)

            cost = torch.exp(-dist_squared / (2 * cost_function_sigma**2)) * peak_height + cost_function_epsilon
            return cost

    if problem == 'hard_for_ei':

        def squared_euclidean_distance(x, center):
            # Calculate the squared Euclidean distance
            return torch.sum((x - center) ** 2, dim=-1)

        def amplitude_function(x):
            center = torch.full_like(x, 0.5)  # Center at [0.5, 0.5, ...]
            dist_squared = squared_euclidean_distance(x, center)
            amplitude = torch.exp(-dist_squared / (2 * amplitude_function_sigma**2)) * (1 - (1-cost_function_epsilon)**2) + (1-cost_function_epsilon)**2
            return amplitude

        def cost_function(x):
            center = torch.full_like(x, 0.5)  # Center at [0.5, 0.5, ...]
            peak_height = 1 + cost_function_delta - cost_function_epsilon
            dist_squared = squared_euclidean_distance(x, center)
            cost = torch.exp(-dist_squared / (2 * cost_function_sigma**2)) * peak_height + cost_function_epsilon
            return cost
    
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

    # Find the global optimum using grid search
    grid_points = torch.linspace(0.45, 0.55, 1+int(1/lengthscale))
    grid_values = objective_function(grid_points.view(-1,1))
    global_optimum_value = torch.max(grid_values)
    print("global_optimum", global_optimum_value)

    test_x = torch.linspace(0, 1, 1001)
    test_pts = test_x.cpu().numpy()
    obj_val = objective_function(test_x.view(-1,1)).numpy()
    cost_val = cost_function(test_x.view(-1,1)).numpy()

    # Set up the kernel
    base_kernel = MaternKernel(nu=nu).double()
    base_kernel.lengthscale = lengthscale
    base_kernel.raw_lengthscale.requires_grad = False
    kernel = VariableAmplitudeKernel(base_kernel, amplitude_function)

    # Test performance of different policies
    init_x = torch.zeros(dim).unsqueeze(0)
    Optimizer = BayesianOptimizer(
        objective=objective_function, 
        dim=dim, 
        maximize=maximize, 
        initial_points=init_x, 
        kernel=kernel, 
        cost=cost_function,
        output_standardize=output_standardize
    )
    if policy == 'ExpectedImprovementWithoutCost':
        Optimizer.run_until_budget(
            budget = budget, 
            acquisition_function_class=ExpectedImprovement
        )
    elif policy == 'ExpectedImprovementPerUnitCost':
        Optimizer.run_until_budget(
            budget = budget, 
            acquisition_function_class = ExpectedImprovementWithCost
        )
    elif policy == 'LogExpectedImprovementWithoutCost':
        Optimizer.run_until_budget(
            budget = budget, 
            acquisition_function_class=LogExpectedImprovement
        )
    elif policy == 'LogExpectedImprovementPerUnitCost':
        Optimizer.run_until_budget(
            budget = budget, 
            acquisition_function_class = LogExpectedImprovementWithCost
        )
    elif policy == 'Gittins_Lambda_01':
        Optimizer.run_until_budget(
            budget = budget, 
            acquisition_function_class = GittinsIndex,
            lmbda = 0.01
        )
    elif policy == 'Gittins_Lambda_001':
        Optimizer.run_until_budget(
            budget = budget, 
            acquisition_function_class = GittinsIndex,
            lmbda = 0.001
        )
    elif policy == 'Gittins_Lambda_0001':
        Optimizer.run_until_budget(
            budget = budget, 
            acquisition_function_class = GittinsIndex,
            lmbda = 0.0001
        )
    elif policy == 'Gittins_Step_Divide2':
        Optimizer.run_until_budget(
            budget=budget, 
            acquisition_function_class=GittinsIndex,
            step_divide = True,
            alpha = 2
        )

    cost_history = Optimizer.get_cost_history()
    best_history = Optimizer.get_best_history()
    regret_history = Optimizer.get_regret_history(global_optimum_value)

    print("Cost history:", cost_history)
    print("Best history:", best_history)
    print("Regret history:", regret_history)
    print()

    return (budget, test_pts, obj_val, cost_val, global_optimum_value, cost_history, best_history, regret_history)

wandb.init()
(budget, test_pts, obj_val, cost_val, global_optimum_value, cost_history, best_history, regret_history) = run_bayesopt_experiment(wandb.config)

for x, y, c in zip(test_pts, obj_val, cost_val):
    wandb.log({"x": x, "f(x)": y, "c(x)":c})

wandb.log({"global optimum value": global_optimum_value})

for cost, best, regret in zip(cost_history, best_history, regret_history):
    wandb.log({"raw cumulative cost": cost, "raw best observed": best, "raw regret": regret, "raw lg(regret)":np.log10(regret)})

interp_cost = np.linspace(0, budget, num=int(10*budget)+1)
interp_func_best = interp1d(cost_history, best_history, kind='linear', bounds_error=False, fill_value="extrapolate")
interp_best = interp_func_best(interp_cost)
interp_func_regret = interp1d(cost_history, regret_history, kind='linear', bounds_error=False, fill_value="extrapolate")
interp_regret = interp_func_regret(interp_cost)
interp_func_log_regret = interp1d(cost_history, list(np.log10(regret_history)), kind='linear', bounds_error=False, fill_value="extrapolate")
interp_log_regret = interp_func_log_regret(interp_cost)

for cost, best, regret, log_regret in zip(interp_cost, interp_best, interp_regret, interp_log_regret):
    wandb.log({"cumulative cost": cost, "best observed": best, "regret": regret, "lg(regret)": log_regret})

wandb.finish()