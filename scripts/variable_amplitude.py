#!/usr/bin/env python
# coding: utf-8

# # BayesOPT Example for showing Gittins >> EIpu
# Extension of the numerical examples presented in Theorem 1 of Raul and Peter's paper which aims to show the limitation of EIpu and EI. The experiment extends the scope from Pandora's box (discrete finite points) to Bayesian optimization (continuous domain) and compares Gittins with EIpu/EI.

import torch
from pandora_bayesopt.utils import fit_gp_model, create_objective_function, find_global_optimum
from gpytorch.kernels import MaternKernel
from pandora_bayesopt.kernel import VariableAmplitudeKernel
from botorch.acquisition import ExpectedImprovement
from pandora_bayesopt.acquisition.ei_puc import ExpectedImprovementWithCost
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
    if config['kernel'] == 'matern12':
        nu = 0.5
    elif config['kernel'] == 'matern32':
        nu = 1.5
    elif config['kernel'] == 'matern52':
        nu = 2.5  
    lengthscale = config['lengthscale']
    outputscale = config['amplitude']
    num_rff_features = config['num_rff_features']
    problem = config['problem']
    if problem == 'hard_for_eipc':
        cost_function_epsilon = 0.1
        cost_function_delta = 1.0
        amplitude_function_width = 0.001
        cost_function_width = 0.001
        budget = 4.0
    seed = config['seed']
    torch.manual_seed(seed)
    input_standardize = config['input_normalization']
    policy = config['policy']
    print("policy:", policy)
    maximize = True

    def squared_euclidean_distance(x, center):
        # Calculate the squared Euclidean distance
        return torch.sum((x - center) ** 2, dim=-1)

    def amplitude_function(x):
        center = torch.full_like(x, 0.5)  # Center at [0.5, 0.5, ...]
        dist_squared = squared_euclidean_distance(x, center)
        amplitude = torch.exp(-dist_squared / (2 * amplitude_function_width**2)) * (1 - cost_function_epsilon**2) + cost_function_epsilon**2
        return amplitude

    def cost_function(x):
        center = torch.full_like(x, 0.5)  # Center at [0.5, 0.5, ...]
        width = cost_function_width
        peak_height = 1 + cost_function_delta - cost_function_epsilon
        dist_squared = squared_euclidean_distance(x, center)
        cost = torch.exp(-dist_squared / (2 * width**2)) * peak_height + cost_function_epsilon
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

    # Find the global optimum
    global_optimum_point, global_optimum_value = find_global_optimum(objective=objective_function, dim=dim, maximize=maximize)
    print("global_optimum", global_optimum_point, global_optimum_value)

    test_x = torch.linspace(0, 1, 1001, dtype=torch.float64, device=device)
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
        input_standardize=input_standardize
    )
    if policy == 'ExpectedImprovement':
        Optimizer.run_until_budget(
            budget = budget, 
            acquisition_function_class=ExpectedImprovement
        )
    elif policy == 'ExpectedImprovementWithCost_Uniform':
        Optimizer.run_until_budget(
            budget = budget, 
            acquisition_function_class = ExpectedImprovementWithCost
        )
    elif policy == 'ExpectedImprovementWithCost_Cooling':
        Optimizer.run_until_budget(
            budget = budget, 
            acquisition_function_class = ExpectedImprovementWithCost,
            cost_cooling = True
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
    elif policy == 'Gittins_Step_EIpu':
        Optimizer.run_until_budget(
            budget=budget, 
            acquisition_function_class=GittinsIndex,
            step_EIpu = True
        )
    cost_history = Optimizer.get_cost_history()
    best_history = Optimizer.get_best_history()
    regret_history = Optimizer.get_regret_history(global_optimum_value)

    print("Cost history:", cost_history)
    print("Best history:", best_history)
    print("Regret history:", regret_history)
    print()

    return (budget, test_pts, obj_val, cost_val, global_optimum_point, global_optimum_value, cost_history, best_history, regret_history)

wandb.init()
(budget, test_pts, obj_val, cost_val, global_optimum_point, global_optimum_value, cost_history, best_history, regret_history) = run_bayesopt_experiment(wandb.config)

for x, y, c in zip(test_pts, obj_val, cost_val):
    wandb.log({"x": x, "f(x)": y, "c(x)":c})

wandb.log({"global optimum point": global_optimum_point, "global optimum value": global_optimum_value})

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