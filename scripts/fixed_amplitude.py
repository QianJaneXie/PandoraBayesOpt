#!/usr/bin/env python
# coding: utf-8

import torch
from pandora_bayesopt.utils import fit_gp_model, create_objective_function, find_global_optimum_scipy
from gpytorch.kernels import MaternKernel, ScaleKernel
from botorch.utils.sampling import draw_sobol_samples
from botorch.acquisition import ExpectedImprovement
from pandora_bayesopt.acquisition.gittins import GittinsIndex
from pandora_bayesopt.bayesianoptimizer import BayesianOptimizer

import numpy as np
import matplotlib.pyplot as plt
import wandb

# use a GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set default tensor type to float64
torch.set_default_dtype(torch.float64)

def run_bayesopt_experiment(config):
    print(config)

    dim = config['dim']
    lengthscale = config['lengthscale']
    outputscale = config['amplitude']
    maximize = True
    if config['kernel'] == 'matern32':
        nu = 1.5
        if lengthsale == 1.0: 
            num_iterations = 10*dim
        elif lengthsale == 0.1:
            num_iterations = 20*dim 
        elif lengthsale == 0.01:
            num_iterations = 30*dim
    elif config['kernel'] == 'matern52':
        nu = 2.5
        if lengthsale == 1.0: 
            num_iterations = 5*dim
        elif lengthsale == 0.1:
            num_iterations = 15*dim 
        elif lengthsale == 0.01:
            num_iterations = 25*dim 
    num_rff_features = config['num_rff_features']
    seed = config['seed']
    torch.manual_seed(seed)
    
    # Create the objective function
    objective_function = create_objective_function(
        dim=dim, 
        nu=nu, 
        lengthscale=lengthscale,
        outputscale=outputscale,
        num_rff_features=num_rff_features
    )

    # Find the global optimum
    global_optimum_point, global_optimum_value = find_global_optimum_scipy(objective=objective_function, dim=dim, maximize=maximize)
    print("global_optimum", global_optimum_point, global_optimum_value)

    # Set up the kernel
    base_kernel = MaternKernel(nu=nu).double()
    base_kernel.lengthscale = lengthscale
    base_kernel.raw_lengthscale.requires_grad = False
    scale_kernel = ScaleKernel(base_kernel).double()
    scale_kernel.outputscale = torch.tensor([[outputscale]])
    scale_kernel.raw_outputscale.requires_grad = False

    # Test performance of different policies
    draw_initial_method = config['draw_initial_method']
    if draw_initial_method == 'sobol':
        bounds = torch.stack([torch.zeros(dim), torch.ones(dim)])
        init_x = draw_sobol_samples(bounds=bounds, n=1, q=2*(dim+1)).squeeze(0)
    input_standardize = config['input_normalization']

    policy = config['policy']
    print("policy:", policy)

    Optimizer = BayesianOptimizer(
        objective=objective_function, 
        dim=dim, 
        maximize=maximize, 
        initial_points=init_x, 
        kernel=kernel,
        input_standardize=input_standardize
    )
    if policy == 'ExpectedImprovement':
        Optimizer.run(
            num_iterations=num_iterations, 
            acquisition_function_class=ExpectedImprovement
        )
    elif policy == 'ExpectedImprovementWithCost_Uniform':
        Optimizer.run(
            num_iterations=num_iterations, 
            acquisition_function_class = ExpectedImprovementWithCost
        )
    elif policy == 'ExpectedImprovementWithCost_Cooling':
        Optimizer.run(
            num_iterations=num_iterations, 
            acquisition_function_class = ExpectedImprovementWithCost,
            cost_cooling = True
        )
    elif policy == 'Gittins_Lambda_01':
        Optimizer.run(
            num_iterations=num_iterations, 
            acquisition_function_class = GittinsIndex,
            lmbda = 0.01
        )
    elif policy == 'Gittins_Lambda_001':
        Optimizer.run(
            num_iterations = num_iterations, 
            acquisition_function_class = GittinsIndex,
            lmbda = 0.001
        )
    elif policy == 'Gittins_Lambda_0001':
        Optimizer.run(
            num_iterations = num_iterations, 
            acquisition_function_class = GittinsIndex,
            lmbda = 0.0001
        )
    elif policy == 'Gittins_Step_Divide2':
        Optimizer.run(
            num_iterations=num_iterations, 
            acquisition_function_class=GittinsIndex,
            step_divide = True,
            alpha = 2
        )
    elif policy == 'Gittins_Step_Divide5':
        Optimizer.run(
            num_iterations=num_iterations, 
            acquisition_function_class=GittinsIndex,
            step_divide = True,
            alpha = 5
        )
    elif policy == 'Gittins_Step_Divide10':
        Optimizer.run(
            num_iterations=num_iterations, 
            acquisition_function_class=GittinsIndex,
            step_divide = True,
            alpha = 10
        )
    elif policy == 'Gittins_Step_EIpu':
        Optimizer.run(
            num_iterations=num_iterations, 
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

    return (cost_history, best_history, regret_history)

wandb.init()
(cost_history, best_history, regret_history) = run_bayesopt_experiment(wandb.config)

for cost, best, regret in zip(cost_history, best_history, regret_history):
    wandb.log({"cumulative cost": cost, "best observed": best, "regret": regret, "lg(regret)": np.log10(regret)})

wandb.finish()