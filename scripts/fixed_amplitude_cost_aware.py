#!/usr/bin/env python
# coding: utf-8

import torch
from pandora_bayesopt.utils import fit_gp_model
from botorch.models import SingleTaskGP
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.kernels import MaternKernel, ScaleKernel
from botorch.utils.sampling import draw_sobol_samples
from botorch.sampling.pathwise import draw_matheron_paths, draw_kernel_feature_paths
from botorch.utils.sampling import optimize_posterior_samples
from botorch.acquisition import ExpectedImprovement
from pandora_bayesopt.acquisition.ei_puc import ExpectedImprovementWithCost
from pandora_bayesopt.acquisition.multi_step_ei import MultiStepLookaheadEI
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
    lengthscale = config['lengthscale']
    outputscale = config['amplitude']
    maximize = True
    kernel = config['kernel']
    if kernel == 'matern32':
        nu = 1.5
        if lengthscale == 1.0: 
            budget = 10*dim
        elif lengthscale == 0.1:
            budget = 15*dim 
        elif lengthscale == 0.01:
            budget = 20*dim
    elif kernel == 'matern52':
        nu = 2.5
        if lengthscale == 1.0: 
            budget = 5*dim
        elif lengthscale == 0.1:
            budget = 10*dim 
        elif lengthscale == 0.01:
            budget = 15*dim 
    seed = config['seed']
    torch.manual_seed(seed)
    
    # Create the objective function
    base_kernel = MaternKernel(nu=nu).double()
    base_kernel.lengthscale = torch.tensor([[lengthscale]])
    scale_kernel = ScaleKernel(base_kernel).double()
    scale_kernel.outputscale = torch.tensor([[outputscale]])

    # Define Noise Level
    noise_level = 1e-4

    # Initialize Placeholder Data with Correct Dimensions
    num_samples = 1  # Replace with actual number of samples
    num_features = dim  # Replace with actual number of features
    train_X = torch.empty(num_samples, num_features)  # Placeholder data
    train_Y = torch.empty(num_samples, 1)             # Placeholder data
    Yvar = torch.ones(num_samples) * noise_level

    # Initialize Model
    model = SingleTaskGP(train_X, train_Y, covar_module=scale_kernel)

    # Draw a sample path
    matern_sample = draw_kernel_feature_paths(model, sample_shape=torch.Size([1]))
    def objective_function(x):
        return matern_sample(x).squeeze(0).detach()

    # Find the global optimum
    bounds = torch.stack([torch.zeros(dim), torch.ones(dim)])
    global_optimum_point, global_optimum_value = optimize_posterior_samples(paths=matern_sample, bounds=bounds, maximize=maximize)

    # Create the cost function
    if config["costs"] == "linear":
        def cost_function(x):
            return 0.1+x.sum(dim=-1)

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
        cost=cost_function,
        input_standardize=input_standardize
    )
    if policy == 'ExpectedImprovement':
        Optimizer.run_until_budget(
            budget=budget, 
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
            budget = budget, 
            acquisition_function_class=GittinsIndex,
            step_divide = True,
            alpha = 2
        )
    elif policy == 'Gittins_Step_Divide5':
        Optimizer.run_until_budget(
            budget = budget, 
            acquisition_function_class=GittinsIndex,
            step_divide = True,
            alpha = 5
        )
    elif policy == 'Gittins_Step_Divide10':
        Optimizer.run_until_budget(
            budget = budget, 
            acquisition_function_class=GittinsIndex,
            step_divide = True,
            alpha = 10
        )
    elif policy == 'Gittins_Step_EIpu':
        Optimizer.run_until_budget(
            budget=budget, 
            acquisition_function_class=GittinsIndex,
            step_EIpu = True
        )
    cost_history = Optimizer.get_cost_history()
    best_history = Optimizer.get_best_history()
    regret_history = Optimizer.get_regret_history(global_optimum_value.item())

    print("Cost history:", cost_history)
    print("Best history:", best_history)
    print("Regret history:", regret_history)
    print()

    return (budget, global_optimum_value.item(), cost_history, best_history, regret_history)

wandb.init()
(budget, global_optimum_value, cost_history, best_history, regret_history) = run_bayesopt_experiment(wandb.config)

for cost, best, regret in zip(cost_history, best_history, regret_history):
    wandb.log({"raw cumulative cost": cost, "raw best observed": best, "raw regret": regret, "raw log(regret)":np.log10(regret)})

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