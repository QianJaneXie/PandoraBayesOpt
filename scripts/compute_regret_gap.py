#!/usr/bin/env python
# coding: utf-8

import torch
from botorch.models import SingleTaskGP
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from botorch.utils.sampling import draw_sobol_samples
from botorch.sampling.pathwise import draw_kernel_feature_paths
from botorch.utils.sampling import optimize_posterior_samples
from botorch.acquisition import UpperConfidenceBound
from pandora_bayesopt.acquisition.gittins import GittinsIndex
from pandora_bayesopt.bayesianoptimizer import BayesianOptimizer

import wandb
import numpy as np

# use a GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set default tensor type to float64
torch.set_default_dtype(torch.float64)

def run_bayesopt_experiment(config, policy, seed):
    print(config)

    torch.manual_seed(seed)
    dim = config['dim']
    lengthscale = config['lengthscale']
    outputscale = config['amplitude']
    maximize = True
    kernel = config['kernel']
    if kernel == 'Matern32':
        nu = 1.5
    if kernel == 'Matern52':
        nu = 2.5
    
    # Create the objective function
    if kernel == 'RBF':
        base_kernel = RBFKernel().double()
    else:
        base_kernel = MaternKernel(nu=nu).double()
    base_kernel.lengthscale = torch.tensor([[lengthscale]])
    scale_kernel = ScaleKernel(base_kernel).double()
    scale_kernel.outputscale = torch.tensor([[outputscale]])

    # Define Noise Level
    noise_level = 1e-4

    # Initialize Placeholder Data with Correct Dimensions
    num_samples = 1  # Replace with actual number of samples
    num_features = dim  # Replace with actual number of features
    train_X = torch.zeros(num_samples, num_features)  # Placeholder data
    train_Y = torch.zeros(num_samples, 1)             # Placeholder data
    Yvar = torch.ones(num_samples) * noise_level

    # Initialize Model
    model = SingleTaskGP(train_X, train_Y, likelihood = FixedNoiseGaussianLikelihood(noise=Yvar), covar_module=scale_kernel)

    # Draw a sample path
    sample_path = draw_kernel_feature_paths(model, sample_shape=torch.Size([1]))
    def objective_function(x):
        return sample_path(x).squeeze(0).detach()

    # Find the global optimum
    bounds = torch.stack([torch.zeros(dim), torch.ones(dim)])
    global_optimum_point, global_optimum_value = optimize_posterior_samples(paths=sample_path, bounds=bounds, raw_samples=1024*dim, num_restarts=20*dim, maximize=maximize)
    print("global optimum point:", global_optimum_point.detach().numpy())
    print("global optimum value:", global_optimum_value.item())

    # Set up the kernel
    if kernel == 'RBF':
        base_kernel = RBFKernel().double()
    else:
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
    num_iterations = config['num_iterations']
    lmbda = config['Gittins_lmbda']

    Optimizer = BayesianOptimizer(
        dim=dim, 
        maximize=maximize, 
        initial_points=init_x,
        objective=objective_function, 
        kernel=scale_kernel,
        input_standardize=input_standardize
    )
    if policy == 'UpperConfidenceBound':
        Optimizer.run(
            num_iterations=num_iterations, 
            acquisition_function_class=UpperConfidenceBound,
            heuristic=True
        )
    elif policy == 'GittinsIndex':
        Optimizer.run(
            num_iterations=num_iterations, 
            acquisition_function_class = GittinsIndex,
            lmbda = lmbda,
            bisection_early_stopping = True
        )
    
    best_history = Optimizer.get_best_history()

    print("Best observed value:", best_history[-1])
    print("Regret:", global_optimum_value.item()-best_history[-1])
    print()

    return (global_optimum_value.item(), best_history[-1])

if __name__ == "__main__":
    # Initialize a new wandb run
    wandb.init()
    UCB_best = []
    UCB_regret = []
    Gittins_best = []
    Gittins_regret = [] 
    for seed in range(16):
        (global_optimum_value, UCB_best_observed_value) = run_bayesopt_experiment(wandb.config, "UpperConfidenceBound", seed)
        (global_optimum_value, Gittins_best_observed_value) = run_bayesopt_experiment(wandb.config, "GittinsIndex", seed)
        UCB_best.append(UCB_best_observed_value)
        Gittins_best.append(Gittins_best_observed_value)
        UCB_regret.append(UCB_best_observed_value)
        Gittins_regret.append(Gittins_best_observed_value)
    
    print("UCB best observed value:", UCB_best)
    print("Gittins best observed value:", Gittins_best)
    print("UCB regrets:", UCB_regret)
    print("Gittins regrets:", Gittins_regret)
    median_ucb = np.median(UCB_regret)
    median_gittins = np.median(Gittins_regret)
    print("UCB regret median:", median_ucb)
    print("Gittins regret median:", median_gittins)
    median_regret_gap = median_ucb - median_gittins

    wandb.log({"median_regret_gap": median_regret_gap, "median_ucb": median_ucb, "median_gittins": median_gittins})

    # Finish the wandb run
    wandb.finish()