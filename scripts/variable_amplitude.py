#!/usr/bin/env python
# coding: utf-8

# # BayesOPT Example for showing Gittins >> EIpu
# Extension of the numerical examples presented in Theorem 1 of Raul and Peter's paper which aims to show the limitation of EIpu and EI. The experiment extends the scope from Pandora's box (discrete finite points) to Bayesian optimization (continuous domain) and compares Gittins with EIpu/EI.

import torch
from pandora_bayesopt.utils import fit_gp_model, create_objective_function, find_global_optimum
from gpytorch.kernels import MaternKernel
from pandora_bayesopt.kernel import VariableAmplitudeKernel
from botorch.acquisition import ExpectedImprovement
from pandora_bayesopt.acquisition import ExpectedImprovementWithCost, GittinsIndex
from pandora_bayesopt.bayesianoptimizer import BayesianOptimizer

import matplotlib.pyplot as plt
import wandb


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
    seed = config['seed']
    torch.manual_seed(seed)
    policy = config['policy']
    print("policy:", policy)
    maximize = True

    # Create the objective function
    def objective_function(x):
        matern_sample = create_objective_function(
        seed=seed, 
        dim=dim, 
        nu=nu, 
        lengthscale=lengthscale,
        outputscale=outputscale,
        num_rff_features=num_rff_features
        )
        return matern_sample(x) * amplitude_function(x)

    # Find the global optimum
    global_optimum_point, global_optimum_value = find_global_optimum(objective=objective_function, dim=dim, maximize=maximize)
    print("global_optimum", global_optimum_point, global_optimum_value)

    test_x = torch.linspace(0, 1, 3001, dtype=torch.float64, device=device)
    # Plot for scaled objective function
    plt.plot(test_x.cpu().numpy(), objective_function(test_x.view(-1,1)).numpy(), color='tab:grey', label="Scaled objective function", alpha=0.6)
    plt.plot(test_x.cpu().numpy(), cost_function(test_x.view(-1,1)).numpy(), label="Cost function", alpha=0.6)
    plt.plot(global_optimum_point.cpu().numpy(), global_optimum_value, 'r*', label="global_optimum", alpha=0.8)
    plt.title(f"Scaled objective function and cost function")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)

    base_kernel = MaternKernel(nu=nu).double()
    base_kernel.lengthscale = torch.tensor([[lengthscale]], dtype=torch.float64)
    kernel = VariableAmplitudeKernel(base_kernel, amplitude_function)

    # Test performance of different policies
    budget = config['budget']
    init_x = torch.zeros(dim).unsqueeze(1)
    Optimizer = BayesianOptimizer(
        objective=objective_function, 
        dim=dim, 
        maximize=maximize, 
        initial_points=init_x, 
        kernel=kernel, 
        cost=cost_function
    )
    if policy == 'EI':
        Optimizer.run_until_budget(
            budget=budget, 
            acquisition_function_class=ExpectedImprovement
        )
    elif policy == 'EIpu':
        Optimizer.run_until_budget(
            budget=budget, 
            acquisition_function_class=ExpectedImprovementWithCost
        )
    elif policy == 'EIcool':
        Optimizer.run_until_budget(
            budget=budget, 
            acquisition_function_class=ExpectedImprovementWithCost,
            cost_cooling = True
        )
    elif policy == 'GIlmbda':
        Optimizer.run_until_budget(
            budget=budget, 
            acquisition_function_class=GittinsIndex,
            lmbda = 0.0001
        )
    elif policy == 'GIfree':
        Optimizer.run_until_budget(
            budget=budget, 
            acquisition_function_class=GittinsIndex
        )
    cost_history = Optimizer.get_cost_history()
    best_history = Optimizer.get_best_history()
    regret_history = Optimizer.get_regret_history(global_optimum_value)

    print("Cost history:", cost_history)
    print("Best history:", best_history)
    print("Regret history:", regret_history)
    print()

    return (budget, plt, global_optimum_value, cost_history, best_history, regret_history)

wandb.init()
(budget, plt, global_optimum_value, cost_history, best_history, regret_history) = run_bayesopt_experiment(wandb.config)

wandb.log({"plot": wandb.Image(plt)})
plt.close()

wandb.log({"global optimum value": global_optimum_value})

for cost, best in zip(cost_history, best_history):
    wandb.log({"raw cumulative cost": cost, "raw best observed": best})
for cost, regret in zip(cost_history, regret_history):
    wandb.log({"raw cumulative cost": cost, "raw regret": regret})

interp_cost = np.linspace(0, budget, num=int(10*budget))
interp_func_best = interp1d(cost_history, best_history, kind='linear', bounds_error=False, fill_value="extrapolate")
interp_best = interp_func_best(interp_cost)
interp_func_regret = interp1d(cost_history, regret_history, kind='linear', bounds_error=False, fill_value="extrapolate")
interp_regret = interp_func_regret(interp_cost)

for cost, best in zip(interp_cost, interp_best):
    wandb.log({"cumulative cost": cost, "best observed": best})
for cost, regret in zip(interp_cost, interp_regret):
    wandb.log({"cumulative cost": cost, "regret": regret})

wandb.finish()