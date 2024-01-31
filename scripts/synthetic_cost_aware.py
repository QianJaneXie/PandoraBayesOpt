#!/usr/bin/env python
# coding: utf-8

import os
import sys
import scipy.io
import torch
from pandora_bayesopt.utils import fit_gp_model
from botorch.utils.sampling import draw_sobol_samples
from botorch.acquisition import ExpectedImprovement
from pandora_bayesopt.acquisition.gittins import GittinsIndex
from pandora_bayesopt.acquisition.ei_puc import ExpectedImprovementWithCost
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

    problem = config['problem']
    seed = config['seed']
    torch.manual_seed(seed)
    input_standardize = config['input_normalization']
    draw_initial_method = config['draw_initial_method']
    policy = config['policy']
    print("policy:", policy)
    budget = config['total_cost_budget']
    maximize = True

    a = torch.tensor([1.1969])
    b = torch.tensor([1.4694])
    c = torch.tensor([3.0965])
    pi = 3.14

    if problem == 'Ackley_5D':
        dim = 5
        ackley_function = Ackley(dim=dim)
        scaled_constant = -1
        def objective_function(X):
            return ackley_function(2*X-1)/scaled_constant
        global_optimum_value = 0

        def cost_function(X):
            X_unnorm = (2.0 * X) - 1.0
            ln_cost_X = a * torch.cos(b * (2 * pi) * (X_unnorm + c)).mean(dim=-1)
            cost_X = torch.exp(ln_cost_X)
            return cost_X

    if problem == 'DropWave_2D':
        dim = 2
        dropwave_function = DropWave()
        scaled_constant = -1
        def objective_function(X):
            return dropwave_function(10.24*X-5.12)/scaled_constant
        global_optimum_value = -1.0

        def cost_function(X):
            X_unnorm = (X * 10.24) - 5.12 
            ln_cost_X = a * torch.cos(b * (2 * pi / 5.12) * (X_unnorm + c)).mean(dim=-1)
            cost_X = torch.exp(ln_cost_X)
            return cost_X

    if problem == 'Shekel5_4D':
        dim = 4
        shekel_function = Shekel(m=5)
        scaled_constant = -2
        def objective_function(X):
            return shekel_function(10*X)/scaled_constant
        global_optimum_value = -10.1532

        def cost_function(X):
            X_unnorm = 10 * X 
            ln_cost_X = a * torch.cos(b * (2 * pi / 5.0) * (X_unnorm + c)).mean(dim=-1)
            cost_X = torch.exp(ln_cost_X)
            return cost_X

    if problem == 'Rosenbrock_5D':
        dim = 5
        rosenbrock_function = Rosenbrock(dim=dim)
        scaled_constant = -1000
        def objective_function(X):
            return rosenbrock_function(15*X-5)/scaled_constant
        global_optimum_value = 0

        def cost_function(X):
            X_unnorm = 15 * X - 5
            ln_cost_X = a * torch.cos(b * (2 * pi / 7.5) * (X_unnorm + c)).mean(dim=-1)
            cost_X = torch.exp(ln_cost_X)
            return cost_X

    if problem == 'Levy_5D':
        dim = 5
        levy_function = Levy(dim=dim)
        scaled_constant = -100
        def objective_function(X):
            return levy_function(20*X-10)/scaled_constant
        global_optimum_value = 0
        
        def cost_function(X):
            X_unnorm = 20 * X - 10
            ln_cost_X = a * torch.cos(b * (2 * pi / 10.0) * (X_unnorm + c)).mean(dim=-1)
            cost_X = torch.exp(ln_cost_X)
            return cost_X


    seed = config['seed']
    torch.manual_seed(seed)
    draw_initial_method = config['draw_initial_method']
    if draw_initial_method == 'sobol':
        bounds = torch.stack([torch.zeros(dim), torch.ones(dim)])
        init_x = draw_sobol_samples(bounds=bounds, n=1, q=2*(dim+1)).squeeze(0)
    input_standardize = config['input_normalization']

    # Test performance of different policies
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
    regret_history = Optimizer.get_regret_history(global_optimum_value)

    print("Cost history:", cost_history)
    print("Best history:", best_history)
    print("Regret history:", regret_history)
    print()

    return (budget, scaled_constant, cost_history, best_history, regret_history)

wandb.init()
(budget, scaled_constant, cost_history, best_history, regret_history) = run_bayesopt_experiment(wandb.config)

for cost, best, regret in zip(cost_history, best_history, regret_history):
    wandb.log({"raw cumulative cost": cost, "raw best observed": -scaled_constant*best, "raw regret": scaled_constant*regret, "raw log(regret)":np.log10(scaled_constant)+np.log10(regret)})

interp_cost = np.linspace(0, budget, num=int(10*budget)+1)
interp_func_best = interp1d(cost_history, best_history, kind='linear', bounds_error=False, fill_value="extrapolate")
interp_best = interp_func_best(interp_cost)
interp_func_regret = interp1d(cost_history, regret_history, kind='linear', bounds_error=False, fill_value="extrapolate")
interp_regret = interp_func_regret(interp_cost)
interp_func_log_regret = interp1d(cost_history, list(np.log10(regret_history)), kind='linear', bounds_error=False, fill_value="extrapolate")
interp_log_regret = interp_func_log_regret(interp_cost)

for cost, best, regret, log_regret in zip(interp_cost, interp_best, interp_regret, interp_log_regret):
    wandb.log({"cumulative cost": cost, "best observed": -scaled_constant*best, "regret": scaled_constant*regret, "log(regret)": np.log10(scaled_constant)+log_regret})

wandb.finish()