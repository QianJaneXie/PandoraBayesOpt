#!/usr/bin/env python3

import os
import sys
import scipy.io
import torch
from botorch.test_functions.synthetic import Ackley, DropWave, Shekel, Rosenbrock, Levy
from botorch.utils.sampling import draw_sobol_samples
from botorch.acquisition import ExpectedImprovement
from pandora_bayesopt.acquisition.gittins import GittinsIndex
from pandora_bayesopt.acquisition.ei_puc import ExpectedImprovementWithCost
from pandora_bayesopt.acquisition.budgeted_multi_step_ei import BudgetedMultiStepLookaheadEI
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
    dim = config['dim']
    seed = config['seed']
    torch.manual_seed(seed)
    output_standardize = config['output_standardize']
    draw_initial_method = config['draw_initial_method']
    policy = config['policy']
    budget = config['budget_to_dimension_ratio']*dim
    maximize = True
    c_min = config['cost_min']
    cost_scale_factor = config['cost_function_scale_factor']

    # Create the cost function
    if config["cost_function_type"] == "mean":
        def cost_function(x):
            return cost_scale_factor*(c_min+x.mean(dim=-1))
    if config["cost_function_type"] == "periodic":
        a = torch.tensor([1.1969])
        b = torch.tensor([1.4694])
        c = torch.tensor([3.0965])
        pi = 3.14

    if problem == 'Ackley':
        ackley_function = Ackley(dim=dim)
        objective_scale_factor = -1
        def objective_function(X):
            return ackley_function(2*X-1)/objective_scale_factor
        global_optimum_value = 0

        if config["cost_function_type"] == "periodic":
            def cost_function(X):
                X_unnorm = (2.0 * X) - 1.0
                ln_cost_X = a * torch.cos(b * (2 * pi) * (X_unnorm + c)).mean(dim=-1)
                cost_X = torch.exp(ln_cost_X)
                return cost_X

    if problem == 'Rosenbrock':
        rosenbrock_function = Rosenbrock(dim=dim)
        objective_scale_factor = -100000
        def objective_function(X):
            return rosenbrock_function(15*X-5)/objective_scale_factor
        global_optimum_value = 0

        if config["cost_function_type"] == "periodic":
            def cost_function(X):
                X_unnorm = 15 * X - 5
                ln_cost_X = a * torch.cos(b * (2 * pi / 7.5) * (X_unnorm + c)).mean(dim=-1)
                cost_X = torch.exp(ln_cost_X)
                return cost_X

    if problem == 'Levy':
        levy_function = Levy(dim=dim)
        objective_scale_factor = -100
        def objective_function(X):
            return levy_function(20*X-10)/objective_scale_factor
        global_optimum_value = 0
        
        if config["cost_function_type"] == "periodic":
            def cost_function(X):
                X_unnorm = 20 * X - 10
                ln_cost_X = a * torch.cos(b * (2 * pi / 10.0) * (X_unnorm + c)).mean(dim=-1)
                cost_X = torch.exp(ln_cost_X)
                return cost_X

    if draw_initial_method == 'sobol':
        bounds = torch.stack([torch.zeros(dim), torch.ones(dim)])
        init_x = draw_sobol_samples(bounds=bounds, n=1, q=2*(dim+1)).squeeze(0)

    # Test performance of different policies
    print("policy:", policy)
    
    Optimizer = BayesianOptimizer( 
        dim=dim, 
        maximize=maximize, 
        initial_points=init_x,
        objective=objective_function, 
        cost=cost_function,
        output_standardize=output_standardize
    )
    if policy == 'RandomSearch':
        Optimizer.run_until_budget(
            budget=budget, 
            acquisition_function_class="RandomSearch"
        )
    if policy == 'ExpectedImprovementWithoutCost':
        Optimizer.run_until_budget(
            budget=budget, 
            acquisition_function_class=ExpectedImprovement
        )
    elif policy == 'ExpectedImprovementPerUnitCost':
        Optimizer.run_until_budget(
            budget = budget, 
            acquisition_function_class = ExpectedImprovementWithCost
        )
    elif policy == 'ExpectedImprovementWithCostCooling':
        Optimizer.run_until_budget(
            budget = budget, 
            acquisition_function_class = ExpectedImprovementWithCost,
            cost_cooling = True
        )
    elif policy == 'BudgetedMultiStepLookaheadEI':
        Optimizer.run_until_budget(
            budget=budget, 
            acquisition_function_class=BudgetedMultiStepLookaheadEI
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
    regret_history = Optimizer.get_regret_history(global_optimum_value/objective_scale_factor)

    print("Cost history:", cost_history)
    print("Best history:", best_history)
    print("Regret history:", regret_history)
    print()

    return (budget, objective_scale_factor, cost_history, best_history, regret_history)

wandb.init()
(budget, objective_scale_factor, cost_history, best_history, regret_history) = run_bayesopt_experiment(wandb.config)

for cost, best, regret in zip(cost_history, best_history, regret_history):
    wandb.log({"raw cumulative cost": cost, "raw best observed": objective_scale_factor*best, "raw regret": -objective_scale_factor*regret, "raw log(regret)":np.log10(-objective_scale_factor)+np.log10(regret)})

interp_cost = np.linspace(0, budget, num=budget+1)
interp_func_best = interp1d(cost_history, best_history, kind='linear', bounds_error=False, fill_value="extrapolate")
interp_best = interp_func_best(interp_cost)
interp_func_regret = interp1d(cost_history, regret_history, kind='linear', bounds_error=False, fill_value="extrapolate")
interp_regret = interp_func_regret(interp_cost)
interp_func_log_regret = interp1d(cost_history, list(np.log10(regret_history)), kind='linear', bounds_error=False, fill_value="extrapolate")
interp_log_regret = interp_func_log_regret(interp_cost)

for cost, best, regret, log_regret in zip(interp_cost, interp_best, interp_regret, interp_log_regret):
    wandb.log({"cumulative cost": cost, "best observed": objective_scale_factor*best, "regret": -objective_scale_factor*regret, "lg(regret)": np.log10(-objective_scale_factor)+log_regret})

wandb.finish()