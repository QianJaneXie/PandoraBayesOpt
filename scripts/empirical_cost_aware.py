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
    if problem == "LDA":
        dim = 3
        maximize = True
        
        script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
        data = scipy.io.loadmat(script_dir + "/data/lda/lda_on_grid.mat")
        data = data["lda_on_grid"]
        X = torch.tensor(data[:, :3])
        X[:, 0] = 2.0 * X[:, 0] - 1.0
        X[:, 1] = torch.log2(X[:, 1]) / 10.0
        X[:, 2] = torch.log2(X[:, 2]) / 14.0
        objective_X = -torch.tensor(data[:, 3]).unsqueeze(-1)
        cost_X = torch.tensor(data[:, 4]).unsqueeze(-1) / 3600.0
        
        torch.manual_seed(0)
        objective_model = fit_gp_model(X, objective_X, input_standardize=True)
        
        
        # Define the objective function
        scaled_constant = 1000
        def objective_function(X):
            if X.ndim == 1:
                X = X.unsqueeze(0)
            posterior_X = objective_model.posterior(X)
            objective_X = posterior_X.mean.detach()
            return objective_X/scaled_constant
        
        global_optimum_value = -1.2606842790227435

        log_cost_X = torch.log(cost_X)
        log_cost_model = fit_gp_model(X, log_cost_X, input_standardize=True)
        
        
        # Define the cost function
        def cost_function(X):
            posterior_X = log_cost_model.posterior(X)
            cost_X = torch.exp(posterior_X.mean)
            return cost_X

        budget = 80

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
    elif policy == 'BudgetedMultiStepLookaheadEI':
        Optimizer.run_until_budget(
            budget=budget, 
            acquisition_function_class=BudgetedMultiStepLookaheadEI
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