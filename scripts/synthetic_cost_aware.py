#!/usr/bin/env python3

import torch
from botorch.test_functions.synthetic import Ackley, Rosenbrock, Levy
from botorch.utils.sampling import draw_sobol_samples
from botorch.acquisition import ExpectedImprovement
from pandora_bayesopt.acquisition.gittins import GittinsIndex
from pandora_bayesopt.acquisition.ei_puc import ExpectedImprovementWithCost
from pandora_bayesopt.acquisition.budgeted_multi_step_ei import BudgetedMultiStepLookaheadEI
from botorch.acquisition.max_value_entropy_search import qMultiFidelityMaxValueEntropy
from pandora_bayesopt.bayesianoptimizer import BayesianOptimizer
import numpy as np
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

    if problem == 'Ackley':
        ackley_function = Ackley(dim=dim)
        objective_scale_factor = -1
        def objective_function(X):
            return ackley_function(2*X-1)/objective_scale_factor
        global_optimum_value = 0

    if problem == 'Rosenbrock':
        rosenbrock_function = Rosenbrock(dim=dim)
        objective_scale_factor = -100000
        def objective_function(X):
            return rosenbrock_function(15*X-5)/objective_scale_factor
        global_optimum_value = 0

    if problem == 'Levy':
        levy_function = Levy(dim=dim)
        objective_scale_factor = -100
        def objective_function(X):
            return levy_function(20*X-10)/objective_scale_factor
        global_optimum_value = 0

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
            budget = budget, 
            acquisition_function_class = "RandomSearch"
        )
    if policy == 'ExpectedImprovementWithoutCost':
        Optimizer.run_until_budget(
            budget = budget, 
            acquisition_function_class = ExpectedImprovement
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
            budget = budget, 
            acquisition_function_class = BudgetedMultiStepLookaheadEI
        )
    elif policy == 'MultiFidelityMaxValueEntropy':
        Optimizer.run_until_budget(
            budget = budget, 
            acquisition_function_class = qMultiFidelityMaxValueEntropy
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
    elif policy == 'GittinsDecay_InitLambda_0001':
        Optimizer.run_until_budget(
            budget = budget, 
            acquisition_function_class=GittinsIndex,
            step_divide = True,
            init_lmbda = 0.0001,
            alpha = 2
        )
    cost_history = Optimizer.get_cost_history()
    best_history = Optimizer.get_best_history()
    regret_history = Optimizer.get_regret_history(global_optimum_value/objective_scale_factor)
    acq_history = Optimizer.get_acq_history()

    print("Cost history:", cost_history)
    print("Best history:", best_history)
    print("Regret history:", regret_history)
    print("Acquisition history:", acq_history)
    print()

    return (budget, objective_scale_factor, cost_history, best_history, regret_history, acq_history)

wandb.init()
(budget, objective_scale_factor, cost_history, best_history, regret_history, acq_history) = run_bayesopt_experiment(wandb.config)

for cost, best, regret, acq in zip(cost_history, best_history, regret_history, acq_history):
    wandb.log({"raw cumulative cost": cost, "raw best observed": objective_scale_factor*best, "raw regret": -objective_scale_factor*regret, "raw log(regret)":np.log10(-objective_scale_factor)+np.log10(regret), "acq": acq})

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