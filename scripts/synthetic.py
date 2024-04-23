#!/usr/bin/env python
# coding: utf-8

import torch
from botorch.test_functions.synthetic import Ackley, Rosenbrock, Levy
from botorch.utils.sampling import draw_sobol_samples
from botorch.acquisition import ExpectedImprovement
from pandora_bayesopt.acquisition.multi_step_ei import MultiStepLookaheadEI
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
from botorch.acquisition.predictive_entropy_search import qPredictiveEntropySearch
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

    problem = config['problem']
    dim = config['dim']
    num_iterations = 10*dim
    seed = config['seed']
    torch.manual_seed(seed)
    input_standardize = config['input_normalization']
    draw_initial_method = config['draw_initial_method']
    policy = config['policy']
    print("policy:", policy)
    maximize = True

    if problem == 'Ackley':
        ackley_function = Ackley(dim=dim)
        scaled_constant = -1
        def objective_function(X):
            return ackley_function(2*X-1)/scaled_constant
        global_optimum_value = 0
    if problem == 'Rosenbrock':
        rosenbrock_function = Rosenbrock(dim=dim)
        scaled_constant = -1000
        def objective_function(X):
            return rosenbrock_function(15*X-5)/scaled_constant
        global_optimum_value = 0
    if problem == 'Levy':
        levy_function = Levy(dim=dim)
        scaled_constant = -100
        def objective_function(X):
            return levy_function(20*X-10)/scaled_constant
        global_optimum_value = 0

    # Test performance of different policies
    if draw_initial_method == 'sobol':
        bounds = torch.stack([torch.zeros(dim), torch.ones(dim)])
        init_x = draw_sobol_samples(bounds=bounds, n=1, q=2*(dim+1)).squeeze(0)

    Optimizer = BayesianOptimizer(
        dim=dim, 
        maximize=maximize, 
        initial_points=init_x,
        objective=objective_function,
        input_standardize=input_standardize
    )
    if policy == 'RandomSearch':
        Optimizer.run(
            num_iterations=num_iterations, 
            acquisition_function_class="RandomSearch"
        )
    elif policy == 'ThompsonSampling':
        Optimizer.run(
            num_iterations=num_iterations, 
            acquisition_function_class="ThompsonSampling"
        )
    elif policy == 'ExpectedImprovement':
        Optimizer.run(
            num_iterations=num_iterations, 
            acquisition_function_class=ExpectedImprovement
        )
    elif policy == 'PredictiveEntropySearch':
        Optimizer.run(
            num_iterations=num_iterations, 
            acquisition_function_class=qPredictiveEntropySearch
        )
    elif policy == 'KnowledgeGradient':
        Optimizer.run(
            num_iterations=num_iterations, 
            acquisition_function_class=qKnowledgeGradient
        )
    elif policy == 'MultiStepLookaheadEI':
        Optimizer.run(
            num_iterations=num_iterations, 
            acquisition_function_class=MultiStepLookaheadEI
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
    regret_history = Optimizer.get_regret_history(global_optimum_value/scaled_constant)

    print("Cost history:", cost_history)
    print("Best history:", best_history)
    print("Regret history:", regret_history)

    print()

    return (scaled_constant, cost_history, best_history, regret_history)

wandb.init()
(scaled_constant, cost_history, best_history, regret_history) = run_bayesopt_experiment(wandb.config)

for cost, best, regret in zip(cost_history, best_history, regret_history):
    wandb.log({"cumulative cost": cost, "best observed": scaled_constant*best, "regret": -scaled_constant*regret, "lg(regret)":np.log10(-scaled_constant)+np.log10(regret)})

wandb.finish()