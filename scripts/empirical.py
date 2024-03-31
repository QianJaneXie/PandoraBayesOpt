#!/usr/bin/env python
# coding: utf-8

import os
import sys
import scipy.io
import numpy as np
import pandas as pd
# from hpobench.benchmarks.ml.tabular_benchmark import TabularBenchmark
# from ConfigSpace.hyperparameters import OrdinalHyperparameter
# from ConfigSpace.configuration_space import ConfigurationSpace
from pandora_bayesopt.test_functions.lunar_lander import LunarLanderProblem
from pandora_bayesopt.test_functions.pest_control import PestControl

import torch
from typing import Dict
from pandora_bayesopt.utils import fit_gp_model
from botorch.utils.sampling import draw_sobol_samples
from botorch.sampling.pathwise import draw_matheron_paths, draw_kernel_feature_paths
from botorch.utils.sampling import optimize_posterior_samples
from botorch.acquisition import ExpectedImprovement
from pandora_bayesopt.acquisition.multi_step_ei import MultiStepLookaheadEI
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
from botorch.acquisition.predictive_entropy_search import qPredictiveEntropySearch
from pandora_bayesopt.acquisition.gittins import GittinsIndex
from pandora_bayesopt.bayesianoptimizer import BayesianOptimizer

import matplotlib.pyplot as plt
import wandb


# use a GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set default tensor type to float64
torch.set_default_dtype(torch.float64)

def run_bayesopt_experiment(config):
    print(config)

    problem = config['problem']
    # if problem == "FreeSolv":
    #     dim = 3
        
    #     script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
    #     df = pd.read_csv(script_dir + "/data/freesolv/freesolv_NN_rep3dim.csv")
    #     data_x_3D = torch.tensor(df[["x1", "x2", "x3"]].values, dtype=torch.float64)
    #     data_y = torch.tensor(df["expt"], dtype=torch.float64)

    #     # Fit the GP model using the data
    #     torch.manual_seed(0)
    #     objective_model = fit_gp_model(data_x_3D, data_y, input_standardize=True)
        
    #     # Define the objective function
    #     scaled_constant = 10
    #     def objective_function(X):
    #         if X.ndim == 1:
    #             X = X.unsqueeze(0)
    #         posterior_X = objective_model.posterior(X)
    #         objective_X = posterior_X.mean.detach()
    #         return objective_X/scaled_constant
        
    #     global_optimum_value = 25.591700836662685

    #     num_iterations = 80

    # if problem == "NN":
    #     dim = 5
    #     scaled_constant = -1
    #     global_optimum_value = 0.08956228956228955
    #     num_iterations = 60

    #     benchmark = TabularBenchmark('nn', 31)

    #     def find_nearest_ordinal(value: float, hyperparameter_type: OrdinalHyperparameter):
    #         # Convert the sequence to a PyTorch tensor
    #         valid_values = torch.tensor(hyperparameter_type.sequence)
            
    #         # Calculate the nearest value using torch operations
    #         nearest = torch.argmin((valid_values - value) ** 2).item()
    #         order = hyperparameter_type.get_seq_order()

    #         return hyperparameter_type.get_value(order[nearest])

    #     def round_to_valid_config(values: Dict[str, float], space: ConfigurationSpace):
    #         # Iterate over hyperparameters using a dictionary comprehension
    #         return {hyperparameter.name: find_nearest_ordinal(values[hyperparameter.name], hyperparameter) for hyperparameter in space.get_hyperparameters()}

    #     def objective_function(values: torch.Tensor):
    #         results = []
    #         for i in range(values.size(0)):  # Looping over each point
    #             config = {
    #                 "alpha": 10 ** ((values[i, 0].detach().item() - 1) * 8),
    #                 "batch_size": 2 ** (values[i, 1].detach().item() * 8),
    #                 "depth": int(2 * values[i, 2].detach().item() + 1),  # Ensuring depth is an integer
    #                 "learning_rate_init": 10 ** ((values[i, 3].detach().item() - 1) * 5),
    #                 "width": 2 ** (values[i, 4].detach().item() * 10)
    #             }

    #             result = benchmark.objective_function(round_to_valid_config(config, benchmark.configuration_space))
    #             results.append(result['function_value'])

    #         return torch.tensor(results)/scaled_constant  # Convert the list of results to an N*1 tensor

    if problem == "LunarLander":
        dim = 12
        num_iterations = 40*dim
        def objective_function(X):
            return LunarLanderProblem()(X)
        
    if problem == "PestControl":
        dim = 25
        num_iterations = 4*dim
        def objective_function(X):
            choice_X = torch.floor(5*X)
            choice_X[choice_X == 5] = 4
            return PestControl(negate=True)(choice_X)

    seed = config['seed']
    torch.manual_seed(seed)
    draw_initial_method = config['draw_initial_method']
    if draw_initial_method == 'sobol':
        bounds = torch.stack([torch.zeros(dim), torch.ones(dim)])
        init_x = draw_sobol_samples(bounds=bounds, n=1, q=2*(dim+1)).squeeze(0)
    input_standardize = config['input_normalization']
    maximize = True

    # Test performance of different policies
    policy = config['policy']
    print("policy:", policy)
    
    Optimizer = BayesianOptimizer(
        objective=objective_function, 
        dim=dim, 
        maximize=maximize, 
        initial_points=init_x, 
        input_standardize=input_standardize
    )
    if policy == 'RandomSearch':
        Optimizer.run(
            num_iterations=num_iterations, 
            acquisition_function_class="RandomSearch"
        )
    elif policy == 'ExpectedImprovement':
        Optimizer.run(
            num_iterations=num_iterations, 
            acquisition_function_class=ExpectedImprovement
        )
    elif policy == 'ThompsonSampling':
        Optimizer.run(
            num_iterations=num_iterations, 
            acquisition_function_class="ThompsonSampling"
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
    # regret_history = Optimizer.get_regret_history(global_optimum_value/scaled_constant)

    print("Cost history:", cost_history)
    print("Best history:", best_history)
    # print("Regret history:", regret_history)
    print()

    # return (scaled_constant, cost_history, best_history, regret_history)
    return (cost_history, best_history)

wandb.init(resume=True)

# (scaled_constant, cost_history, best_history, regret_history) = run_bayesopt_experiment(wandb.config)
(cost_history, best_history) = run_bayesopt_experiment(wandb.config)

# for cost, best, regret in zip(cost_history, best_history, regret_history):
#     wandb.log({"cumulative cost": cost, "best observed": scaled_constant*best, "regret": -scaled_constant*regret, "log(regret)":np.log10(-scaled_constant)+np.log10(regret)})

for cost, best in zip(cost_history, best_history):
    wandb.log({"cumulative cost": cost, "best observed": best})

wandb.finish()