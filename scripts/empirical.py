#!/usr/bin/env python
# coding: utf-8

from pandora_bayesopt.test_functions.lunar_lander import LunarLanderProblem
from pandora_bayesopt.test_functions.pest_control import PestControl
from pandora_bayesopt.test_functions.robot_pushing.robot_pushing import robot_pushing_3d, robot_pushing_4d, robot_pushing_14d

import torch
from typing import Dict
from botorch.utils.sampling import draw_sobol_samples
from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound
from pandora_bayesopt.acquisition.multi_step_ei import MultiStepLookaheadEI
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
from botorch.acquisition.predictive_entropy_search import qPredictiveEntropySearch
from pandora_bayesopt.acquisition.gittins import GittinsIndex
from pandora_bayesopt.bayesianoptimizer import BayesianOptimizer

import numpy as np
import wandb


# use a GPU if available
torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set default tensor type to float64
torch.set_default_dtype(torch.float64)

def run_bayesopt_experiment(config):
    print(config)

    problem = config['problem']
    print("problem:", problem)

    if problem == "LunarLander":
        dim = 12
        num_iterations = 200
        def objective_function(X):
            objective_X, cost_X = LunarLanderProblem()(X)
            return objective_X
        
    if problem == "PestControl":
        dim = 25
        num_iterations = 100
        def objective_function(X):
            choice_X = torch.floor(5*X)
            choice_X[choice_X == 5] = 4
            return PestControl(negate=True)(choice_X)
        
    if problem == "RobotPushing4D":
        dim = 4
        num_iterations = 40
        target_location = torch.tensor([-4.4185, -4.3709])
        def unnorm_X(X: torch.Tensor) -> torch.Tensor:
            X_unnorm = X.clone()
            # Check if the tensor is higher than 3-dimensional
            if X.dim() > 3:
                # Assuming the extra unwanted dimension is at position 1 (the second position)
                X_unnorm = X_unnorm.view(-1, X.size(-1))  # Remove the singleton dimension
            # Check if the tensor is 3-dimensional
            if X.dim() == 3:
                # Assuming the extra unwanted dimension is at position 1 (the second position)
                X_unnorm = X_unnorm.squeeze(1)  # Remove the singleton dimension
            elif X.dim() == 1:
                # If 1-dimensional, add a dimension to make it 2D (e.g., for batch size of 1)
                X_unnorm = X_unnorm.unsqueeze(0)
            X_unnorm[:, :2] = 10.0 * X_unnorm[:, :2] - 5.0
            X_unnorm[:, 2] = 29.0 * X_unnorm[:, 2] + 1.0
            X_unnorm[:, 3] = 2 * 3.14 * X_unnorm[:, 3]
            return X_unnorm

        def objective_function(X: torch.Tensor) -> torch.Tensor:
            X_unnorm = unnorm_X(X)
            objective_X = []
            for x in X_unnorm:
                # Set the seed based on X to ensure consistent randomness
                np.random.seed(0)
                object_location = torch.tensor(robot_pushing_4d(x[0].item(), x[1].item(), x[2].item(), x[3].item()))
                objective_X.append(-torch.dist(target_location, object_location))
            np.random.seed()  # Reset the seed
            return torch.tensor(objective_X)
        
    if problem == "RobotPushing14D":
        dim = 14
        num_iterations = 200
        target_location = torch.tensor([-4.4185, -4.3709])
        target_location2 = torch.tensor([-3.7641, -4.4742])
        def unnorm_X(X: torch.Tensor) -> torch.Tensor:
            X_unnorm = X.clone()
            # Check if the tensor is higher than 3-dimensional
            if X.dim() > 3:
                # Assuming the extra unwanted dimension is at position 1 (the second position)
                X_unnorm = X_unnorm.view(-1, X.size(-1))  # Remove the singleton dimension
            # Check the dimensionality of X and adjust accordingly
            if X.dim() == 3:
                # Remove the singleton dimension assuming it's the second one
                X_unnorm = X_unnorm.squeeze(1)
            elif X.dim() == 1:
                # If 1-dimensional, add a dimension to make it 2D (e.g., for batch size of 1)
                X_unnorm = X_unnorm.unsqueeze(0)
            X_unnorm[:, :2] = 10.0 * X_unnorm[:, :2] - 5.0
            X_unnorm[:, 2:4] = 5 * X_unnorm[:, 2:4]
            X_unnorm[:, 4] = 29.0 * X_unnorm[:, 4] + 1.0
            X_unnorm[:, 5] = 2 * np.pi * X_unnorm[:, 5]
            X_unnorm[:, 6:8] = 10.0 * X_unnorm[:, 6:8] - 5.0
            X_unnorm[:, 8:10] = 5 * X_unnorm[:, 8:10]
            X_unnorm[:, 10] = 29 * X_unnorm[:, 10] + 1.0
            X_unnorm[:, 11] = 2 * np.pi * X_unnorm[:, 11]
            X_unnorm[:, 12:] = 2 * np.pi * X_unnorm[:, 12:]

            return X_unnorm

        def objective_function(X: torch.Tensor) -> torch.Tensor:
            X_unnorm = unnorm_X(X)
            objective_X = []
            for x in X_unnorm:
                # Set the seed based on X to ensure consistent randomness
                np.random.seed(0)
                object_location, object_location2, robot_location, robot_location2 = torch.tensor(robot_pushing_14d(x[0].item(), x[1].item(), x[2].item(), x[3].item(), x[4].item(), x[5].item(), x[6].item(), x[7].item(), x[8].item(), x[9].item(), x[10].item(), x[11].item(), x[12].item(), x[13].item()))
                objective_X.append(-torch.dist(target_location, object_location)-torch.dist(target_location2, object_location2))
            np.random.seed()  # Reset the seed
            return torch.tensor(objective_X)

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
    elif policy == 'UpperConfidenceBound':
        Optimizer.run(
            num_iterations=num_iterations, 
            acquisition_function_class=UpperConfidenceBound,
            heuristic=True
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
    runtime_history = Optimizer.get_runtime_history()

    print("Cost history:", cost_history)
    print("Best history:", best_history)
    print("Runtime history:", runtime_history)
    print()

    return cost_history, best_history, runtime_history

wandb.init(resume=True)

cost_history, best_history, runtime_history = run_bayesopt_experiment(wandb.config)

for cost, best, runtime in zip(cost_history, best_history, runtime_history):
    wandb.log({"cumulative cost": cost, "best observed": best, "run time": runtime})

wandb.finish()