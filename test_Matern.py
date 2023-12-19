#!/usr/bin/env python
# coding: utf-8

# In[4]:


import torch


# In[5]:


# use a GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set default tensor type to float64
torch.set_default_dtype(torch.float64)


# In[15]:


import utils

# Re-import GittinsIndex from the reloaded module
from utils import fit_gp_model


# In[11]:


import torch
from gpytorch.kernels import MaternKernel, ScaleKernel
from botorch.utils.gp_sampling import get_deterministic_model, RandomFourierFeatures

def create_objective_model(dim, nu, lengthscale, outputscale, num_rff_features, seed):
    """
    Create and return the objective model for sampling from a Matern kernel.

    Parameters:
    - dim (int): Number of dimensions of the sample space.
    - nu (float): Smoothness parameter for the Matern kernel. E.g., 0.5.
    - lengthscale (float): Lengthscale parameter for the Matern kernel.
    - outputscale (float): Outputscale parameter for the Matern kernel.
    - num_rff_features (int): Number of random Fourier features. E.g., 1280.
    - seed (int): Random seed for reproducibility. E.g., 42.

    Returns:
    - objective_model: The model used to generate the objective function.
    """
    # Set the seed for reproducibility
    torch.manual_seed(seed)

    # Set up the Matern kernel
    base_kernel = MaternKernel(nu=nu).double()
    base_kernel.lengthscale = torch.tensor([[lengthscale]], dtype=torch.float64)
    scale_kernel = ScaleKernel(base_kernel).double()
    scale_kernel.outputscale = torch.tensor([[outputscale]], dtype=torch.float64)

    # Random Fourier Features
    rff = RandomFourierFeatures(
        kernel=scale_kernel,
        input_dim=dim,
        num_rff_features=num_rff_features
    )

    # Generate weights for the Random Fourier Features
    weights = torch.randn(num_rff_features, dtype=torch.float64)
    objective_model = get_deterministic_model(weights=[weights], bases=[rff])

    return objective_model


# In[12]:


def create_objective_function(num_dimensions, lengthscale, outputscale, nu=0.5, num_rff_features=1280, seed=42):
    
    """
    Create and return the objective function sampled from a Matern kernel.
    
    Parameters:
    - dim (int): Number of dimensions of the sample space.
    - nu (float): Smoothness parameter for the Matern kernel. E.g., 0.5.
    - lengthscale (float): Lengthscale parameter for the Matern kernel.
    - outputscale (float): Outputscale parameter for the Matern kernel.
    - num_rff_features (int): Number of random Fourier features. E.g., 1280.
    - seed (int): Random seed for reproducibility. E.g., 42.

    Returns:
    - objective_model: The model used to generate the objective function.
    """
    
    # Create the objective model inside the closure
    objective_model = create_objective_model(num_dimensions, lengthscale, outputscale, nu, num_rff_features, seed)

    # Define the objective function that only takes X
    def objective(X):
        """
        Evaluate the objective function using the provided model.

        Parameters:
        - X (Tensor): Input points where the objective function should be evaluated.
        - objective_model: The model used to evaluate the objective function.

        Returns:
        - Tensor: Evaluated mean of the model's posterior.
        """
        return objective_model.posterior(X).mean.detach()

    return objective


# In[13]:


import numpy as np

# Example Usage for 1D: 

# Create the objective model
dim = 1
nu = 0.5
lengthscale = 1/16
outputscale = 2.0
num_rff_features = 1280

# In[67]:

from botorch.acquisition import ExpectedImprovement
from acquisition import GittinsIndex
from bayesianoptimizer import BayesianOptimizer


# In[82]:


maximize = False
num_iterations = 24


# In[83]:


num_trials = 30  # Number of trials for each policy
num_iterations = 24  # Budget for cumulative cost

EI_best_histories = []
EI_cost_histories = []

lmbda_values = [0.1, 0.05, 0.01, 0.005, 0.001]
GI_best_histories = {lmbda: [] for lmbda in lmbda_values}
GI_cost_histories = {lmbda: [] for lmbda in lmbda_values}

for trial in range(num_trials):
    
    seed = trial
    
    print("seed:", seed)

    objective = create_objective_function(dim, nu, lengthscale, outputscale, num_rff_features, seed=seed)
    
    # Run trial with ExpectedImprovement
    EI_optimizer = BayesianOptimizer(objective=objective, dim=dim, maximize=maximize, seed=seed)
    EI_optimizer.run(num_iterations=num_iterations, acquisition_function_class=ExpectedImprovement)
    EI_best_histories.append(EI_optimizer.get_best_history())
    EI_cost_histories.append(EI_optimizer.get_cost_history())

    for lmbda in lmbda_values:
        print("lmbda:", lmbda)

        # Run trial with GittinsIndex for the current lambda
        GI_optimizer = BayesianOptimizer(objective=objective, dim=dim, maximize=maximize, seed=seed)
        GI_optimizer.run(num_iterations=num_iterations, acquisition_function_class=GittinsIndex, lmbda=lmbda)
        GI_best_histories[lmbda].append(GI_optimizer.get_best_history())
        GI_cost_histories[lmbda].append(GI_optimizer.get_cost_history())
        print()
    print()


import json

# Open a file and write the string to it
with open('EI best (Matern12).txt', 'w') as file:
    file.write(json.dumps(EI_best_histories, indent=4))
    
with open('GI best (Matern12).txt', 'w') as file:
    file.write(json.dumps(GI_best_histories, indent=4))
