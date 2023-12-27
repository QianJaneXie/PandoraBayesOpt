import torch
from gpytorch.kernels import MaternKernel, ScaleKernel
from botorch.utils.gp_sampling import get_deterministic_model, RandomFourierFeatures

from botorch.models import FixedNoiseGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.constraints import GreaterThan
from botorch.models.transforms import Standardize

import numpy as np
from scipy.optimize import differential_evolution

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

def create_objective_function(dim, lengthscale, outputscale, nu, num_rff_features, seed):
    
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
    objective_model = create_objective_model(dim, lengthscale, outputscale, nu, num_rff_features, seed)

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

def fit_gp_model(X, Y, nu=2.5, lengthscale=1.0, outputscale=1.0, Yvar=None):
    if X.ndim == 1:
        X = X.unsqueeze(dim=-1)
    if Y.ndim == 1:
        Y = Y.unsqueeze(dim=-1)
    if Yvar is None:
        Yvar = torch.ones(Y.shape) * 1e-4

    model = FixedNoiseGP(X, Y, Yvar, outcome_transform=Standardize(m=Y.shape[-1]))
    
    # Set the nu parameter for the Matern kernel: 1/2, 3/2, or 5/2
    model.covar_module.base_kernel.nu = nu
    
    # Set the length scale of the kernel
    model.covar_module.base_kernel.lengthscale = lengthscale

    # Set the output scale (alpha0)
    model.covar_module.outputscale = outputscale

    # Enforce positive constraints
    model.covar_module.base_kernel.raw_lengthscale_constraint = GreaterThan(1e-5)
    model.covar_module.raw_outputscale_constraint = GreaterThan(1e-5)
    
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)
    return model


def find_global_optimum(objective, dim, maximize):
    """
    Find the global optimum using differential evolution.

    Parameters:
    - objective (function): The objective function to optimize.
    - dim (int): The number of dimensions.
    - maximize (bool): If True, maximizes the objective; otherwise, minimizes.

    Returns:
    - float: The global optimum found.
    """

    def scipy_objective(x):
        x_tensor = torch.tensor(x, dtype=torch.float64)
        return -objective(x_tensor) if maximize else objective(x_tensor)

    scipy_bounds = list(zip(np.zeros(dim), np.ones(dim)))

    # Run differential evolution
    result = differential_evolution(scipy_objective, scipy_bounds)

    global_optimum = scipy_objective(result.x).item()

    return -global_optimum if maximize else global_optimum