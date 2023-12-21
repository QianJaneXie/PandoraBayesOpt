import torch
from botorch.models import FixedNoiseGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.constraints import GreaterThan
from botorch.models.transforms import Standardize

import numpy as np
from scipy.optimize import minimize

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


def find_global_optimum(objective, dim, maximize, num_restarts=None, method='L-BFGS-B'):
    """
    Find the global optimum using multi-start optimization.

    Parameters:
    - objective (function): The objective function to optimize.
    - dim (int): The number of dimensions
    - maximize (bool): If True, maximizes the objective; otherwise, minimizes.
    - num_restarts (int): Number of starting points for the optimization.

    Returns:
    - float: The global optimum found.
    """

    def scipy_objective(x):
        x_tensor = torch.tensor(x, dtype=torch.float64)
        return -objective(x_tensor) if maximize else objective(x_tensor)

    scipy_bounds = list(zip(np.zeros(dim), np.ones(dim)))
    
    best_result = None

    if num_restarts == None:
        num_restarts = 200*dim

    for _ in range(num_restarts):
        # Generate a random initial guess within the bounds
        initial_guess = torch.rand(dim)

        # Run the optimization
        result = minimize(scipy_objective, initial_guess, method=method, bounds=scipy_bounds)

        # Update the best result if this result is better
        if best_result is None or result.fun < best_result.fun:
            best_result = result

    global_optimum = scipy_objective(best_result.x).item()

    return -global_optimum if maximize else global_optimum
