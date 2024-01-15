import torch
from gpytorch.kernels import MaternKernel, ScaleKernel
from botorch.utils.gp_sampling import get_deterministic_model, RandomFourierFeatures

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood

import numpy as np
from scipy.optimize import differential_evolution

def create_objective_model(seed, dim, nu, lengthscale, outputscale=1.0, num_rff_features=1280):
    """
    Create and return the objective model for sampling from a Matern kernel.

    Parameters:
    - seed (int): Random seed for reproducibility. E.g., 42.
    - dim (int): Number of dimensions of the sample space.
    - nu (float): Smoothness parameter for the Matern kernel. E.g., 0.5, 1.5, 2.5.
    - lengthscale (float): Lengthscale parameter for the Matern kernel.
    - outputscale (float): Outputscale parameter for the Matern kernel. E.g., 1.0.
    - num_rff_features (int): Number of random Fourier features. E.g., 1280.

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

def create_objective_function(seed, dim, nu, lengthscale, outputscale=1.0, num_rff_features=1280):
    
    """
    Create and return the objective function sampled from a Matern kernel.
    
    Parameters:
    - seed (int): Random seed for reproducibility. E.g., 42.
    - dim (int): Number of dimensions of the sample space.
    - nu (float): Smoothness parameter for the Matern kernel. E.g., 0.5.
    - lengthscale (float): Lengthscale parameter for the Matern kernel.
    - outputscale (float): Outputscale parameter for the Matern kernel. E.g., 1.0.
    - num_rff_features (int): Number of random Fourier features. E.g., 1280.

    Returns:
    - objective_model: The model used to generate the objective function.
    """
    
    # Create the objective model inside the closure
    objective_model = create_objective_model(
        seed=seed, 
        dim=dim, 
        nu=nu, 
        lengthscale=lengthscale,
        outputscale=outputscale, 
        num_rff_features=num_rff_features
    )

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
                
        return objective_model.posterior(X).mean.detach().squeeze(-1)

    return objective

def fit_gp_model(X, Y, kernel, Yvar=None, noise_level=1e-4):
    # Ensure X is a 2D tensor [num_data, num_features]
    if X.ndim == 1:
        X = X.unsqueeze(dim=-1)
    
    # Ensure Y is a 2D tensor [num_data, 1]
    if Y.ndim == 1:
        Y = Y.unsqueeze(dim=-1)
        
    if Yvar is None:
        Yvar = torch.ones(len(Y)) * noise_level
        
    model = SingleTaskGP(train_X=X, train_Y=Y, likelihood = FixedNoiseGaussianLikelihood(noise=Yvar), covar_module=kernel)

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