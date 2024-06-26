#!/usr/bin/env python3

from typing import Optional
import torch
from torch import Tensor
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
from gpytorch.likelihoods.gaussian_likelihood import GaussianLikelihood
from gpytorch.constraints.constraints import Interval

from botorch.acquisition import AcquisitionFunction
from botorch.optim import optimize_acqf

from gpytorch.kernels import MaternKernel, ScaleKernel
from botorch.utils.gp_sampling import get_deterministic_model, RandomFourierFeatures

def fit_gp_model(
        X: Tensor, 
        objective_X: Tensor, 
        cost_X: Optional[Tensor] = None, 
        unknown_cost: bool = False, 
        kernel: Optional[torch.nn.Module] = None, 
        Yvar: Optional[torch.Tensor] = None, 
        noise_level: float = 1e-4,
        gaussian_likelihood: bool = False,
        input_standardize: bool = False,
    ):
    # Ensure X is a 2D tensor [num_data, num_features]
    if X.ndim == 1:
        X = X.unsqueeze(dim=-1)
    
    # Ensure objective_X is a 2D tensor [num_data, 1]
    if objective_X.ndim == 1:
        objective_X = objective_X.unsqueeze(dim=-1)

    # Ensure cost_X is a 2D tensor [num_data, 1]
    if unknown_cost == True:
        if cost_X.ndim == 1:
            log_cost_X = torch.log(cost_X).unsqueeze(dim=-1)

    Y = torch.cat((objective_X, log_cost_X), dim=-1) if unknown_cost else objective_X
        
    if gaussian_likelihood:
        _, aug_batch_shape = SingleTaskGP.get_batch_dimensions(
                train_X=X,
                train_Y=Y,
            )
        likelihood = GaussianLikelihood(
                batch_shape=aug_batch_shape,
                noise_constraint=Interval(lower_bound=noise_level, upper_bound=10*noise_level),
            )

    else:
        if Yvar is None:
            Yvar = torch.ones(len(Y)) * noise_level
        
        likelihood = FixedNoiseGaussianLikelihood(noise=Yvar)

    # Outcome transform
    if input_standardize == True:
        outcome_transform = Standardize(m=Y.shape[-1])
    else:
        outcome_transform = None
   
    model = SingleTaskGP(train_X=X, train_Y=Y, likelihood = likelihood, covar_module=kernel, outcome_transform=outcome_transform)

    if input_standardize == True:
        model.outcome_transform.eval()
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)
    return model


def create_objective_model(dim, nu, lengthscale, outputscale=1.0, num_rff_features=1280):
    """
    Create and return the objective model for sampling from a Matern kernel.
    Parameters:
    - dim (int): Number of dimensions of the sample space.
    - nu (float): Smoothness parameter for the Matern kernel. E.g., 0.5, 1.5, 2.5.
    - lengthscale (float): Lengthscale parameter for the Matern kernel.
    - outputscale (float): Outputscale parameter for the Matern kernel. E.g., 1.0.
    - num_rff_features (int): Number of random Fourier features. E.g., 1280.
    Returns:
    - objective_model: The model used to generate the objective function.
    """
    # Set up the Matern kernel
    base_kernel = MaternKernel(nu=nu).double()
    base_kernel.lengthscale = torch.tensor([[lengthscale]])
    scale_kernel = ScaleKernel(base_kernel).double()
    scale_kernel.outputscale = torch.tensor([[outputscale]])
    # Random Fourier Features
    rff = RandomFourierFeatures(
        kernel=scale_kernel,
        input_dim=dim,
        num_rff_features=num_rff_features
    )
    # Generate weights for the Random Fourier Features
    weights = torch.randn(num_rff_features)
    objective_model = get_deterministic_model(weights=[weights], bases=[rff])
    return objective_model


def create_objective_function(dim, nu, lengthscale, outputscale=1.0, num_rff_features=1280):
    
    """
    Create and return the objective function sampled from a Matern kernel.
    
    Parameters:
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