import torch
from botorch.models import FixedNoiseGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.constraints import GreaterThan
from botorch.models.transforms import Standardize

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