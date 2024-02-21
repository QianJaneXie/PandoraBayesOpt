import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood

from botorch.acquisition import AcquisitionFunction
from botorch.optim import optimize_acqf

import numpy as np
from scipy.optimize import differential_evolution

def fit_gp_model(X, Y, kernel=None, Yvar=None, input_standardize=False, noise_level=1e-4):
    # Ensure X is a 2D tensor [num_data, num_features]
    if X.ndim == 1:
        X = X.unsqueeze(dim=-1)
    
    # Ensure Y is a 2D tensor [num_data, 1]
    if Y.ndim == 1:
        Y = Y.unsqueeze(dim=-1)
        
    if Yvar is None:
        Yvar = torch.ones(len(Y)) * noise_level

    # Outcome transform
    if input_standardize == True:
        outcome_transform = Standardize(m=Y.shape[-1])
    else:
        outcome_transform = None
   
    model = SingleTaskGP(train_X=X, train_Y=Y, likelihood = FixedNoiseGaussianLikelihood(noise=Yvar), covar_module=kernel, outcome_transform=outcome_transform)

    if input_standardize == True:
        model.outcome_transform.eval()
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)
    return model

class ObjectiveAcquisitionFunction(AcquisitionFunction):
    def __init__(self, model, objective, maximize=True):
        super().__init__(model)
        self.objective = objective
        self.maximize = maximize

    def forward(self, X):
        if self.maximize:
            return self.objective(X)
        else:
            return -self.objective(X)

def find_global_optimum(objective, dim, maximize, raw_samples=None, method='L-BFGS-B'):
    """
    Find the global optimum using multi-start optimization.
    Parameters:
    - objective (function): The objective function to optimize.
    - dim (int): The number of dimensions
    - maximize (bool): If True, maximizes the objective; otherwise, minimizes.
    - raw_samples (int): Number of raw samples for the optimization.
    - method: Optimization method; e.g., 'L-BFGS-B'
    Returns:
    - float: The global optimum found.
    """

    # Define the dummy model
    model = None
    
    # If raw_samples is None, set a default value based on the dimension
    if raw_samples is None:
        raw_samples = 10 * dim

    # Initialize the acquisition function
    obj_acqf = ObjectiveAcquisitionFunction(model=model, objective=objective, maximize=maximize)

    # Define bounds based on the dimension
    bounds = torch.stack([torch.zeros(dim), torch.ones(dim)])

    # Optimize the acquisition function
    global_optimum_point, _ = optimize_acqf(
        acq_function=obj_acqf,
        bounds=bounds,
        q=1,
        num_restarts=1,
        raw_samples=1024*dim,
        options={'method': method},
    )

    # Evaluate the objective function at the optimum point to get the true value
    global_optimum_value = objective(global_optimum_point).item()

    return global_optimum_point.squeeze(-1).squeeze(-1), global_optimum_value

def find_global_optimum_scipy(objective, dim, maximize):
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
    global_optimum_point = result.x
    global_optimum_value = scipy_objective(result.x).item()
    return global_optimum_point, -global_optimum_value if maximize else global_optimum_value