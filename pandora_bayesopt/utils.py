from typing import Optional
import torch
from torch import Tensor
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood

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
        input_standardize: bool = False, 
        noise_level: float = 1e-4
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