import torch
from torch import Tensor
from botorch.acquisition import AcquisitionFunction
from botorch.models import FixedNoiseGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms import Standardize
from acquisition import GittinsIndex

# Set default tensor type to float64
torch.set_default_dtype(torch.float64)


# Custom function to fit the GP model
def fit_gp_model(X: Tensor, Y: Tensor, Yvar=None):
    if X.ndim == 1:
        X = X.unsqueeze(dim=-1)
    if Y.ndim == 1:
        Y = Y.unsqueeze(dim=-1)
    model = FixedNoiseGP(
        X, Y, torch.ones(Y.shape) * 1e-4, outcome_transform=Standardize(m=Y.shape[-1])
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)
    return model


# TODO: Replace cost with a cost_fn. Gittins will need to be updated to take in
# the cost function instead of a float. We can also consider trying to abstract
# this to take in any acquisition function and arguments.
def run_acquisition_gittins(
    X: Tensor,
    Y: Tensor,
    lmbda: float,
    acq_fn: AcquisitionFunction,
    maximize: bool,
    q: int,
    cost: float,
    num_itr: int,
    bound: Tensor,
):
    """Runs Bayesian optimization with q = 1 Gittins function.

    Args:
        X: `n x d` `Tensor` of datapoints. First dim is the number of points and
            second dim is the dimension of each data point.
        Y: `n x 1` tensor of objective values, corresponding to each datapoint
            in X.
        acq_fn: An `AcquisitionFunction`. Currently this only supports
            `GittinsIndex`.
        maximize: Whether we are maximizing or minimizing.
        q: How many points to query at each step.
        cost: The cost of sampling a point.
        num_itr: The number of points to sample.
        bound: A `2 x d` `Tensor` of lower and upper bound for each feature of
            `X`.

    Returns:
        3-tuple that has a `Tensor` of sampled points, a `Tensor` of values at
        those points, and the list of the best value seen so far.
    """
    # Unsqueeze the 2nd-to-last dimension to get a n x q x d tensor.
    test_X = X.unsqueeze(-2, dim=q)
    test_Y = Y.unsqueeze(-2, dim=q)
    x = X[0]
    y = Y[0]

    best_f = y.item()
    best_so_far = [best_f]

    for _ in range(num_itr):
        # Fit the model.
        model = fit_gp_model(x.detach(), y.detach(), Yvar=None)

        # Optimize the Gittins index.
        gi = acq_fn(model=model, cost=cost, lmbda=lmbda, maximize=maximize, bound=bound)
        # TODO: Do we need a copy of test_X/test_Y here?
        acq = gi.forward(test_X)

        if maximize:
            new_index = torch.argmax(acq).item()
            new_point = test_X[new_index]
            best_f = max(best_f, test_Y[new_index].item())
        else:
            new_index = torch.argmin(acq).item()
            new_point = test_X[new_index]
            best_f = min(best_f, test_Y[new_index].item())

        # Add new data point.
        x = torch.cat((x, new_point))
        y = torch.cat((y, test_Y[new_index]))

        best_so_far.append(best_f)

    return x, y, best_so_far
