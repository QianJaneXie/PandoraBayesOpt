#!/usr/bin/env python3

from typing import Callable, Optional, Union
import torch
from torch import Tensor
from torch.autograd import (Function, grad)
from botorch.acquisition import AnalyticAcquisitionFunction
from botorch.models.model import Model
from botorch.acquisition.analytic import (_scaled_improvement, _ei_helper)
from botorch.acquisition.objective import PosteriorTransform
from botorch.utils.transforms import t_batch_mode_transform
from botorch.utils.probability.utils import (
    ndtr as Phi,
    phi,
)
from gittins import GittinsIndexFunction, GittinsIndex


# Set default tensor type to float64
torch.set_default_dtype(torch.float64)

class GittinsIndex(AnalyticAcquisitionFunction):
    r"""Single-outcome/Two-outcome Gittins Index (analytic).

    Computes Gittins index using the analytic formula for a Normal posterior distribution. Unlike the
    MC-based acquisition functions, this relies on the posterior at single test
    point being Gaussian (and require the posterior to implement `mean` and
    `variance` properties). Only supports the case of `q=1`. The model must be
    single-outcome.

    `GI(x) = argmin_g |E(max(f(x) - g, 0))-lmbda * c(x)|,`

    where the expectation is taken over the value of stochastic function `f` at `x`.

    Example:
        Uniform-cost:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> GI = GittinsIndex(model, lmbda=0.0001)
        >>> gi = GI(test_X)
        
        Varing-cost:
        >>> def cost_function(x):
        >>>     return 1+20*x.mean(dim=-1))
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> GI = GittinsIndex(model, lmbda=0.0001, cost=cost_function)
        >>> gi = GI(test_X)

        Unknown-cost:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> GI = GittinsIndex(model, lmbda=0.0001, cost=cost_function, unknown_cost=True)
        >>> gi = GI(test_X)
    """

    def __init__(
        self,
        model: Model,
        lmbda: float,
        posterior_transform: Optional[PosteriorTransform] = None,
        maximize: bool = True,
        bound: torch.Tensor = torch.tensor([[-1.0], [1.0]], dtype=torch.float64),
        eps: float = 1e-6,
        cost: Optional[Callable] = None,
        unknown_cost: bool = False,
        bisection_early_stopping: bool = False
    ):
        r"""Single-outcome/Two-outcome Gittins Index (analytic).
        
        Args:
            model: A fitted single-outcome model or a fitted two-outcome model, 
                where the first output corresponds to the objective 
                and the second one to the log-cost.
            lmbda: A scalar representing the cost-per-sample or the scaling factor of the cost function.
            posterior_transform: A PosteriorTransform. If using a multi-output model,
                a PosteriorTransform that transforms the multi-output posterior into a
                single-output posterior is required.
            maximize: If True, consider the problem a maximization problem.
            cost: A callable cost function. If None, consider the problem a uniform-cost problem.
            unknown_cost: If True, consider the problem an unknown-cost problem.
            bound: A `2 x d` tensor of lower and upper bound for each column of `X`.
        """
        # use AcquisitionFunction constructor to avoid check for objective
        super(AnalyticAcquisitionFunction, self).__init__(model=model)
        self.lmbda = lmbda
        self.posterior_transform = posterior_transform
        self.maximize = maximize
        self.bound = bound
        self.eps = eps
        self.cost = cost if cost is not None else 1.0
        self.unknown_cost = unknown_cost
        self.bisection_early_stopping = bisection_early_stopping
      
        
    @t_batch_mode_transform(expected_q=1, assert_output_shape=False)
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate Gittins Index on the candidate set X using bisection method.

        Args:
            X: A `(b1 x ... bk) x 1 x d`-dim batched tensor of `d`-dim design points.
                Gittins Index is computed for each point individually,
                i.e., what is considered are the marginal posteriors, not the joint.

        Returns:
            A `(b1 x ... bk)`-dim tensor of Gittins Index values at the
            given design points `X`.
        """
        
        if self.unknown_cost:
            # Handling the unknown cost scenario
            posterior = self.model.posterior(X)
            means = posterior.mean  # (b) x 2
            vars = posterior.variance.clamp_min(1e-6)  # (b) x 2
            stds = vars.sqrt()

            mean_obj = means[..., 0].squeeze(dim=-1)
            std_obj = stds[..., 0].squeeze(dim=-1)

            mgf = (torch.exp(means[..., 1]) + 0.5 * vars[..., 1]).squeeze(dim=-1)

            gi_value = GittinsIndexFunction.apply(X, mean_obj, std_obj, self.lmbda, self.maximize, self.bound, self.eps, mgf, self.bisection_early_stopping)

        else:
            # Handling the known cost scenario
            mean, sigma = self._mean_and_sigma(X)

            if callable(self.cost):
                cost_X = self.cost(X).view(mean.shape)
            else:
                cost_X = torch.ones_like(mean)

            gi_value = GittinsIndexFunction.apply(X, mean, sigma, self.lmbda, self.maximize, self.bound, self.eps, cost_X, self.bisection_early_stopping)

        # If maximizing, return the GI value as is; if minimizing, return its negative
        return gi_value if self.maximize else -gi_value