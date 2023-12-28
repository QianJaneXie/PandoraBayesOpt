import torch
from torch import Tensor
from torch.autograd import (Function, grad)
from botorch.acquisition import AnalyticAcquisitionFunction, ExpectedImprovement
from botorch.models.model import Model
from botorch.acquisition.analytic import (_scaled_improvement, _ei_helper)
from botorch.acquisition.objective import PosteriorTransform
from typing import Dict, Optional, Tuple, Union
from botorch.utils.transforms import t_batch_mode_transform
from botorch.utils.probability.utils import (
    ndtr as Phi,
    phi,
)

# Set default tensor type to float64
torch.set_default_dtype(torch.float64)

class GittinsIndexFunction(Function):
    @staticmethod
        
    def forward(ctx, X: torch.Tensor, mean: torch.Tensor, sigma: torch.Tensor, lmbda: float, maximize: bool, bound: torch.Tensor, eps: float, cost_X):

        def cost_adjusted_expected_improvement(best_f):
            u = _scaled_improvement(mean, sigma, best_f, maximize)
            return sigma * _ei_helper(u) - lmbda * cost_X

        size = X.size()[0]
        l = bound[0]*torch.ones(size)
        h = bound[1]*torch.ones(size)
        m = (h + l) / 2

        if maximize:
            while torch.any(cost_adjusted_expected_improvement(best_f=l) < 0):
                l = 2 * l
            while torch.any(cost_adjusted_expected_improvement(best_f=h) > 0):
                h = 2 * h
        else:
            while torch.any(cost_adjusted_expected_improvement(best_f=l) > 0):
                l = 2 * l
            while torch.any(cost_adjusted_expected_improvement(best_f=h) < 0):
                h = 2 * h

        # Bisection method
#         while torch.max(torch.abs(cost_adjusted_expected_improvement(best_f=m))) >= eps:
        for _ in range(100):
            sgn_m = torch.sign(cost_adjusted_expected_improvement(best_f=m))
            if maximize:
                l = torch.where(sgn_m >= 0, m, l)
                h = torch.where(sgn_m <= 0, m, h)
            else:
                l = torch.where(sgn_m <= 0, m, l)
                h = torch.where(sgn_m >= 0, m, h)
            m = (h + l) / 2

        # Save u for backward computation
        u = _scaled_improvement(mean, sigma, m, maximize)
        
        # Save values needed in the backward pass
        ctx.save_for_backward(X, mean, sigma, u, cost_X)
        
        # Save boolean flag directly in ctx
        ctx.maximize = maximize

        # Save lmbda in ctx for later use in backward
        ctx.lmbda = lmbda
            
        return m
    
    @staticmethod
    def backward(ctx, grad_output):
                
        # Retrieve saved tensors
        X, mean, sigma, u, cost_X = ctx.saved_tensors
        maximize = ctx.maximize  # Retrieve the boolean flag directly from ctx
        lmbda = ctx.lmbda  # Retrieve lmbda

                
        # Gradient of mean function with respect to x
        dmean_dX = grad(outputs=mean, inputs=X, grad_outputs=torch.ones_like(mean), retain_graph=True, allow_unused=True)[0].clone()

        # Gradient of the std function with respect to x
        dsigma_dX = grad(outputs=sigma, inputs=X, grad_outputs=torch.ones_like(sigma), retain_graph=True, allow_unused=True)[0].clone()

        if cost_X.requires_grad:
            # Compute gradient only if cost_X is not a scalar
            dcost_dX = grad(outputs=cost_X, inputs=X, grad_outputs=torch.ones_like(cost_X), retain_graph=True, allow_unused=True)[0].clone()
        else:
            # If cost_X does not require grad, set its gradient to zero
            dcost_dX = torch.zeros_like(X)

        # Check if gradients are None and handle accordingly
        if dmean_dX is None or dsigma_dX is None or dcost_dX is None:
            raise RuntimeError("Gradients could not be computed for one or more components.")
        
        # Compute the gradient of the Gittins acquisition function
        if maximize:
            grad_X = grad_output.unsqueeze(-1).unsqueeze(-1) * (dmean_dX + (phi(u).unsqueeze(-1).unsqueeze(-1) * dsigma_dX - lmbda * dcost_dX) / Phi(u).unsqueeze(-1).unsqueeze(-1))
        else:
            grad_X = grad_output.unsqueeze(-1).unsqueeze(-1) * (dmean_dX - (phi(u).unsqueeze(-1).unsqueeze(-1) * dsigma_dX - lmbda * dcost_dX) / Phi(u).unsqueeze(-1).unsqueeze(-1))

        return grad_X, None, None, None, None, None, None, None

class GittinsIndex(AnalyticAcquisitionFunction):
    r"""Single-outcome Gittins Index (analytic).

    Computes Gittins index using the analytic formula for a Normal posterior distribution. Unlike the
    MC-based acquisition functions, this relies on the posterior at single test
    point being Gaussian (and require the posterior to implement `mean` and
    `variance` properties). Only supports the case of `q=1`. The model must be
    single-outcome.

    `GI(x) = argmin_g |E(max(f(x) - g, 0))-lmbda|,`

    where the expectation is taken over the value of stochastic function `f` at `x`.

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> GI = GittinsIndex(model, lmbda=0.05)
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
        cost = None
    ):
        r"""Single-outcome Gittins Index (analytic).
        
        Args:
            model: A fitted single-outcome model.
            lmbda: A scalar representing the Lagrangian multiplier of the budget constraint/cost function.
            cost: Either a scalar or a `b`-dim Tensor (batch mode) representing
                the cost function.
            posterior_transform: A PosteriorTransform. If using a multi-output model,
                a PosteriorTransform that transforms the multi-output posterior into a
                single-output posterior is required.
            maximize: If True, consider the problem a maximization problem.
            bound: A `2 x d` tensor of lower and upper bound for each column of `X`.
        """
        super().__init__(model=model, posterior_transform=posterior_transform)
        self.lmbda = lmbda
        self.maximize = maximize
        self.bound = bound
        self.eps = eps
        self.cost = cost if cost is not None else 1
      
        
    @t_batch_mode_transform(expected_q=1)
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
        
        mean, sigma = self._mean_and_sigma(X)

        if callable(self.cost):
            cost_X = self.cost(X).view(mean.shape)
        else:
            cost_X = torch.ones_like(mean)

        gi_value = GittinsIndexFunction.apply(X, mean, sigma, self.lmbda, self.maximize, self.bound, self.eps, cost_X)

        # If maximizing, return the GI value as is; if minimizing, return its negative
        return gi_value if self.maximize else -gi_value

class ExpectedImprovementWithCost(AnalyticAcquisitionFunction):
    """
    This is the acquisition function EI(x) / c(x) ^ alpha, where alpha is a decay
    factor that reduces or increases the emphasis of the cost function c(x).
    """

    def __init__(self, model, best_f, maximize, cost, alpha=1):
        super().__init__(model=model)
        self.model = model
        self.cost = cost
        self.ei = ExpectedImprovement(model=model, best_f=best_f, maximize=maximize)
        self.alpha = alpha

    def forward(self, X):
        return self.ei(X) / torch.pow(self.cost(X).view(self.ei(X).shape), self.alpha)