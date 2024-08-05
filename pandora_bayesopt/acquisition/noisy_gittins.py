#!/usr/bin/env python3

from .gittins import GittinsIndex
from typing import Callable, Optional, Union
import torch
from torch import Tensor
from torch.autograd import (Function, grad)
from botorch.models.gpytorch import GPyTorchModel
from botorch.models import SingleTaskGP
from botorch.acquisition.analytic import (_scaled_improvement, _ei_helper, nullcontext, _get_noiseless_fantasy_model)
from botorch.acquisition.objective import PosteriorTransform
from botorch.utils.transforms import t_batch_mode_transform
from botorch.utils.probability.utils import (
    ndtr as Phi,
    phi,
)
from botorch.exceptions import UnsupportedError
from gpytorch.likelihoods.gaussian_likelihood import FixedNoiseGaussianLikelihood


# Set default tensor type to float64
torch.set_default_dtype(torch.float64)

class NoisyGittinsIndex(GittinsIndex):
    r"""Single-outcome/Two-outcome Noisy Gittins Index (via fantasies).

    Computes Noisy Gittins index by replacing the Expected Improvement with the average Expected
    Improvement value of a number of fantasy models in the standard Gittins index computation.
    Assumes that the posterior distribution of the model is Gaussian.
    The model can be either single-outcome or two-outcome.

    `NGI(x) = argmin_g |E(max(y - g, 0))-lmbda * c(x)|, (y, Y_baseline) ~ f((x, X_baseline))`

    where `X_baseline` are previously observed points.

    Example:
        Uniform-cost:
        >>> model = SingleTaskGP(train_X, train_Y, train_Yvar=train_Yvar)
        >>> NGI = NoisyGittinsIndex(model, train_X, lmbda=0.0001)
        >>> ngi = NGI(test_X)
        
        Varing-cost:
        >>> def cost_function(x):
        >>>     return 1+20*x.mean(dim=-1))
        >>> model = SingleTaskGP(train_X, train_Y, train_Yvar=train_Yvar)
        >>> NGI = NoisyGittinsIndex(model, train_X, lmbda=0.0001, cost=cost_function)
        >>> ngi = NGI(test_X)

        Unknown-cost:
        >>> model = SingleTaskGP(train_X, train_Y, train_Yvar=train_Yvar)
        >>> NGI = NoisyGittinsIndex(model, train_X, lmbda=0.0001, cost=cost_function, unknown_cost=True)
        >>> ngi = NGI(test_X)
    """

    def __init__(
        self,
        model: GPyTorchModel,
        X_observed: Tensor,
        lmbda: float,
        num_fantasies: int = 20,
        posterior_transform: Optional[PosteriorTransform] = None,
        maximize: bool = True,
        bound: torch.Tensor = torch.tensor([[-1.0], [1.0]], dtype=torch.float64),
        eps: float = 1e-6,
        cost: Optional[Callable] = None,
        unknown_cost: bool = False,
        bisection_early_stopping: bool = False
    ) -> None:
        r"""Single-outcome/Two-outcome Noisy Gittins Index (via fantasies).

        Args:
            model: A fitted single-outcome model. Only `SingleTaskGP` models with
                known observation noise are currently supported.
            X_observed: A `n x d` Tensor of observed points that are likely to
                be the best observed points so far.
            lmbda: A scalar representing the cost-per-sample or the scaling factor of the cost function.
            num_fantasies: The number of fantasies to generate. The higher this
                number the more accurate the model (at the expense of model
                complexity and performance).
            posterior_transform: A PosteriorTransform. If using a multi-output model,
                a PosteriorTransform that transforms the multi-output posterior into a
                single-output posterior is required.
            maximize: If True, consider the problem a maximization problem.
            cost: A callable cost function. If None, consider the problem a uniform-cost problem.
            unknown_cost: If True, consider the problem an unknown-cost problem.
            bound: A `2 x d` tensor of lower and upper bound for each column of `X`.
        """
        _check_noisy_ei_model(model=model)
        legacy_ei_numerics_warning(legacy_name=type(self).__name__)
        # Sample fantasies.
        from botorch.sampling.normal import SobolQMCNormalSampler

        # Drop gradients from model.posterior if X_observed does not require gradients
        # as otherwise, gradients of the GP's kernel's hyper-parameters are tracked
        # through the rsample_from_base_sample method of GPyTorchPosterior. These
        # gradients are usually only required w.r.t. the marginal likelihood.
        with nullcontext() if X_observed.requires_grad else torch.no_grad():
            posterior = model.posterior(X=X_observed)
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([num_fantasies]))
        Y_fantasized = sampler(posterior).squeeze(-1)
        batch_X_observed = X_observed.expand(num_fantasies, *X_observed.shape)
        # The fantasy model will operate in batch mode
        fantasy_model = _get_noiseless_fantasy_model(
            model=model, batch_X_observed=batch_X_observed, Y_fantasized=Y_fantasized
        )
        super().__init__(
            model=fantasy_model, 
            lmbda=lmbda, 
            maximize=maximize, 
            posterior_transform=posterior_transform, 
            bound=bound, 
            eps=eps, 
            cost=cost, 
            unknown_cost=unknown_cost, 
            bisection_early_stopping=bisection_early_stopping
        )
      
        
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
        # add batch dimension for broadcasting to fantasy models
        X_unsqueezed = X.unsqueeze(-3)

        if self.unknown_cost:
            # Handling the unknown cost scenario
            posterior = self.model.posterior(X_unsqueezed)
            means = posterior.mean  # (b) x 2
            vars = posterior.variance.clamp_min(1e-6)  # (b) x 2
            stds = vars.sqrt()

            mean_obj = means[..., 0].squeeze(dim=-1)
            std_obj = stds[..., 0].squeeze(dim=-1)

            mgf = (torch.exp(means[..., 1]) + 0.5 * vars[..., 1]).squeeze(dim=-1)

            gi_value = BatchGittinsIndexFunction.apply(X_unsqueezed, mean_obj, std_obj, self.lmbda, self.maximize, self.bound, self.eps, mgf, self.bisection_early_stopping)

        else:
            # Handling the known cost scenario
            mean, sigma = self._mean_and_sigma(X_unsqueezed)

            if callable(self.cost):
                cost_X = self.cost(X_unsqueezed).view(mean.size(0))
            else:
                cost_X = torch.ones(mean.size(0))

            gi_value = BatchGittinsIndexFunction.apply(X_unsqueezed, mean, sigma, self.lmbda, self.maximize, self.bound, self.eps, cost_X, self.bisection_early_stopping)

        # If maximizing, return the GI value as is; if minimizing, return its negative
        return gi_value if self.maximize else -gi_value
    

class BatchGittinsIndexFunction(Function):
    @staticmethod
        
    def forward(
        ctx, 
        X: Tensor, 
        mean: Tensor, 
        sigma: Tensor, 
        lmbda: float, 
        maximize: bool, 
        bound: Tensor, 
        eps: float, 
        cost_X: Union[float, Tensor],
        bisection_early_stopping: bool = False,
    ):

        def cost_adjusted_expected_improvement(best_f):
            u = _scaled_improvement(mean, sigma, best_f.unsqueeze(1).repeat(1, mean.size(1)), maximize)
            return (sigma * _ei_helper(u)).mean(dim=-1) - lmbda * cost_X

        size = X.size(0)
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
        for i in range(100):
            sgn_m = torch.sign(cost_adjusted_expected_improvement(best_f=m))
            if maximize:
                l = torch.where(sgn_m >= 0, m, l)
                h = torch.where(sgn_m <= 0, m, h)
            else:
                l = torch.where(sgn_m <= 0, m, l)
                h = torch.where(sgn_m >= 0, m, h)
            m = (h + l) / 2
            # if bisection_early_stopping and torch.max(torch.abs(cost_adjusted_expected_improvement(best_f=m))) <= eps:
            #     break

        # Save u for backward computation
        u = _scaled_improvement(mean, sigma, m.unsqueeze(1).repeat(1, mean.size(1)), maximize)
        
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
        dmean_dX = grad(outputs=mean, inputs=X, grad_outputs=torch.ones_like(mean), retain_graph=True, allow_unused=True)[0]

        # Gradient of the std function with respect to x
        dsigma_dX = grad(outputs=sigma, inputs=X, grad_outputs=torch.ones_like(sigma), retain_graph=True, allow_unused=True)[0]

        if cost_X.requires_grad:
            # Compute gradient only if cost_X is not a scalar
            dcost_dX = grad(outputs=cost_X, inputs=X, grad_outputs=torch.ones_like(cost_X), retain_graph=True, allow_unused=True)[0]
        else:
            # If cost_X does not require grad, set its gradient to zero
            dcost_dX = torch.zeros_like(X)

        # Check if gradients are None and handle accordingly
        if dmean_dX is None or dsigma_dX is None or dcost_dX is None:
            raise RuntimeError("Gradients could not be computed for one or more components.")
        
        # Compute the gradient of the Gittins acquisition function
        if maximize:
            grad_X = grad_output.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * (dmean_dX + (phi(u).unsqueeze(-1).unsqueeze(-1) * dsigma_dX - lmbda * dcost_dX) / Phi(u).unsqueeze(-1).unsqueeze(-1))
        else:
            grad_X = grad_output.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * (dmean_dX - (phi(u).unsqueeze(-1).unsqueeze(-1) * dsigma_dX - lmbda * dcost_dX) / Phi(u).unsqueeze(-1).unsqueeze(-1))

        return grad_X, None, None, None, None, None, None, None, None
    

def _check_noisy_ei_model(model: GPyTorchModel) -> None:
    message = (
        "Only single-output `SingleTaskGP` models with known observation noise "
        "are currently supported for fantasy-based NEI & LogNEI."
    )
    if not isinstance(model, SingleTaskGP):
        raise UnsupportedError(f"{message} Model is not a `SingleTaskGP`.")
    if not isinstance(model.likelihood, FixedNoiseGaussianLikelihood):
        raise UnsupportedError(
            f"{message} Model likelihood is not a `FixedNoiseGaussianLikelihood`."
        )
    if model.num_outputs != 1:
        raise UnsupportedError(f"{message} Model has {model.num_outputs} outputs.")
    

def legacy_ei_numerics_warning(legacy_name: str) -> None:
    """Raises a warning for legacy EI acquisition functions that are known to have
    numerical issues and should be replaced with the LogEI version for virtually all
    use-cases except for explicit benchmarking of the numerical issues of legacy EI.

    Args:
        legacy_name: The name of the legacy EI acquisition function.
        logei_name: The name of the associated LogEI acquisition function.
    """
    legacy_to_logei = {
        "ExpectedImprovement": "LogExpectedImprovement",
        "ConstrainedExpectedImprovement": "LogConstrainedExpectedImprovement",
        "NoisyExpectedImprovement": "LogNoisyExpectedImprovement",
        "qExpectedImprovement": "qLogExpectedImprovement",
        "qNoisyExpectedImprovement": "qLogNoisyExpectedImprovement",
        "qExpectedHypervolumeImprovement": "qLogExpectedHypervolumeImprovement",
        "qNoisyExpectedHypervolumeImprovement": (
            "qLogNoisyExpectedHypervolumeImprovement"
        ),
    }
    # Only raise the warning if the legacy name is in the mapping. It can fail to be in
    # the mapping if the legacy acquisition function derives from a legacy EI class,
    # e.g. MOMF, which derives from qEHVI, but there is not corresponding LogMOMF yet.
    if legacy_name in legacy_to_logei:
        logei_name = legacy_to_logei[legacy_name]
        msg = (
            f"{legacy_name} has known numerical issues that lead to suboptimal "
            "optimization performance. It is strongly recommended to simply replace"
            f"\n\n\t {legacy_name} \t --> \t {logei_name} \n\n"
            "instead, which fixes the issues and has the same "
            "API. See https://arxiv.org/abs/2310.20708 for details."
        )
        warnings.warn(msg, NumericsWarning, stacklevel=2)