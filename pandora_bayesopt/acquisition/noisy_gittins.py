#!/usr/bin/env python3

from typing import Callable, Optional
import torch
from torch import Tensor
from botorch.models.gpytorch import GPyTorchModel
from botorch.acquisition.analytic import (_check_noisy_ei_model, nullcontext, legacy_ei_numerics_warning, _get_noiseless_fantasy_model)
from botorch.acquisition.objective import PosteriorTransform
from botorch.utils.transforms import t_batch_mode_transform
from gittins import GittinsIndexFunction, GittinsIndex


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
        r"""Single-outcome Noisy Expected Improvement (via fantasies).

        Args:
            model: A fitted single-outcome model. Only `SingleTaskGP` models with
                known observation noise are currently supported.
            X_observed: A `n x d` Tensor of observed points that are likely to
                be the best observed points so far.
            num_fantasies: The number of fantasies to generate. The higher this
                number the more accurate the model (at the expense of model
                complexity and performance).
            maximize: If True, consider the problem a maximization problem.
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
        super().__init__(model=fantasy_model, maximize=maximize)
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

            gi_value = GittinsIndexFunction.apply(X, mean_obj, std_obj, self.lmbda, self.maximize, self.bound, self.eps, mgf, self.bisection_early_stopping)

        else:
            # Handling the known cost scenario
            mean, sigma = self._mean_and_sigma(X_unsqueezed)

            if callable(self.cost):
                cost_X = self.cost(X_unsqueezed).view(mean.shape)
            else:
                cost_X = torch.ones_like(mean)

            gi_value = GittinsIndexFunction.apply(X_unsqueezed, mean, sigma, self.lmbda, self.maximize, self.bound, self.eps, cost_X, self.bisection_early_stopping)

        # If maximizing, return the GI value as is; if minimizing, return its negative
        return gi_value if self.maximize else -gi_value