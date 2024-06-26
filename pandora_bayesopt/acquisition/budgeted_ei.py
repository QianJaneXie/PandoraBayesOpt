#!/usr/bin/env python3

# Original code from Raul Astudillo https://github.com/RaulAstudillo06/BudgetedBO
# Copyright (c) 2021 Raul Astudillo
# Modifications made by Raul Asutidllo and Qian Xie, 2024 to handle known costs

from __future__ import annotations

from typing import Callable, Optional, Union

import torch
from botorch.acquisition.analytic import AnalyticAcquisitionFunction, _ei_helper, _scaled_improvement
from botorch.acquisition.objective import PosteriorTransform
from botorch.models.model import Model
from botorch.utils.objective import soft_eval_constraint
from botorch.utils.transforms import  t_batch_mode_transform
from torch import Tensor
from torch.distributions import Normal


class BudgetedExpectedImprovement(AnalyticAcquisitionFunction):
    r"""
    """

    def __init__(
        self,
        model: Model,
        unknown_cost: bool,
        best_f: Union[float, Tensor],
        budget: Union[float, Tensor],
        posterior_transform: Optional[PosteriorTransform] = None,
        maximize: bool = True,
        cost_function: Optional[Callable] = None
    ):
        r"""Single-outcome Expected Improvement (analytic).

        Args:
            model: A fitted single-outcome model.
            best_f: Either a scalar or a `b`-dim Tensor (batch mode) representing
                the best function value observed so far (assumed noiseless).
            budget: Either a scalar or a `b`-dim Tensor (batch mode) representing
                the budget constraint.
            posterior_transform: A PosteriorTransform. If using a multi-output model,
                a PosteriorTransform that transforms the multi-output posterior into a
                single-output posterior is required.
            maximize: If True, consider the problem a maximization problem.
            cost_function: None if the costs are uniform or unknown.
            unkown_cost: True if the costs are unknown. 
        """
        super(AnalyticAcquisitionFunction, self).__init__(model=model)
        self.posterior_transform = None
        self.maximize = maximize
        self.register_buffer("best_f", torch.as_tensor(best_f))
        self.register_buffer("budget", torch.as_tensor(budget))
        self.cost_function = cost_function
        self.unknown_cost = unknown_cost

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate Expected Improvement on the candidate set X.

        Args:
            X: A `(b1 x ... bk) x 1 x d`-dim batched tensor of `d`-dim design points.
                Expected Improvement is computed for each point individually,
                i.e., what is considered are the marginal posteriors, not the
                joint.

        Returns:
            A `(b1 x ... bk)`-dim tensor of Expected Improvement values at the
            given design points `X`.
        """
        if self.unknown_cost:
            # Handling the unknown cost scenario
            posterior = self.model.posterior(X)
            means = posterior.mean  # (b) x 2
            sigmas = posterior.variance.sqrt().clamp_min(1e-6)  # (b) x 2

            # (b) x 1
            mean_obj = means[..., 0]
            sigma_obj = sigmas[..., 0]
            u = (mean_obj - self.best_f) / sigma_obj

            if not self.maximize:
                u = -u
            standard_normal = Normal(
                torch.zeros(1, device=u.device, dtype=u.dtype),
                torch.ones(1, device=u.device, dtype=u.dtype),
            )
            pdf_u = torch.exp(standard_normal.log_prob(u))
            cdf_u = standard_normal.cdf(u)
            ei = sigma_obj * (pdf_u + u * cdf_u)  # (b) x 1
            # (b) x 1
            prob_feas = self._compute_prob_feas(means=means[..., 1], sigmas=sigmas[..., 1])
            bc_ei = ei.mul(prob_feas)  # (b) x 1
            return bc_ei.squeeze(dim=-1)
        else:
            # Handling the known cost scenario
            mean, sigma = self._mean_and_sigma(X)
            u = _scaled_improvement(mean, sigma, self.best_f, self.maximize)
            sum_costs = self.cost_function(X).sum(dim=-1, keepdim=False)
            smooth_feas_ind = soft_eval_constraint(lhs=sum_costs - self.budget)
            return sigma * _ei_helper(u) * smooth_feas_ind
        
    def _compute_prob_feas(self, means: Tensor, sigmas: Tensor) -> Tensor:
        r"""Compute feasibility probability for each batch of X.
        Args:
            X: A `(b) x 1 x d`-dim Tensor of `(b)` t-batches of `d`-dim design
                points each.
            means: A `(b) x 1`-dim Tensor of means.
            sigmas: A `(b) x 1`-dim Tensor of standard deviations.
        Returns:
            A `(b) x 1`-dim tensor of feasibility probabilities.
        """
        standard_normal = Normal(
            torch.zeros(1, device=means.device, dtype=means.dtype),
            torch.ones(1, device=means.device, dtype=means.dtype),
            validate_args=True,
        )
        prob_feas = standard_normal.cdf(
            (torch.log(self.budget.clamp_min(1e-6)) - means) / sigmas
        )
        prob_feas = torch.where(
            self.budget > 1e-6,
            prob_feas,
            torch.zeros(1, device=means.device, dtype=means.dtype),
        )
        return prob_feas
