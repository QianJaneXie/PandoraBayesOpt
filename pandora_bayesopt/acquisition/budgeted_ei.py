#!/usr/bin/env python3

from __future__ import annotations

from typing import Callable, Optional, Union

import torch
from botorch.acquisition.analytic import AnalyticAcquisitionFunction, _ei_helper, _scaled_improvement
from botorch.acquisition.objective import PosteriorTransform
from botorch.models.model import Model
from botorch.utils.objective import soft_eval_constraint
from botorch.utils.transforms import  t_batch_mode_transform
from torch import Tensor


class BudgetedExpectedImprovement(AnalyticAcquisitionFunction):
    r"""
    """

    def __init__(
        self,
        model: Model,
        cost_function: Callable,
        best_f: Union[float, Tensor],
        budget: Union[float, Tensor],
        posterior_transform: Optional[PosteriorTransform] = None,
        maximize: bool = True,
    ):
        r"""Single-outcome Expected Improvement (analytic).

        Args:
            model: A fitted single-outcome model.
            cost_function: .
            best_f: Either a scalar or a `b`-dim Tensor (batch mode) representing
                the best function value observed so far (assumed noiseless).
            budget: Either a scalar or a `b`-dim Tensor (batch mode) representing
                the budget constraint.
            posterior_transform: A PosteriorTransform. If using a multi-output model,
                a PosteriorTransform that transforms the multi-output posterior into a
                single-output posterior is required.
            maximize: If True, consider the problem a maximization problem.
        """
        super(AnalyticAcquisitionFunction, self).__init__(model=model)
        self.cost_function = cost_function
        self.posterior_transform = None
        self.maximize = maximize
        self.register_buffer("best_f", torch.as_tensor(best_f))
        self.register_buffer("budget", torch.as_tensor(budget))

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
        mean, sigma = self._mean_and_sigma(X)
        u = _scaled_improvement(mean, sigma, self.best_f, self.maximize)
        sum_costs = self.cost_function(X).sum(dim=-1, keepdim=False)
        smooth_feas_ind = soft_eval_constraint(lhs=sum_costs - self.budget)
        return sigma * _ei_helper(u) * smooth_feas_ind