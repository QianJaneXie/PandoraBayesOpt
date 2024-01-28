#!/usr/bin/env python3

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import torch
from botorch.acquisition.multi_step_lookahead import qMultiStepLookahead
from botorch.acquisition.objective import LinearMCObjective, ScalarizedObjective
from botorch.models.model import Model
from botorch.sampling.samplers import MCSampler, SobolQMCNormalSampler
from budgeted_bo.acquisition_functions.budgeted_ei import (
    BudgetedExpectedImprovement,
    qBudgetedExpectedImprovement,
)
from budgeted_bo.samplers import PosteriorMeanSampler
from torch import Tensor
from torch.nn import Module


class BudgetedMultiStepExpectedImprovement(qMultiStepLookahead):
    r"""Budget-Constrained Multi-Step Look-Ahead Expected Improvement (one-shot optimization)."""

    def __init__(
        self,
        model: Model,
        budget_plus_cumulative_cost: Union[float, Tensor],
        batch_size: int,
        lookahead_batch_sizes: List[int],
        num_fantasies: Optional[List[int]] = None,
        samplers: Optional[List[MCSampler]] = None,
        X_pending: Optional[Tensor] = None,
        collapse_fantasy_base_samples: bool = True,
    ) -> None:
        r"""Budgeted Multi-Step Expected Improvement.

        Args:
            model: A fitted two-output model, where the first output corresponds to the
                objective, and the second one to the log-cost.
            budget: A value determining the budget constraint.
            batch_size: Batch size of the current step.
            lookahead_batch_sizes: A list `[q_1, ..., q_k]` containing the batch sizes for the
            `k` look-ahead steps.
            num_fantasies: A list `[f_1, ..., f_k]` containing the number of fantasy
                for the `k` look-ahead steps.
            samplers: A list of MCSampler objects to be used for sampling fantasies in
                each stage.
            X_pending: A `m x d`-dim Tensor of `m` design points that have points that
                have been submitted for function evaluation but have not yet been
                evaluated. Concatenated into `X` upon forward call. Copied and set to
                have no gradient.
            collapse_fantasy_base_samples: If True, collapse_batch_dims of the Samplers
                will be applied on fantasy batch dimensions as well, meaning that base
                samples are the same in all subtrees starting from the same level.
        """
        self.budget_plus_cumulative_cost = budget_plus_cumulative_cost
        self.batch_size = batch_size
        batch_sizes = [batch_size] + lookahead_batch_sizes

        # TODO: This objective is never really used.
        weights = torch.zeros(model.num_outputs, dtype=torch.double)
        weights[0] = 1.0

        use_mc_val_funcs = any(bs != 1 for bs in batch_sizes)

        if use_mc_val_funcs:
            objective = LinearMCObjective(weights=weights)

            valfunc_cls = [qBudgetedExpectedImprovement for _ in batch_sizes]

            inner_mc_samples = [128 for bs in batch_sizes]
        else:
            objective = ScalarizedObjective(weights=weights)

            valfunc_cls = [BudgetedExpectedImprovement for _ in batch_sizes]

            inner_mc_samples = None

        valfunc_argfacs = [
            budgeted_ei_argfac(
                budget_plus_cumulative_cost=self.budget_plus_cumulative_cost
            )
            for _ in batch_sizes
        ]

        # Set samplers
        if samplers is None:
            # The batch_range is not set here and left to sampler default of (0, -2),
            # meaning that collapse_batch_dims will be applied on fantasy batch dimensions.
            # If collapse_fantasy_base_samples is False, the batch_range is updated during
            # the forward call.
            samplers: List[MCSampler] = [
                PosteriorMeanSampler(collapse_batch_dims=True)
                if nf == 1
                else SobolQMCNormalSampler(
                    num_samples=nf, resample=False, collapse_batch_dims=True
                )
                for nf in num_fantasies
            ]

        super().__init__(
            model=model,
            batch_sizes=lookahead_batch_sizes,
            samplers=samplers,
            valfunc_cls=valfunc_cls,
            valfunc_argfacs=valfunc_argfacs,
            objective=objective,
            inner_mc_samples=inner_mc_samples,
            X_pending=X_pending,
            collapse_fantasy_base_samples=collapse_fantasy_base_samples,
        )


class budgeted_ei_argfac(Module):
    r"""Extract the best observed value and reamaining budget from the model."""

    def __init__(self, budget_plus_cumulative_cost: Union[float, Tensor]) -> None:
        super().__init__()
        self.budget_plus_cumulative_cost = budget_plus_cumulative_cost

    def forward(self, model: Model, X: Tensor) -> Dict[str, Any]:
        y = torch.transpose(model.train_targets, -2, -1)
        y_original_scale = model.outcome_transform.untransform(y)[0]
        obj_vals = y_original_scale[..., 0]
        log_costs = y_original_scale[..., 1]
        costs = torch.exp(log_costs)
        current_budget = self.budget_plus_cumulative_cost - costs.sum(
            dim=-1, keepdim=True
        )

        params = {
            "best_f": obj_vals.max(dim=-1, keepdim=True).values,
            "budget": current_budget,
        }
        return params