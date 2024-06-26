#!/usr/bin/env python3

# Original code from Raul Astudillo https://github.com/RaulAstudillo06/BudgetedBO
# Copyright (c) 2021 Raul Astudillo
# Modifications made by Raul Asutidllo and Qian Xie, 2024 to handle known costs

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Union

from botorch.acquisition.multi_step_lookahead import qMultiStepLookahead
from botorch.models.model import Model
from botorch.sampling.base import MCSampler
from botorch.sampling.normal import SobolQMCNormalSampler
from torch import Tensor, Size
from torch.nn import Module

import torch

from .budgeted_ei import BudgetedExpectedImprovement
from ..sampling.posterior_mean_sampler import PosteriorMeanSampler


class BudgetedMultiStepLookaheadEI(qMultiStepLookahead):
    r"""Budget-Constrained Multi-Step Look-Ahead Expected Improvement (one-shot optimization)."""

    def __init__(
        self,
        model: Model,
        unknown_cost: bool,
        budget_plus_cumulative_cost: Union[float, Tensor],
        batch_size: int,
        lookahead_batch_sizes: List[int],
        num_fantasies: Optional[List[int]] = None,
        samplers: Optional[List[MCSampler]] = None,
        X_pending: Optional[Tensor] = None,
        collapse_fantasy_base_samples: bool = True,
        cost_function: Optional[Callable] = None,
    ) -> None:
        r"""Budgeted Multi-Step Expected Improvement.

        Args:
            model: A fitted single-outcome model or a fitted two-output model, where the first output corresponds to the
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
        self.unknown_cost = unknown_cost
        self.cost_function = cost_function
        self.budget_plus_cumulative_cost = budget_plus_cumulative_cost
        self.batch_size = batch_size
        batch_sizes = [batch_size] + lookahead_batch_sizes

        # TODO: This objective is never really used.
        weights = torch.zeros(model.num_outputs, dtype=torch.double)
        weights[0] = 1.0

        use_mc_val_funcs = any(bs != 1 for bs in batch_sizes)

        if use_mc_val_funcs:
            valfunc_cls = [BudgetedExpectedImprovement for _ in batch_sizes]
            inner_mc_samples = [128 for bs in batch_sizes]
        else:
            valfunc_cls = [BudgetedExpectedImprovement for _ in batch_sizes]
            inner_mc_samples = None

        valfunc_argfacs = [
            budgeted_ei_argfac(
                budget_plus_cumulative_cost=self.budget_plus_cumulative_cost,
                cost_function=self.cost_function,
                unknown_cost=self.unknown_cost
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
                PosteriorMeanSampler(sample_shape=Size([nf]))
                if nf == 1
                else SobolQMCNormalSampler(sample_shape=Size([nf]))
                for nf in num_fantasies
            ]

        super().__init__(
            model=model,
            batch_sizes=lookahead_batch_sizes,
            samplers=samplers,
            valfunc_cls=valfunc_cls,
            valfunc_argfacs=valfunc_argfacs,
            inner_mc_samples=inner_mc_samples,
            X_pending=X_pending,
            collapse_fantasy_base_samples=collapse_fantasy_base_samples,
        )


class budgeted_ei_argfac(Module):
    r"""Extract the best observed value and reamaining budget from the model."""

    def __init__(
            self,
            unknown_cost: bool,
            budget_plus_cumulative_cost: Union[float, Tensor], 
            cost_function: Optional[Callable] = None, 
    ) -> None:
        super().__init__()
        self.budget_plus_cumulative_cost = budget_plus_cumulative_cost
        self.cost_function = cost_function
        self.unknown_cost = unknown_cost

    def forward(self, model: Model, X: Tensor) -> Dict[str, Any]:
        if self.unknown_cost:
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
                "cost_function": self.cost_function,
                "unknown_cost": self.unknown_cost
            }
        else:
            x = model.train_inputs[0]
            y = model.train_targets
            if hasattr(model, 'outcome_transform'):
                y_original_scale = model.outcome_transform.untransform(y)[0]
            else:
                y_original_scale = y
            obj_vals = y_original_scale
            costs = self.cost_function(x)
            current_budget = self.budget_plus_cumulative_cost - costs.sum(dim=-1, keepdim=False)
            params = {
                "best_f": obj_vals.max(dim=-1, keepdim=False).values,
                "budget": current_budget,
                "cost_function": self.cost_function,
                "unknown_cost": self.unknown_cost
            }
        return params