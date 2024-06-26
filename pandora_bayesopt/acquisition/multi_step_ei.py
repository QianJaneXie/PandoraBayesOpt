#!/usr/bin/env python3

# Original code from Raul Astudillo https://github.com/RaulAstudillo06/BudgetedBO
# Copyright (c) 2021 Raul Astudillo
# Copyright (c) Meta Platforms, Inc. and affiliates.

from __future__ import annotations

from typing import Any, Dict, List, Optional

from botorch.acquisition.analytic import ExpectedImprovement
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.acquisition.multi_step_lookahead import qMultiStepLookahead
from botorch.models.model import Model
from botorch.sampling.base import MCSampler
from botorch.sampling.normal import SobolQMCNormalSampler
from torch import Tensor, Size

from ..sampling.posterior_mean_sampler import PosteriorMeanSampler


class MultiStepLookaheadEI(qMultiStepLookahead):
    r"""Multi-Step Look-Ahead Expected Improvement (one-shot optimization)."""

    def __init__(
        self,
        model: Model,
        batch_size: int,
        lookahead_batch_sizes: List[int],
        num_fantasies: Optional[List[int]] = None,
        samplers: Optional[List[MCSampler]] = None,
        X_pending: Optional[Tensor] = None,
        collapse_fantasy_base_samples: bool = True,
    ) -> None:
        r"""Multi-Step Look-Ahead Expected Improvement.

        Args:
            model: A single-output Model of appropriate batch size.
            batch_size: Batch size for the current step.
            lookahead_batch_sizes: A list `[q_1, ..., q_k]` containing the batch sizes for the
                `k` look-ahead steps.
            num_fantasies: A list `[f_1, ..., f_k]` containing the number of fantasy
                points to use for the `k` look-ahead steps.
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
        self.batch_size = batch_size
        batch_sizes = [batch_size] + lookahead_batch_sizes

        if any(bs != 1 for bs in batch_sizes):
            valfunc_cls = [qExpectedImprovement for _ in batch_sizes]
            inner_mc_samples = [512 for bs in batch_sizes]
        else:
            valfunc_cls = [ExpectedImprovement for _ in batch_sizes]
            inner_mc_samples = None

        valfunc_argfacs = [multi_step_ei_argfac for _ in batch_sizes]

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


def multi_step_ei_argfac(model: Model, X: Tensor) -> Dict[str, Any]:
    y = model.train_targets
    if hasattr(model, 'outcome_transform'):
        y_original_scale = model.outcome_transform.untransform(y)[0]
    else:
        y_original_scale = y
    obj_vals = y_original_scale
    params = {
        "best_f": obj_vals.max(dim=-1).values,
    }
    return params