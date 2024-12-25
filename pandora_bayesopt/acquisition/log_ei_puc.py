#!/usr/bin/env python3

import math
from typing import Callable, Optional, Union
import torch
from torch import Tensor
from botorch.acquisition import AnalyticAcquisitionFunction, LogExpectedImprovement
from botorch.models.model import Model
from botorch.utils.transforms import t_batch_mode_transform
from botorch.utils.constants import get_constants_like
from botorch.utils.probability.utils import (
    log_phi,
    ndtr as Phi,
    phi,
)
from botorch.utils.safe_math import log1mexp


class LogExpectedImprovementWithCost(AnalyticAcquisitionFunction):
    """
    Computes the logarithm of the classic Expected Improvement With Cost acquisition function in a numerically robust manner:

    `LogEIC(x; alpha) = LogEI(x) - alpha * log(c(x)),' 
    
    where LogEI(x) = log(E(max(f(x) - best_f, 0))), alpha is a cost exponent (decay factor) that reduces or increases the emphasis of the cost function c(x).
    """
    def __init__(
        self,
        model: Model,
        best_f: Union[float, Tensor],
        maximize: bool = True,
        cost: Optional[Callable] = None,
        cost_exponent: Union[float, Tensor] = 1.0,
        unknown_cost: bool = False
    ):
        r"""Logarithm of Single-outcome/Two-outcome ExpectedImprovementWithCost (analytic).
        Args:
            model: A fitted single-outcome model or a fitted two-outcome model, 
                where the first output corresponds to the objective 
                and the second one to the log-cost.
            best_f: Either a scalar or a `b`-dim Tensor (batch mode) representing
                the best function value observed so far (assumed noiseless).
            maximize: If True, consider the problem a maximization problem.
        """
        # use AcquisitionFunction constructor to avoid check for objective
        super(AnalyticAcquisitionFunction, self).__init__(model=model)
        self.maximize = maximize
        self.cost = cost
        self.unknown_cost = unknown_cost
        self.register_buffer("best_f", torch.as_tensor(best_f))
        self.register_buffer("cost_exponent", torch.as_tensor(cost_exponent))
    
    @t_batch_mode_transform(expected_q=1, assert_output_shape=False)
    def forward(self, X: Tensor) -> Tensor:
        if self.unknown_cost:
            # Handling the unknown cost scenario
            posterior = self.model.posterior(X)
            means = posterior.mean  # (b) x 2
            vars = posterior.variance.clamp_min(1e-6)  # (b) x 2
            stds = vars.sqrt()

            mean_obj = means[..., 0]
            std_obj = stds[..., 0]
            u = (mean_obj - self.best_f) / std_obj
            if not self.maximize:
                u = -u
            log_ei = _log_ei_helper(u) + std_obj.log()  # (b) x 1
            log_mgf = -(self.cost_exponent * means[..., 1]) + 0.5 * (torch.square(self.cost_exponent) * vars[..., 1])
            log_ei_puc = log_ei + log_mgf  # (b) x 1
            return log_ei_puc.squeeze(dim=-1)
        else:
            # Handling the known cost scenario
            LogEI = LogExpectedImprovement(model=self.model, best_f=self.best_f, maximize=self.maximize)
            log_eic = LogEI(X) - self.cost_exponent * self.cost(X).view(LogEI(X).shape).log()
            return log_eic
        

# the following two numbers are needed for _log_ei_helper
_neg_inv_sqrt2 = -(2**-0.5)
_log_sqrt_pi_div_2 = math.log(math.pi / 2) / 2

# --------------- Helper functions ---------------

def _ei_helper(u: Tensor) -> Tensor:
    """Computes phi(u) + u * Phi(u), where phi and Phi are the standard normal
    pdf and cdf, respectively. This is used to compute Expected Improvement.
    """
    return phi(u) + u * Phi(u)


def _log_ei_helper(u: Tensor) -> Tensor:
    """Accurately computes log(phi(u) + u * Phi(u)) in a differentiable manner for u in
    [-10^100, 10^100] in double precision, and [-10^20, 10^20] in single precision.
    Beyond these intervals, a basic squaring of u can lead to floating point overflow.
    In contrast, the implementation in _ei_helper only yields usable gradients down to
    u ~ -10. As a consequence, _log_ei_helper improves the range of inputs for which a
    backward pass yields usable gradients by many orders of magnitude.
    """
    if not (u.dtype == torch.float32 or u.dtype == torch.float64):
        raise TypeError(
            f"LogExpectedImprovement only supports torch.float32 and torch.float64 "
            f"dtypes, but received {u.dtype = }."
        )
    # The function has two branching decisions. The first is u < bound, and in this
    # case, just taking the logarithm of the naive _ei_helper implementation works.
    bound = -1
    u_upper = u.masked_fill(u < bound, bound)  # mask u to avoid NaNs in gradients
    log_ei_upper = _ei_helper(u_upper).log()

    # When u <= bound, we need to be more careful and rearrange the EI formula as
    # log(phi(u)) + log(1 - exp(w)), where w = log(abs(u) * Phi(u) / phi(u)).
    # To this end, a second branch is necessary, depending on whether or not u is
    # smaller than approximately the negative inverse square root of the machine
    # precision. Below this point, numerical issues in computing log(1 - exp(w)) occur
    # as w approaches zero from below, even though the relative contribution to log_ei
    # vanishes in machine precision at that point.
    neg_inv_sqrt_eps = -1e6 if u.dtype == torch.float64 else -1e3

    # mask u for to avoid NaNs in gradients in first and second branch
    u_lower = u.masked_fill(u > bound, bound)
    u_eps = u_lower.masked_fill(u < neg_inv_sqrt_eps, neg_inv_sqrt_eps)
    # compute the logarithm of abs(u) * Phi(u) / phi(u) for moderately large negative u
    w = _log_abs_u_Phi_div_phi(u_eps)

    # 1) Now, we use a special implementation of log(1 - exp(w)) for moderately
    # large negative numbers, and
    # 2) capture the leading order of log(1 - exp(w)) for very large negative numbers.
    # The second special case is technically only required for single precision numbers
    # but does "the right thing" regardless.
    log_ei_lower = log_phi(u) + (
        torch.where(
            u > neg_inv_sqrt_eps,
            log1mexp(w),
            # The contribution of the next term relative to log_phi vanishes when
            # w_lower << eps but captures the leading order of the log1mexp term.
            -2 * u_lower.abs().log(),
        )
    )
    return torch.where(u > bound, log_ei_upper, log_ei_lower)


def _log_abs_u_Phi_div_phi(u: Tensor) -> Tensor:
    """Computes log(abs(u) * Phi(u) / phi(u)), where phi and Phi are the normal pdf
    and cdf, respectively. The function is valid for u < 0.

    NOTE: In single precision arithmetic, the function becomes numerically unstable for
    u < -1e3. For this reason, a second branch in _log_ei_helper is necessary to handle
    this regime, where this function approaches -abs(u)^-2 asymptotically.

    The implementation is based on the following implementation of the logarithm of
    the scaled complementary error function (i.e. erfcx). Since we only require the
    positive branch for _log_ei_helper, _log_abs_u_Phi_div_phi does not have a branch,
    but is only valid for u < 0 (so that _neg_inv_sqrt2 * u > 0).

        def logerfcx(x: Tensor) -> Tensor:
            return torch.where(
                x < 0,
                torch.erfc(x.masked_fill(x > 0, 0)).log() + x**2,
                torch.special.erfcx(x.masked_fill(x < 0, 0)).log(),
        )

    Further, it is important for numerical accuracy to move u.abs() into the
    logarithm, rather than adding u.abs().log() to logerfcx. This is the reason
    for the rather complex name of this function: _log_abs_u_Phi_div_phi.
    """
    # get_constants_like allocates tensors with the appropriate dtype and device and
    # caches the result, which improves efficiency.
    a, b = get_constants_like(values=(_neg_inv_sqrt2, _log_sqrt_pi_div_2), ref=u)
    return torch.log(torch.special.erfcx(a * u) * u.abs()) + b