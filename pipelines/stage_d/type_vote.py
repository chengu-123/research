"""S3: P1-end deterministic type vote + dual-clone commit.

method.md / pipeline.md S3 fix:

  1. At ``iter == round(total_iters * f_main_g1a_end)`` (default 50%):
     Run ``N_t = cfg.type_vote_n_t_samples`` (4) deterministic SS-DiT
     forwards with fixed seeds, average the resulting ``type_logit``.

  2. Confidence = ``max(p, 1 - p)`` where ``p = sigmoid(mean_logit)``.
     If ``confidence >= cfg.type_vote_confidence_threshold`` (0.7):
        commit ``type_hard = (p > 0.5)`` directly.

  3. Otherwise (low confidence): dual-clone:
        - Clone the entire ``StageDLearnable`` into ``learnable_rev`` and
          ``learnable_pri``; force ``psi_param[6] = -10`` (revolute) and
          ``+10`` (prismatic) respectively.
        - Run each clone for ``f_main_g1b_end - f_main_g1a_end`` * 0.5
          extra iters (so both clones together cost the same wall-clock
          as Main G1b would have used).
        - Pick the clone with lower final ``L_sds + lambda_rgb * L_rgb``.

  4. Commit. P2 (Stage F) renders single-branch from here on.

This file implements step 1, 2, and the manager for step 3. The train
loop calls ``run_type_vote`` once at the trigger iter; if it returns
``vote.committed`` is True, training continues with the committed type.
Otherwise the train loop enters ``run_dual_clone_branches`` which
internally runs the two cloned training streams.
"""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Literal, Optional, Tuple

import torch

from .config import StageDConfig
from .learnable import StageDLearnable


logger = logging.getLogger(__name__)


# =============================================================================
# Deterministic type-logit vote
# =============================================================================

@dataclass
class TypeVoteResult:
    """Outcome of the P1-end vote."""
    committed: bool                         # True if direct commit; False -> dual-clone
    committed_type: Optional[Literal["revolute", "prismatic"]]
    mean_type_logit: float
    mean_prob_prismatic: float              # sigmoid(mean_type_logit)
    confidence: float                       # max(p, 1 - p)
    n_evals: int                            # total deterministic forwards
    per_eval_logits: List[float] = field(default_factory=list)


def _deterministic_type_logit_eval(
    learnable: StageDLearnable,
    forward_single_pass: Callable[[float, int], torch.Tensor],
    t_ss_grid: Tuple[float, ...],
    seed_grid: Tuple[int, ...],
) -> List[float]:
    """Run ``len(t_ss_grid) * len(seed_grid)`` deterministic forwards.

    The caller supplies ``forward_single_pass(t_ss, seed) -> psi_pred`` —
    a closure that does one SS-DiT one-step refiner forward and returns
    the projected joint params (in particular the ``type_logit`` scalar
    which we read out below). Closing over the training loop's state
    keeps this module free of TRELLIS / Wan imports.
    """
    logits: List[float] = []
    for t_ss in t_ss_grid:
        for seed in seed_grid:
            with torch.no_grad():
                psi_pred = forward_single_pass(t_ss, seed)
                logits.append(float(psi_pred.type_logit.detach().item()))
    return logits


def run_type_vote(
    learnable: StageDLearnable,
    cfg: StageDConfig,
    forward_single_pass: Callable[[float, int], torch.Tensor],
) -> TypeVoteResult:
    """P1-end vote. Returns a ``TypeVoteResult``.

    Parameters
    ----------
    learnable : StageDLearnable
    cfg : StageDConfig
    forward_single_pass : (t_ss, seed) -> psi_pred
        Closure returning the projected JointParams for a given t_ss and
        seed. Used to sample multiple deterministic estimates of
        ``type_logit`` and average them.
    """
    # Build the eval grid: cfg.type_vote_n_t_samples x cfg.type_vote_n_seed_samples
    n_t = int(cfg.type_vote_n_t_samples)
    n_seed = int(cfg.type_vote_n_seed_samples)
    if n_t < 1 or n_seed < 1:
        raise ValueError(f"vote requires >= 1 t and seed samples; got {n_t}, {n_seed}")

    # Spread t_ss in main_g1 range.
    t_ss_grid = tuple(
        cfg.t_ss_main_low + (i + 0.5) * (cfg.t_ss_main_high - cfg.t_ss_main_low) / n_t
        for i in range(n_t)
    )
    seed_grid = tuple(1000 + i for i in range(n_seed))

    logits = _deterministic_type_logit_eval(
        learnable, forward_single_pass, t_ss_grid, seed_grid,
    )
    mean_logit = float(sum(logits) / len(logits))
    p = float(1.0 / (1.0 + pow(2.71828182845904523536, -mean_logit)))
    # safer: use torch.sigmoid for numerical stability
    p_t = torch.sigmoid(torch.tensor(mean_logit, dtype=torch.float64))
    p = float(p_t.item())
    confidence = max(p, 1.0 - p)

    threshold = float(cfg.type_vote_confidence_threshold)
    if confidence >= threshold:
        committed_type: Literal["revolute", "prismatic"] = (
            "prismatic" if p > 0.5 else "revolute"
        )
        committed = True
        logger.info(
            "[stage_d type vote] confidence %.3f >= %.3f -> direct commit %s",
            confidence, threshold, committed_type,
        )
    else:
        committed_type = None
        committed = False
        logger.warning(
            "[stage_d type vote] confidence %.3f < %.3f -> trigger dual-clone",
            confidence, threshold,
        )

    return TypeVoteResult(
        committed=committed,
        committed_type=committed_type,
        mean_type_logit=mean_logit,
        mean_prob_prismatic=p,
        confidence=confidence,
        n_evals=len(logits),
        per_eval_logits=logits,
    )


# =============================================================================
# Dual-clone state management
# =============================================================================

@dataclass
class DualCloneState:
    """Tracks the two cloned training streams during S3 dual-clone resolution.

    Both clones inherit the full param state of the original ``learnable``
    at the moment the vote triggered. ``psi_param[6]`` (type_logit) is
    *frozen* in each clone to its forced value (-10 for rev, +10 for pri)
    by zero-ing its gradient before each ``optimizer.step()``.
    """
    learnable_rev: StageDLearnable
    learnable_pri: StageDLearnable
    optimizer_rev: torch.optim.Optimizer
    optimizer_pri: torch.optim.Optimizer
    forced_type_logit_rev: float = -10.0    # sigmoid -> ~4e-5  -> revolute
    forced_type_logit_pri: float = +10.0    # sigmoid -> 1-4e-5 -> prismatic
    final_loss_rev: Optional[float] = None
    final_loss_pri: Optional[float] = None


def make_dual_clones(
    learnable: StageDLearnable,
    optimizer_state_dict: Dict,
    cfg: StageDConfig,
    optimizer_factory: Callable[[StageDLearnable], torch.optim.Optimizer],
) -> DualCloneState:
    """Deep-copy ``learnable`` twice, freezing type_logit in each branch.

    The optimizer state (AdamW moment buffers) is deep-copied too so each
    clone's optimizer can resume cleanly from the vote moment.

    Parameters
    ----------
    learnable : the current P1 state (call before applying any further updates).
    optimizer_state_dict : dict
        ``optimizer.state_dict()`` from the original optimizer.
    cfg : StageDConfig
    optimizer_factory : (learnable) -> Optimizer
        Used to instantiate the per-clone optimizer with the right param
        groups. Typically a closure ``lambda lr: torch.optim.AdamW(
        lr.make_param_groups(), lr=cfg.lr_scalar, ...)``; see train.py.
    """
    # Deep copy weights + buffers but NOT references to frozen TRELLIS / Wan.
    learnable_rev = copy.deepcopy(learnable)
    learnable_pri = copy.deepcopy(learnable)

    with torch.no_grad():
        learnable_rev.psi_param[6].fill_(-10.0)
        learnable_pri.psi_param[6].fill_(+10.0)

    opt_rev = optimizer_factory(learnable_rev)
    opt_pri = optimizer_factory(learnable_pri)
    opt_rev.load_state_dict(copy.deepcopy(optimizer_state_dict))
    opt_pri.load_state_dict(copy.deepcopy(optimizer_state_dict))

    return DualCloneState(
        learnable_rev=learnable_rev,
        learnable_pri=learnable_pri,
        optimizer_rev=opt_rev,
        optimizer_pri=opt_pri,
    )


def zero_type_logit_grad(learnable: StageDLearnable) -> None:
    """Zero the gradient on ``psi_param[6]`` (type_logit).

    Call right before ``optimizer.step()`` in each clone's inner loop so
    the forced type is preserved through the remaining training iters.
    Other psi_param slots (axis, origin, theta_max_raw, disp_max_raw)
    still receive gradient and are refined normally.
    """
    if learnable.psi_param.grad is not None:
        learnable.psi_param.grad[6] = 0.0


def commit_dual_clone(
    state: DualCloneState,
) -> Tuple[StageDLearnable, torch.optim.Optimizer, Literal["revolute", "prismatic"]]:
    """Select the lower-final-loss clone and return its state for use post-vote.

    Both ``final_loss_rev`` and ``final_loss_pri`` must be set before
    calling; the train loop assigns them at the end of each clone's
    extra-iter window from a sum of ``L_sds + L_rgb`` (or other choice).
    """
    if state.final_loss_rev is None or state.final_loss_pri is None:
        raise RuntimeError(
            "commit_dual_clone called before both clone final losses set"
        )
    if state.final_loss_rev <= state.final_loss_pri:
        committed_type: Literal["revolute", "prismatic"] = "revolute"
        winner_learnable = state.learnable_rev
        winner_optimizer = state.optimizer_rev
    else:
        committed_type = "prismatic"
        winner_learnable = state.learnable_pri
        winner_optimizer = state.optimizer_pri
    logger.info(
        "[stage_d dual-clone commit] rev_loss %.4f, pri_loss %.4f -> %s",
        state.final_loss_rev, state.final_loss_pri, committed_type,
    )
    return winner_learnable, winner_optimizer, committed_type


__all__ = [
    "TypeVoteResult", "run_type_vote",
    "DualCloneState", "make_dual_clones", "zero_type_logit_grad",
    "commit_dual_clone",
]
