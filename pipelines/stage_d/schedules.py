"""Training schedules: tau (inverse-CDF), CFG, phase manager, lambdas, gates.

Single source of truth for every per-iter scalar that varies with training
progress ``f_global = it / total_iters`` in [0, 1]. The phase manager
classifies ``f_global`` into one of:

  - "warmup_g_minus"  [0, 0.05]
  - "warmup_g0"       [0.05, 0.10]
  - "main_g1a"        [0.10, 0.50]      learnable type_logit; two-branch render
  - "main_g1b"        [0.50, 0.65]      type committed (S3); single-branch
  - "transition"      [0.65, 0.75]      stop W-RFSDS push, prepare P2
  - "post"            [0.75, 1.00]      Stage D ends here; Stage F starts P2

All schedule functions are pure; nothing here imports torch.nn or owns
state.  ``sample_tau`` is the only stochastic call; everything else is
deterministic in ``f_global``.

CHORD note (Eq. 4): the recommended ``tau`` schedule is the inverse CDF of
the training-time weighting function ``w_hat(sigma)``. For SS-DiT this is
``logit-normal(mean=1.0, std=1.0)`` per
``paper/TRELLIS/configs/generation/ss_flow_img_dit_L_16l8_fp16.json``
("t_schedule": {"name": "logitNormal", "args": {"mean": 1.0, "std": 1.0}}).
This skill puts the most density around tau ~= sigmoid(1.0) = 0.73 with
sufficient mass at low values to drive late refinement.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

from .config import (
    StageDConfig,
    TRELLIS_SS_T_SCHEDULE_MEAN,
    TRELLIS_SS_T_SCHEDULE_STD,
)


# =============================================================================
# Phase classification
# =============================================================================

def phase_of(f_global: float, cfg: StageDConfig) -> str:
    """Classify training progress into a phase name."""
    if f_global < cfg.f_warmup_g_minus_end:
        return "warmup_g_minus"
    if f_global < cfg.f_warmup_g0_end:
        return "warmup_g0"
    if f_global < cfg.f_main_g1a_end:
        return "main_g1a"
    if f_global < cfg.f_main_g1b_end:
        return "main_g1b"
    if f_global < cfg.f_transition_end:
        return "transition"
    return "post"


def _lerp(a: float, b: float, t: float) -> float:
    """Clamp-free linear interpolation."""
    return a + (b - a) * t


def _phase_progress(f_global: float, f_start: float, f_end: float) -> float:
    """Fraction of progress through a phase, in [0, 1]."""
    if f_end <= f_start:
        return 0.0
    return max(0.0, min(1.0, (f_global - f_start) / (f_end - f_start)))


# =============================================================================
# tau sampling (W-RFSDS)
# =============================================================================

def _standard_normal_inverse_cdf(u: torch.Tensor) -> torch.Tensor:
    """Phi^{-1}(u) where Phi is the standard normal CDF.

    Computed via ``sqrt(2) * erfinv(2u - 1)`` (a standard identity).
    Result is finite for u in (0, 1); we clamp inputs into a safe range to
    avoid +/-inf at the edges.
    """
    u_safe = u.clamp(1.0e-6, 1.0 - 1.0e-6)
    return math.sqrt(2.0) * torch.erfinv(2.0 * u_safe - 1.0)


def sample_tau_inverse_cdf_logitnormal(
    mean: float = TRELLIS_SS_T_SCHEDULE_MEAN,
    std: float = TRELLIS_SS_T_SCHEDULE_STD,
    device: torch.device = torch.device("cpu"),
    seed: Optional[int] = None,
) -> float:
    """Draw a single ``tau`` from logit-normal(mean, std) via inverse CDF.

    Steps:
        1. u ~ Uniform(0, 1)
        2. z = mean + std * Phi^{-1}(u)
        3. tau = sigmoid(z)            in (0, 1)

    The inverse-CDF formulation is equivalent to direct sampling
    ``sigmoid(N(mean, std))`` but lets the caller supply a deterministic
    quantile (e.g. CHORD-style annealing via ``u = 1 - i / (I + 1)``).

    For random sampling, do NOT pass ``seed`` (we use the global generator).
    """
    if seed is not None:
        g = torch.Generator(device=device).manual_seed(int(seed))
        u = torch.rand(1, generator=g, device=device)
    else:
        u = torch.rand(1, device=device)
    z = mean + std * _standard_normal_inverse_cdf(u)
    return float(torch.sigmoid(z).item())


def sample_tau_chord_anneal(
    iter_idx: int,
    total_iters: int,
    mean: float = TRELLIS_SS_T_SCHEDULE_MEAN,
    std: float = TRELLIS_SS_T_SCHEDULE_STD,
    device: torch.device = torch.device("cpu"),
) -> float:
    """CHORD Eq. 4 deterministic annealing: ``tau_i = h^{-1}(1 - i / (I+1))``.

    With ``h`` = CDF of logit-normal(mean, std), the inverse evaluated at
    ``q = 1 - iter_idx / (total_iters + 1)`` decreases monotonically as
    training proceeds: coarse motion early (high tau), fine detail late
    (low tau). Use this as the default ablation contrast against
    ``sample_tau_inverse_cdf_logitnormal``.
    """
    if iter_idx < 0 or iter_idx >= total_iters:
        raise ValueError(
            f"iter_idx={iter_idx} out of range [0, {total_iters})"
        )
    q = 1.0 - float(iter_idx) / float(total_iters + 1)
    u = torch.tensor([q], device=device, dtype=torch.float32)
    z = mean + std * _standard_normal_inverse_cdf(u)
    return float(torch.sigmoid(z).item())


# =============================================================================
# CFG schedule (linear 25 -> 12 over training, per CHORD A.1)
# =============================================================================

def schedule_cfg(f_global: float, cfg: StageDConfig) -> float:
    """CFG scale for W-RFSDS as a function of training progress.

    Stage D runs from 0 to ``f_transition_end`` (default 0.75). Past that
    the project is in Stage F (P2 texture) which uses a constant
    ``cfg.cfg_p2_default``.

    Defaults:
        warmup_g_minus  (0-5%)   :  25.0  (highest; SS-DiT skipped anyway, so unused)
        warmup_g0       (5-10%)  :  25.0
        main_g1a / g1b  (10-65%) :  lerp(25, 20)  ← gradient over the bulk of P1
        transition      (65-75%) :  lerp(20, 16)
        post / Stage F  (>= 75%) :  12.0
    """
    phase = phase_of(f_global, cfg)
    if phase in ("warmup_g_minus", "warmup_g0"):
        return cfg.cfg_warmup_g0
    if phase in ("main_g1a", "main_g1b"):
        s = _phase_progress(f_global, cfg.f_warmup_g0_end, cfg.f_main_g1b_end)
        return _lerp(cfg.cfg_warmup_g0, cfg.cfg_main_g1_end, s)
    if phase == "transition":
        s = _phase_progress(f_global, cfg.f_main_g1b_end, cfg.f_transition_end)
        return _lerp(cfg.cfg_main_g1_end, cfg.cfg_transition_end, s)
    # post / Stage F
    return cfg.cfg_p2_default


# =============================================================================
# t_ss sampler (SS-DiT one-step inner timestep)
# =============================================================================

def sample_t_ss(f_global: float, cfg: StageDConfig,
                device: torch.device = torch.device("cpu")) -> Optional[float]:
    """Inner SS-DiT timestep for the one-step refiner forward.

    ``warmup_g_minus`` returns ``None``: SS-DiT is skipped in that phase
    (only Delta_z_s + alpha_g get gradient via the SS-VAE decoder path; see
    pipeline.md section 9.6 / method.md 10.2). Other phases sample from
    the mid-flow region where DIFT-analogous semantics are richest.
    """
    phase = phase_of(f_global, cfg)
    if phase == "warmup_g_minus":
        return None
    if phase == "warmup_g0":
        return float(cfg.t_ss_warmup_g0_fixed)
    if phase in ("main_g1a", "main_g1b"):
        u = torch.rand(1, device=device).item()
        return cfg.t_ss_main_low + u * (cfg.t_ss_main_high - cfg.t_ss_main_low)
    if phase == "transition":
        u = torch.rand(1, device=device).item()
        return cfg.t_ss_transition_low + u * (cfg.t_ss_transition_high - cfg.t_ss_transition_low)
    # ★ Fix SD-3: post phase. Stage D's outer loop in train.py iterates up to
    # cfg.total_iters; if any iter lands in [f_transition_end, 1.0] (e.g. due
    # to staged Stage F resolution switch, see pipeline.md sec 14.1), the
    # _regular_forward call would receive None and crash. We return the
    # transition_low value as a safe fallback (preserves the schedule's
    # downward trend without introducing a new mid-flow region). Note:
    # Stage F (P2 texture) has its own tau sampling at lower noise.
    if phase == "post":
        return float(cfg.t_ss_transition_low)
    return None


# =============================================================================
# BinaryConcrete temperatures
# =============================================================================

def schedule_gate_temperature(f_global: float, cfg: StageDConfig) -> Tuple[float, float]:
    """``(T_g, T_m)`` temperatures for ``binary_concrete_ste``.

    Higher T -> softer gate (smoother gradient, fuzzier 0/1 forward).
    Lower T  -> sharper gate (closer to true binary, but spikier gradient).

    Anneal:
        warmup           : T_warmup (default 1.5)
        main_g1a / g1b   : lerp(T_warmup, T_main_end) over the two phases
        transition       : lerp(T_main_end, T_transition_end)
        post             : T_transition_end (Stage F holds it constant)
    """
    phase = phase_of(f_global, cfg)
    if phase in ("warmup_g_minus", "warmup_g0"):
        return cfg.T_g_warmup, cfg.T_m_warmup
    if phase in ("main_g1a", "main_g1b"):
        s = _phase_progress(f_global, cfg.f_warmup_g0_end, cfg.f_main_g1b_end)
        T_g = _lerp(cfg.T_g_warmup, cfg.T_g_main_end, s)
        T_m = _lerp(cfg.T_m_warmup, cfg.T_m_main_end, s)
        return T_g, T_m
    if phase == "transition":
        s = _phase_progress(f_global, cfg.f_main_g1b_end, cfg.f_transition_end)
        T_g = _lerp(cfg.T_g_main_end, cfg.T_g_transition_end, s)
        T_m = _lerp(cfg.T_m_main_end, cfg.T_m_transition_end, s)
        return T_g, T_m
    return cfg.T_g_transition_end, cfg.T_m_transition_end


# =============================================================================
# Head lambda ramps (sup / part / joint)
# =============================================================================

def schedule_head_lambdas(f_global: float, cfg: StageDConfig
                          ) -> Tuple[float, float, float]:
    """``(lambda_sup, lambda_part, lambda_joint)`` from method.md section 10.

    Phase-gated ramp-in:
        f < 0.05          : (0, 0, 0)        - warmup_g_minus, heads off
        0.05 <= f < 0.10  : (0, 0.02, 0)     - warmup_g0, small lambda_part anchor
        0.10 <= f < 0.30  : lerp to (max, max, max)
        f >= 0.30         : (lambda_sup_max, lambda_part_max, lambda_joint_max)
    """
    if f_global < cfg.f_warmup_g_minus_end:
        return 0.0, 0.0, 0.0
    if f_global < cfg.f_warmup_g0_end:
        return 0.0, 0.02, 0.0
    ramp_end = cfg.f_warmup_g0_end + 0.20    # 30% global progress
    if f_global < ramp_end:
        s = _phase_progress(f_global, cfg.f_warmup_g0_end, ramp_end)
        return (
            _lerp(0.0, cfg.lambda_sup_max,   s),
            _lerp(0.02, cfg.lambda_part_max, s),
            _lerp(0.0, cfg.lambda_joint_max, s),
        )
    return cfg.lambda_sup_max, cfg.lambda_part_max, cfg.lambda_joint_max


# =============================================================================
# Per-loss lambda schedules
# =============================================================================

def schedule_w_rfsds_weights(f_global: float, cfg: StageDConfig
                              ) -> Tuple[float, float, float]:
    """``(lambda_sds, lambda_lat, lambda_rgb)``.

    method.md section 10.1 defaults:
        warmup_g_minus :  0.0, 0.0, 0.0     (no W-RFSDS, no Wan call)
        warmup_g0      :  1.0, 0.0, 0.0     (SDS only, no latent recon, no rgb)
        main_g1a/b     :  1.0, 0.1, 0.0     (SDS + light lat_rec for stability)
        transition     :  0.5, 0.5, 0.1     (SDS down, lat_rec up, rgb starts)
        post (Stage F) :  handled by Stage F itself

    Note: ``lambda_lat`` is for ``latent_recon_loss`` (not in CHORD). Stage D
    keeps it disabled by default at 0 in P1 (C2 fix). The values above are
    the legacy v3.3 defaults; set them to 0 to ablate.
    """
    phase = phase_of(f_global, cfg)
    if phase == "warmup_g_minus":
        return 0.0, 0.0, 0.0
    if phase == "warmup_g0":
        return 1.0, 0.0, 0.0
    if phase in ("main_g1a", "main_g1b"):
        return 1.0, 0.1, 0.0
    if phase == "transition":
        return 0.5, 0.5, 0.1
    return 0.2, 1.0, 1.0    # post / handed off to Stage F


def schedule_lambda_shell(f_global: float, cfg: StageDConfig) -> float:
    """``L_shell_sparse`` weight: ramp on between 10% and 30%, then constant.

    Off in warmup so the optimizer first finds plausible occupancy before
    the sparsity prior starts trimming uncertain shell voxels.
    """
    if f_global < cfg.f_warmup_g0_end:
        return 0.0
    ramp_end = cfg.f_warmup_g0_end + 0.20
    if f_global < ramp_end:
        s = _phase_progress(f_global, cfg.f_warmup_g0_end, ramp_end)
        return _lerp(0.0, cfg.lambda_shell_main, s)
    return cfg.lambda_shell_main


def schedule_lambda_m_prior(f_global: float, cfg: StageDConfig) -> float:
    """``L_m_prior`` weight: only nonzero in warmup_g0; decays to 0 by 30%."""
    if f_global < cfg.f_warmup_g_minus_end:
        return 0.0
    if f_global < cfg.f_warmup_g0_end:
        return cfg.lambda_m_prior_warmup
    decay_end = cfg.f_warmup_g0_end + 0.20
    if f_global < decay_end:
        s = _phase_progress(f_global, cfg.f_warmup_g0_end, decay_end)
        return _lerp(cfg.lambda_m_prior_warmup, 0.0, s)
    return 0.0


# =============================================================================
# Convenience snapshot
# =============================================================================

def schedule_snapshot(f_global: float, cfg: StageDConfig,
                      device: torch.device = torch.device("cpu")) -> dict:
    """Return all schedule values at this iter, useful for logging.

    Does NOT consume randomness for tau / t_ss; those are sampled
    separately at training-loop time.
    """
    phase = phase_of(f_global, cfg)
    cfg_scale = schedule_cfg(f_global, cfg)
    T_g, T_m = schedule_gate_temperature(f_global, cfg)
    l_sup, l_part, l_joint = schedule_head_lambdas(f_global, cfg)
    l_sds, l_lat, l_rgb = schedule_w_rfsds_weights(f_global, cfg)
    l_shell = schedule_lambda_shell(f_global, cfg)
    l_m_prior = schedule_lambda_m_prior(f_global, cfg)
    return {
        "phase": phase,
        "f_global": f_global,
        "cfg_scale": cfg_scale,
        "T_g": T_g, "T_m": T_m,
        "lambda_sup": l_sup, "lambda_part": l_part, "lambda_joint": l_joint,
        "lambda_sds": l_sds, "lambda_lat": l_lat, "lambda_rgb": l_rgb,
        "lambda_shell": l_shell, "lambda_m_prior": l_m_prior,
    }


__all__ = [
    "phase_of",
    "sample_tau_inverse_cdf_logitnormal", "sample_tau_chord_anneal",
    "schedule_cfg", "sample_t_ss",
    "schedule_gate_temperature",
    "schedule_head_lambdas", "schedule_w_rfsds_weights",
    "schedule_lambda_shell", "schedule_lambda_m_prior",
    "schedule_snapshot",
]
