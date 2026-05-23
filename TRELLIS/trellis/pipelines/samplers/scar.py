"""SCAR v3: Symmetric Consensus Attention Refinement sampler.

Replaces VGCF's mid-step sin schedule and 1/K normalization with an
early-step schedule ``alpha(s) = [0.7, 1.0, 1.0, 0.5]`` for s in 0..3
and lambda(s, t) = alpha(s) / t (removing the 1/K factor). This makes
the gradient push force sufficient to achieve full consensus at base
voxels (alpha=1 -> full alignment) rather than 1/K fraction.

The core mechanism is unchanged from VGCF: compute cross-state Tweedie
variance, log-percentile + sigmoid mask in space, M^2 selectivity,
gradient push toward consensus Tweedie in low-variance (base) regions.

See record/2026-04-17-pipeline-redesign.md section 3 for derivation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from easydict import EasyDict as edict
from tqdm import tqdm

from .flow_euler import FlowEulerGuidanceIntervalSampler


def generate_alpha_schedule(
    peak: float,
    total_steps: int,
    decay: str = "quadratic",
) -> List[float]:
    """Generate a per-step push strength schedule by decaying from ``peak``
    toward 0 over ``total_steps`` Euler steps.

    Rationale: push is most reliable at step 0 (Tweedie cross-state variance
    is at its most discriminative), and becomes noisier as mix homogenises
    the latent. Continuous decay matches push strength to mask reliability.

    The push formula has ``push = (alpha * t) * M^2 * diff`` (σ_t ≈ t in
    rectified flow; multiplies by σ_t rather than dividing, so it does not
    blow up at t→0). With ``t_s = 1 - s / (total_steps - 1)`` (approximately),
    the effective per-step push magnitude is:

      - linear    : alpha*t = peak * (1 - s/(total-1))        (linear decay)
      - quadratic : alpha*t = peak * (1 - s/(total-1))^3      (cubic decay)
      - cosine    : alpha*t smoothly tapers at both ends

    Parameters
    ----------
    peak : float
        Peak alpha value at step 0.
    total_steps : int
        Total number of Euler steps (schedule length).
    decay : {"quadratic", "linear", "cosine"}
        Shape of decay. See above.

    Returns
    -------
    list of float, length ``total_steps``
    """
    assert total_steps >= 1, f"total_steps must be >= 1, got {total_steps}"
    if total_steps == 1:
        return [float(peak)]
    denom = float(total_steps - 1)
    if decay == "linear":
        return [float(peak) * (1.0 - s / denom) for s in range(total_steps)]
    if decay == "quadratic":
        return [float(peak) * ((1.0 - s / denom) ** 2) for s in range(total_steps)]
    if decay == "cosine":
        import math
        return [
            float(peak) * 0.5 * (1.0 + math.cos(math.pi * s / denom))
            for s in range(total_steps)
        ]
    raise ValueError(f"unknown decay: {decay!r}; expected linear / quadratic / cosine")


def compute_tweedie_variance_mask(
    x_0: torch.Tensor,
    tau_percentile: float = 0.65,
    eta: float = 0.5,
    active_fraction: float = 0.1,
    eps_log: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute a spatial mask at latent resolution where M~1 means base
    (cross-state latent consistent) and M~0 means move (cross-state latent
    varies).

    Design principle: SCAR's push operates in LATENT space, so the mask
    that decides "push vs don't push" must also be computed in LATENT
    space. Previous attempts decoded to 64^3 occupancy and pooled back to
    16^3; that approach introduced two systematic errors:
      1. Pool 64^3 -> 16^3 mixes base and move voxels within each 4^3 block,
         smearing the bimodal structure.
      2. Decoder's sigmoid non-linearity saturates, adding noise without
         adding signal (SS VAE has lambda_kl ~ 0, so latent ≈ occupancy
         encoding already; decoding round-trip is wasted work).

    Method:
      1. Compute per-voxel cross-state latent variance at 16^3 directly:
            sigma2(x) = sum_channel Var_k(x_0^(k)(x))
      2. Identify air voxels by latent energy percentile and exclude them:
            energy(x) = ||x0_bar(x)||^2
            active = energy >= percentile_(1-active_fraction)(energy)
         With ``active_fraction=0.1``, keep top 10% by energy (object
         typically occupies 5-15% of the 16^3 latent volume).
      3. Within active voxels, threshold ``log sigma2`` at ``tau_percentile``
         and apply sigmoid to get the soft mask.

    Parameters
    ----------
    x_0 : torch.Tensor
        Shape ``(K, C, D, H, W)``, Tweedie estimates for K states at the
        flow-model's latent resolution (16^3 for TRELLIS-image-large).
    tau_percentile : float, default 0.65
        Percentile of log-variance (over active voxels) used as threshold.
    eta : float, default 0.5
        Sigmoid sharpness (log-space).
    active_fraction : float, default 0.1
        Fraction of voxels kept as object by energy percentile. 0.1 means
        "keep top 10% by energy" -- matches typical object volume in the
        16^3 latent.
    eps_log : float, default 1e-6
        Added to sigma2 before log for numerical stability.

    Returns
    -------
    M : torch.Tensor
        Shape ``(D, H, W)`` at latent resolution, mask in [0, 1], 1 = base.
    active : torch.Tensor
        Shape ``(D, H, W)``, bool mask of non-air voxels.
    """
    x0_bar = x_0.mean(dim=0, keepdim=True)
    diff = x_0 - x0_bar
    # Cross-state variance at each latent voxel, summed over channels
    sigma2 = (diff * diff).sum(dim=1).mean(dim=0)

    # Air filter: keep only voxels with high latent energy (object)
    energy = (x0_bar[0] ** 2).sum(dim=0)
    energy_flat = energy.flatten()
    cut_q = max(0.0, 1.0 - active_fraction)
    energy_thresh = torch.quantile(energy_flat, cut_q)
    # `>=` so uniform-energy edge cases keep all voxels rather than none
    active = energy >= energy_thresh

    log_sigma2 = torch.log(sigma2 + eps_log)
    if active.any():
        tau = torch.quantile(log_sigma2[active], tau_percentile)
    else:
        tau = torch.quantile(log_sigma2.flatten(), tau_percentile)

    M = torch.sigmoid((tau - log_sigma2) / eta)
    M = M * active.to(M.dtype)
    return M, active


def apply_scar_gradient_push(
    v_cfg: torch.Tensor,
    x_0: torch.Tensor,
    M: torch.Tensor,
    alpha: float,
    t: float,
    w_floor: float = 0.0,
    active: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply SCAR gradient push to the per-state CFG velocity with a soft
    mask floor.

    v_aug^(k) = v_cfg^(k) + (alpha * t) * w(x)^2 * (x_0^(k) - x0_bar)

    where ``w(x) = [M(x) * (1 - w_floor) + w_floor] * active(x)``.

    The (alpha * t) scaling is σ_t-normalized (rectified flow: σ_t ≈ t),
    replacing the earlier (alpha / t) which diverged as t→0. See the NOTE
    inside the function body for the derivation pointer.

    Motivation for w_floor > 0:
      The variance-based mask M conflates (a) true move (high var), (b)
      base-interior generation noise (medium var), and (c) single-state
      rigid drift (high var at drift boundary). Hard-gating (w_floor=0)
      causes (b) and (c) to receive no push, leaving them uncorrected.
      A small floor ensures every object voxel receives at least
      ``w_floor^2`` fraction of the push strength, rescuing (b) and (c)
      at the cost of a bounded contamination in true-move regions.

    Parameters
    ----------
    v_cfg : torch.Tensor
        Shape ``(K, C, D, H, W)``.
    x_0 : torch.Tensor
        Shape ``(K, C, D, H, W)``, Tweedie estimates.
    M : torch.Tensor
        Shape ``(D, H, W)``, base-probability mask in [0, 1].
    alpha : float
        Alignment strength in [0, 1]; 1 = full alignment at base.
    t : float
        Current Euler step time in (0, 1].
    w_floor : float, default 0.0
        Minimum weight at active (object) voxels. When 0, hard gate
        (original behavior). When > 0, move voxels still receive
        ``w_floor^2 * alpha`` fraction of push. Recommended ~0.2.
    active : torch.Tensor, optional
        Shape ``(D, H, W)``, 0/1 object mask from the mask function. If
        provided, the floor is applied only inside the object; outside
        voxels still receive zero push. If None, floor is applied
        globally (may push air voxels — not recommended).

    Returns
    -------
    v_aug : torch.Tensor
        Same shape as `v_cfg`.
    """
    x0_bar = x_0.mean(dim=0, keepdim=True)
    diff = x_0 - x0_bar
    # Soft-weight with floor, gated by active (object) mask
    w = M * (1.0 - w_floor) + w_floor                        # (D, H, W)
    if active is not None:
        w = w * active.to(w.dtype)                           # zero outside object
    w_sq = (w * w).to(diff.dtype)
    # NOTE (2026-04-18): Scaled by (alpha * t) rather than (alpha / t).
    # σ_t ≈ t in rectified flow; the correct score-form guidance multiplies
    # by σ_t (not divides). The old α/t formulation amplified push as t→0
    # (Kynkäänniemi et al. 2024, arXiv:2404.07724). Effective push now
    # monotonically shrinks with t, matching mask reliability.
    push = (alpha * t) * w_sq[None, None] * diff
    return v_cfg + push


class SCARSampler(FlowEulerGuidanceIntervalSampler):
    """Stage-1 sampler with cross-state coupling. v3.3 (default) mixes in
    x_0 space; v4.3 legacy mixes in z_t space. Both keep the optional
    Tweedie-space gradient push as a separate mechanism that operates on
    the velocity prediction.

    Two independent cross-state mechanisms:

      Mix (steps 0..mix_steps-1): blend across K states.
         ``mix_space='x_0'`` (v3.3 default, method.md section 5.2):
            1) run model forward to obtain (x_0_pred, eps_pred, v_pred)
            2) mix x_0_pred per-state:
                  x_0_mixed^(k) = w_first * x_0^(0) + w_self * x_0^(k)
                                + w_last * x_0^(K-1)
            3) rebuild z_{t_next} from mixed x_0 and ORIGINAL per-state eps:
                  z_{t_next} = (1 - t_next) * x_0_mixed
                             + (sigma_min + (1 - sigma_min) * t_next) * eps_pred
            Mean is identical to the z_t-mix formulation, so spatial-position
            alignment is preserved; noise variance is preserved (not reduced
            by ||w||^2 < 1), so the model sees in-distribution z at the next
            step. This is the form that matches CHORD's W-RFSDS derivation.

         ``mix_space='z_t'`` (v4.3 legacy):
            mixed^(k) = w_first * z_t^(0) + w_self * z_t^(k) + w_last * z_t^(K-1)
            The mixed z_t is fed to the model forward AND propagated as the
            next step's z_t. Noise variance after mix shrinks by ||w||^2
            = 0.34 (for default weights), so the model sees slightly OOD
            inputs. Kept for ablation parity with the v4.3 experiments.

      Push (steps 0..len(alpha_schedule)-1, indexed directly): Tweedie
         gradient push on the velocity prediction
            v_aug^(k) = v_cfg^(k) + (alpha(s) * t) * M(x)^2 * (x_0^(k) - x0_bar)
         Refines base-region consistency at the velocity level. Push is
         independent of mix_space; with ``alpha_peak=0`` (current default),
         push is disabled and the augmented velocity equals the CFG velocity.

    Mask mechanism: Tweedie variance + log-space percentile + sigmoid +
    active energy filter (inherited from VGCF, unchanged).
    """

    DEFAULT_ALPHA_SCHEDULE = (0.7, 1.0, 1.0, 0.5)
    DEFAULT_MIX_WEIGHTS = (0.3, 0.4, 0.3)

    def __init__(
        self,
        sigma_min: float,
        alpha_schedule: Tuple[float, ...] = DEFAULT_ALPHA_SCHEDULE,
        active_fraction: float = 0.8,
        tau_percentile: float = 0.65,
        eps_log: float = 1e-6,
        eta: float = 0.5,
        mix_steps: int = 0,
        mix_weights: Tuple[float, float, float] = DEFAULT_MIX_WEIGHTS,
        extreme_mix_mode: str = "symmetric",
        w_floor: float = 0.0,
        scar_enabled: bool = True,
        mix_space: str = "x_0",
    ) -> None:
        super().__init__(sigma_min=sigma_min)
        self.alpha_schedule = tuple(alpha_schedule)
        self.active_fraction = float(active_fraction)
        self.tau_percentile = float(tau_percentile)
        self.eps_log = float(eps_log)
        self.eta = float(eta)
        self.mix_steps = int(mix_steps)
        self.mix_weights = tuple(mix_weights)
        if len(self.mix_weights) != 3:
            raise ValueError(f"mix_weights must have 3 entries (w_first, w_self, w_last), got {self.mix_weights}")
        # How to mix the K latents at each mix step:
        #   "symmetric": per-state formula mixed^k = w_first*s_0 + w_self*s_k + w_last*s_{K-1}
        #                (each state keeps w_self of itself)
        #   "mean_of_middles": UNIFORM formula mixed = w_first*s_0 + w_self*mean(s_1..s_{K-2})
        #                       + w_last*s_{K-1} applied to ALL states, so all K latents
        #                       become identical after mix. No per-state self retention.
        #                       Wipes out latent-init bias; requires K >= 4.
        if extreme_mix_mode not in ("symmetric", "mean_of_middles"):
            raise ValueError(f"extreme_mix_mode must be 'symmetric' or 'mean_of_middles', got {extreme_mix_mode!r}")
        self.extreme_mix_mode = extreme_mix_mode
        # Soft mask floor: move voxels still receive w_floor^2 * alpha push,
        # rescuing base-interior generation noise and single-state drifts
        # that the hard mask misclassifies. See apply_scar_gradient_push.
        self.w_floor = float(w_floor)
        self.scar_enabled = bool(scar_enabled)
        # Mix space: "x_0" (v3.3 default, mix clean signal estimate, preserves
        # noise variance via per-state eps reconstruction) or "z_t" (v4.3
        # legacy, mix noisy latent directly).
        if mix_space not in ("x_0", "z_t"):
            raise ValueError(f"mix_space must be 'x_0' or 'z_t', got {mix_space!r}")
        self.mix_space = mix_space

    def _alpha_for_step(self, step_idx: int) -> float:
        """Push alpha for step_idx. ``alpha_schedule`` is indexed by
        ``step_idx`` DIRECTLY (no mix_steps offset). This allows push and
        mix to overlap in time (both active in early steps when mask is
        most discriminative, push decays as mask degrades)."""
        if step_idx < len(self.alpha_schedule):
            return float(self.alpha_schedule[step_idx])
        return 0.0

    def _mix_x_t(self, sample: torch.Tensor, step_idx: int) -> Tuple[torch.Tensor, bool]:
        """Blend x_t across first/self/last in Phase 1.

        Two modes controlled by ``self.extreme_mix_mode``:

        - ``"symmetric"`` (default, per-state formula preserving individual
          self-identity):
              mixed^(k) = w_first * s_0 + w_self * s_k + w_last * s_{K-1}
          Each state retains its own latent at w_self weight. Extremes
          (k=0, K-1) effectively keep w_first + w_self = 0.7 of themselves.

        - ``"mean_of_middles"`` (ALL states use the same uniform formula, no
          per-state self retention):
              mixed^(k) = w_first * s_0 + w_self * mean(s_1..s_{K-2}) + w_last * s_{K-1}
              (same tensor for every k)
          After this mix, all K latents are IDENTICAL. Any per-state
          differentiation must come from the subsequent DiT forward's v^(k)
          (driven by per-state c_k). Requires K >= 4 so there are middle
          states. Use when per-state latent-init bias is the problem (e.g.,
          state 0 drifts due to weak image constraint); the uniform mix
          wipes out latent-init bias entirely.

          RISK: extreme states lose ALL latent identity during mix phase.
          Only c_k in later plain-sampling steps can re-differentiate them.
          If DiT can't recover extreme topology (e.g., state 0's closed
          solid interior) from c_0 alone, this mode will destroy it. Watch
          for state 0 losing its closed-topology in the output.

        Parameters
        ----------
        sample : torch.Tensor
            Shape ``(K, C, D, H, W)``, current latent trajectory.
        step_idx : int
            Current Euler step index.

        Returns
        -------
        mixed : torch.Tensor
            Either blended sample (if step_idx < mix_steps) or sample unchanged.
        was_mixed : bool
            Whether mixing was applied.
        """
        if not self.scar_enabled or step_idx >= self.mix_steps or sample.shape[0] < 2:
            return sample, False
        K = sample.shape[0]
        w_first, w_self, w_last = self.mix_weights

        if self.extreme_mix_mode == "mean_of_middles" and K >= 4:
            # Uniform mix: all K states receive the same blended latent.
            # mixed = w_first * s_0 + w_self * mean(s_1..s_{K-2}) + w_last * s_{K-1}
            middle_mean = sample[1:K - 1].mean(dim=0, keepdim=True)     # (1, C, D, H, W)
            mixed_single = (
                w_first * sample[0:1]
                + w_self * middle_mean
                + w_last * sample[-1:]
            )                                                            # (1, C, D, H, W)
            mixed = mixed_single.expand(K, -1, -1, -1, -1).contiguous()
        else:
            # Symmetric mode (or K < 4 fallback): per-state self retention.
            mixed = w_first * sample[0:1] + w_self * sample + w_last * sample[-1:]
        return mixed, True

    def _mix_x_0(
        self, pred_x_0: torch.Tensor, step_idx: int,
    ) -> Tuple[torch.Tensor, bool]:
        """v3.3 mix: blend the model's CLEAN-SIGNAL estimate x_0_pred across K
        states. Same per-state symmetric formula as ``_mix_x_t`` but applied
        to ``pred_x_0`` after the model forward, not to ``z_t`` before it.

        After this call, the caller is expected to reconstruct ``z_{t_next}``
        from ``x_0_mixed`` and the ORIGINAL per-state ``eps_pred`` so that
        per-sample noise variance is preserved (ODE consistency).

        Parameters
        ----------
        pred_x_0 : torch.Tensor
            Shape ``(K, C, D, H, W)``, Tweedie clean estimate at the current step.
        step_idx : int
            Current Euler step index. Mix is active only when ``step_idx <
            self.mix_steps`` and ``scar_enabled`` and ``K >= 2``.

        Returns
        -------
        mixed : torch.Tensor
            Same shape as ``pred_x_0``. Identity when mix is inactive.
        was_mixed : bool
            Whether the mix was applied.
        """
        if not self.scar_enabled or step_idx >= self.mix_steps or pred_x_0.shape[0] < 2:
            return pred_x_0, False
        K = pred_x_0.shape[0]
        w_first, w_self, w_last = self.mix_weights

        if self.extreme_mix_mode == "mean_of_middles" and K >= 4:
            middle_mean = pred_x_0[1:K - 1].mean(dim=0, keepdim=True)
            mixed_single = (
                w_first * pred_x_0[0:1]
                + w_self * middle_mean
                + w_last * pred_x_0[-1:]
            )
            mixed = mixed_single.expand(K, -1, -1, -1, -1).contiguous()
        else:
            mixed = (
                w_first * pred_x_0[0:1]
                + w_self * pred_x_0
                + w_last * pred_x_0[-1:]
            )
        return mixed, True

    @torch.no_grad()
    def sample(
        self,
        model: torch.nn.Module,
        noise: torch.Tensor,
        cond: torch.Tensor,
        neg_cond: torch.Tensor,
        steps: int = 12,
        rescale_t: float = 1.0,
        cfg_strength: float = 7.5,
        cfg_interval: Tuple[float, float] = (0.0, 1.0),
        verbose: bool = True,
        **kwargs: Any,
    ) -> edict:
        K = noise.shape[0]
        sample = noise.clone()
        sigma_min = self.sigma_min

        t_seq = np.linspace(1.0, 0.0, steps + 1)
        t_seq = rescale_t * t_seq / (1.0 + (rescale_t - 1.0) * t_seq)
        t_pairs = list(zip(t_seq[:-1], t_seq[1:]))

        ret = edict({
            "samples": None,
            "pred_x_t": [],
            "pred_x_0": [],
            "scar_diagnostics": [],
        })

        pbar = tqdm(
            t_pairs,
            desc=f"SCAR sampling (mix_space={self.mix_space})",
            disable=not verbose,
        )
        for step_idx, (t, t_prev) in enumerate(pbar):
            t = float(t)
            t_prev = float(t_prev)
            dt = t - t_prev

            # ----- Mix in z_t (legacy v4.3) BEFORE model forward -----
            zt_was_mixed = False
            if self.mix_space == "z_t":
                # The mixed z_t is fed to the model AND propagated.
                sample, zt_was_mixed = self._mix_x_t(sample, step_idx)

            # ----- Model forward -----
            pred_x_0, pred_eps, pred_v = self._get_model_prediction(
                model, sample, t,
                cond=cond, neg_cond=neg_cond,
                cfg_strength=cfg_strength,
                cfg_interval=cfg_interval,
                **kwargs,
            )

            # ----- Optional Tweedie-space gradient push on velocity -----
            alpha = self._alpha_for_step(step_idx) if self.scar_enabled else 0.0
            v_aug = pred_v
            diag: Dict[str, Any] = {
                "step": step_idx, "t": t, "t_prev": t_prev,
                "alpha": alpha,
                "mix_space": self.mix_space,
                "mixed": False,
            }

            if alpha > 0.0 and K >= 2:
                M, active = compute_tweedie_variance_mask(
                    pred_x_0,
                    tau_percentile=self.tau_percentile,
                    eta=self.eta,
                    active_fraction=self.active_fraction,
                    eps_log=self.eps_log,
                )
                v_aug = apply_scar_gradient_push(
                    v_cfg=pred_v, x_0=pred_x_0, M=M, alpha=alpha, t=t,
                    w_floor=self.w_floor, active=active,
                )
                diag.update({
                    "M_mean": float(M.mean().item()),
                    "M_mean_active": float(M[active].mean().item() if active.any() else 0.0),
                    "active_voxels": int(active.sum().item()),
                    "push_norm": float((v_aug - pred_v).norm().item()),
                })

            # ----- Mix in x_0 (v3.3 default) AFTER model forward -----
            if self.mix_space == "x_0":
                x_0_mixed, x0_was_mixed = self._mix_x_0(pred_x_0, step_idx)
                if x0_was_mixed:
                    # Reconstruct z_{t_next} from mixed x_0 and ORIGINAL per-state eps.
                    # Mean = mean of z_t-mix formulation (position alignment preserved).
                    # Variance: pred_eps unchanged -> noise variance preserved.
                    pred_x_prev = (
                        (1.0 - t_prev) * x_0_mixed
                        + (sigma_min + (1.0 - sigma_min) * t_prev) * pred_eps
                    )
                    diag["mixed"] = True
                    diag["mix_x0_l2_change"] = float(
                        (x_0_mixed - pred_x_0).norm().item()
                    )
                else:
                    # No mix this step: plain Euler step with (optional) augmented v.
                    pred_x_prev = sample - dt * v_aug
            else:
                # mix_space == "z_t" — mix already happened pre-forward.
                pred_x_prev = sample - dt * v_aug
                diag["mixed"] = bool(zt_was_mixed)

            sample = pred_x_prev

            ret.pred_x_t.append(pred_x_prev)
            ret.pred_x_0.append(pred_x_0)
            ret.scar_diagnostics.append(diag)

        ret.samples = sample
        return ret
