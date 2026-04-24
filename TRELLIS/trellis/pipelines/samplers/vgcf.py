"""VGCF: Velocity-Guided Consensus Flow sampler.

A subclass of :class:`FlowEulerGuidanceIntervalSampler` that runs K
parallel sampling tracks coupled through a velocity-space consistency
term. Designed for TRELLIS Stage-1 where each track is conditioned on a
different image state of the same articulated object.

The core idea (paper method §1): at each Euler step compute a per-voxel
cross-state variance of the Tweedie estimates `x0_hat^{(k)}`. Voxels
with low variance represent the stationary base and receive a
consensus-pulling velocity correction `lambda(t) * g^{(k)}`; voxels
with high variance represent the moving part and are left free. The
consensus gradient is a Jacobian-free DPS approximation, so no extra
network evaluations are required per step.

VGCF overrides only :meth:`sample`; it reuses
:meth:`FlowEulerSampler._get_model_prediction`, which correctly plumbs
classifier-free guidance and guidance interval through the MRO chain
provided by the parent classes.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from easydict import EasyDict as edict
from tqdm import tqdm

from .flow_euler import FlowEulerGuidanceIntervalSampler


class VGCFSampler(FlowEulerGuidanceIntervalSampler):
    """Velocity-Guided Consensus Flow sampler for K-track Stage-1 sampling.

    The P0 mask-quality revision adds four knobs on top of the original
    paper formulation; all of them target the observed long-tail issue
    where median(sigma2) + fixed 1e-8 active filter yields ``M_mean~0.4``
    on doors/cabinets:

    * ``active_fraction``: keep only the top fraction of voxels by
      ``||x0_bar||^2`` energy, excluding empty-air voxels that otherwise
      pull the variance median down.
    * ``tau_percentile``: threshold log(sigma2) at a chosen percentile
      (default 0.65) instead of the median. This targets the physical
      prior that 60-80% of a door/cabinet is base.
    * ``eps_log`` / log-space sigmoid: compresses the 5+ orders of
      magnitude sigma2 long-tail so a single ``eta`` works.
    * ``lambda_schedule``: "warmup" (sin-peaked in the middle, 0 at both
      ends) replaces the legacy cosine decay, because step 0-2 has the
      noisiest Tweedie estimates and we do not want peak guidance there.
    """

    def __init__(
        self,
        sigma_min: float,
        lambda_max: float = 1.0,
        t_stop: float = 0.2,
        eta: float = 0.5,
        vgcf_enabled: bool = True,
        active_fraction: float = 0.8,
        tau_percentile: float = 0.65,
        eps_log: float = 1.0e-6,
        lambda_schedule: str = "warmup",
    ) -> None:
        super().__init__(sigma_min=sigma_min)
        self.lambda_max = float(lambda_max)
        self.t_stop = float(t_stop)
        self.eta = float(eta)
        self.vgcf_enabled = bool(vgcf_enabled)
        self.active_fraction = float(active_fraction)
        self.tau_percentile = float(tau_percentile)
        self.eps_log = float(eps_log)
        self.lambda_schedule = str(lambda_schedule).lower()
        if self.lambda_schedule not in ("warmup", "cosine"):
            raise ValueError(
                f"lambda_schedule must be 'warmup' or 'cosine', got {self.lambda_schedule}"
            )

    # ------------------------------------------------------------------
    # Sampling loop
    # ------------------------------------------------------------------

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
        """Run K-track Stage-1 sampling with velocity-space consensus guidance.

        Parameters
        ----------
        model : torch.nn.Module
            Frozen flow-matching velocity network, e.g.
            ``pipe.models['sparse_structure_flow_model']``.
        noise : torch.Tensor
            Shape ``(K, 8, 16, 16, 16)``. Caller is responsible for the
            shared-noise initialization (``eps.repeat(K, 1, 1, 1, 1)``)
            that preserves the per-track marginal at ``t = 1``.
        cond : torch.Tensor
            DINOv2 patch embeddings stacked across K states,
            shape ``(K, 1369, 768)``.
        neg_cond : torch.Tensor
            Unconditional embedding broadcast to match ``cond``.
        steps : int, default 12
            Number of Euler integration steps (TRELLIS Stage-1 default).
        rescale_t : float, default 1.0
            Flow schedule rescale factor (TRELLIS default 1.0).
        cfg_strength : float, default 7.5
            Classifier-free guidance weight.
        cfg_interval : tuple of float, default (0.0, 1.0)
            Interval over which to apply CFG.
        verbose : bool, default True
            Show a tqdm progress bar.

        Returns
        -------
        edict with:
            samples : torch.Tensor
                Final latents ``(K, 8, 16, 16, 16)``.
            pred_x_t : List[torch.Tensor]
                Per-step ``x_{t-1}`` predictions.
            pred_x_0 : List[torch.Tensor]
                Per-step Tweedie estimates.
            vgcf_diagnostics : List[Dict]
                Per-step metrics for logging.
        """
        K = noise.shape[0]
        sample = noise.clone()

        t_seq = np.linspace(1.0, 0.0, steps + 1)
        t_seq = rescale_t * t_seq / (1.0 + (rescale_t - 1.0) * t_seq)
        t_pairs = list(zip(t_seq[:-1], t_seq[1:]))

        ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": [],
                     "vgcf_diagnostics": []})

        pbar = tqdm(t_pairs, desc="VGCF sampling", disable=not verbose)
        for step_idx, (t, t_prev) in enumerate(pbar):
            t = float(t)
            t_prev = float(t_prev)
            dt = t - t_prev

            # CFG + guidance-interval velocity (plumbed via MRO).
            pred_x_0, _pred_eps, pred_v = self._get_model_prediction(
                model, sample, t,
                cond=cond,
                neg_cond=neg_cond,
                cfg_strength=cfg_strength,
                cfg_interval=cfg_interval,
                **kwargs,
            )

            v_aug = pred_v
            diag: Dict[str, Any] = {"step": step_idx, "t": t, "lambda": 0.0}

            if self.vgcf_enabled and t > self.t_stop and K >= 2:
                # Cross-state mean and per-voxel variance on the Tweedie
                # estimate, reduced over the 8 latent channels.
                x0_bar = pred_x_0.mean(dim=0, keepdim=True)          # (1,C,D,H,W)
                diff = pred_x_0 - x0_bar                              # (K,C,D,H,W)
                sigma2 = (diff * diff).sum(dim=1).mean(dim=0)         # (D,H,W)

                # --- P0: active voxel detection via energy percentile ---
                # Drop the bottom (1 - active_fraction) of voxels by
                # ||x0_bar||^2 energy. These are effectively "air" voxels
                # that contribute a near-zero sigma2 mass and pull the
                # threshold percentile down.
                energy = x0_bar[0].pow(2).sum(0)                      # (D,H,W)
                energy_flat = energy.flatten()
                if energy_flat.numel() > 0:
                    cut_q = max(0.0, 1.0 - self.active_fraction)
                    energy_thresh = torch.quantile(energy_flat, cut_q)
                    active = energy > energy_thresh
                else:
                    active = torch.ones_like(energy, dtype=torch.bool)

                # --- P0: log-space sigma2 + percentile threshold ---
                # log compresses the 5+ orders of magnitude long-tail so
                # that a single percentile is a stable estimator of the
                # "typical base" magnitude.
                log_sigma2 = torch.log(sigma2 + self.eps_log)         # (D,H,W)
                if active.any():
                    tau = torch.quantile(log_sigma2[active], self.tau_percentile)
                else:
                    tau = torch.quantile(log_sigma2.flatten(), self.tau_percentile)

                M = torch.sigmoid((tau - log_sigma2) / self.eta)      # (D,H,W)
                M = M * active.to(M.dtype)
                M5 = M[None, None]                                    # (1,1,D,H,W)

                g_k = (M5 * M5) * diff / float(K)                     # (K,C,D,H,W)

                # --- P0: warmup (sin) or legacy cosine schedule ---
                progress = (1.0 - t) / max(1.0 - self.t_stop, 1e-6)
                if self.lambda_schedule == "warmup":
                    # 0 at both ends, peak at the middle (progress=0.5).
                    lam = self.lambda_max * math.sin(math.pi * progress)
                else:  # "cosine" legacy
                    lam = self.lambda_max * 0.5 * (1.0 + math.cos(math.pi * progress))
                v_aug = pred_v + lam * g_k

                diag.update({
                    "lambda": float(lam),
                    "sigma2_median": float(sigma2.median().item()),
                    "sigma2_max": float(sigma2.max().item()),
                    "log_sigma2_median": float(log_sigma2.median().item()),
                    "log_sigma2_p65": float(
                        torch.quantile(log_sigma2[active], self.tau_percentile).item()
                        if active.any() else float("nan")
                    ),
                    "tau_used": float(tau.item()),
                    "M_mean": float(M.mean().item()),
                    "M_mean_active": float(M[active].mean().item() if active.any() else 0.0),
                    "active_voxels": int(active.sum().item()),
                    "schedule": self.lambda_schedule,
                })
            elif not self.vgcf_enabled:
                diag["reason"] = "disabled"
            elif K < 2:
                diag["reason"] = "K<2"
            else:
                diag["reason"] = f"t<=t_stop ({self.t_stop})"

            pred_x_prev = sample - dt * v_aug
            sample = pred_x_prev

            ret.pred_x_t.append(pred_x_prev)
            ret.pred_x_0.append(pred_x_0)
            ret.vgcf_diagnostics.append(diag)

        ret.samples = sample
        return ret
