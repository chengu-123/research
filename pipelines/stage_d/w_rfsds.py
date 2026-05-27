"""W-RFSDS distillation from Wan2.2 I2V into our rendered video.

The gradient signal that drives all Stage D learning (geometry, joint,
gates) flows from this single function. The math is CHORD Eq.(3)
(arxiv:2601.04194):

    L_W-RFSDS(theta; z, y) = E_{sigma ~ w_hat(sigma), eps}
                              [ (v_hat(z_sigma; sigma, y) - eps + z) . dz/dtheta ]

where:
    - z       = Wan_VAE.encode(rendered_video)              (grad-enabled)
    - z_sigma = (1 - sigma) * z + sigma * eps                (CHORD forward)
    - v_hat   = Wan2.2 DiT velocity prediction              (no-grad, dual-expert)
    - sigma   = tau ~ inverse-CDF of logit-normal(mean, std) (schedules.py)

Implementation uses ``residual = v_pred - eps + z.detach()`` and forms the
SDS loss as ``loss = (residual.detach() * z).sum() / z.numel()``. The
``.detach()`` chain ensures only z's gradient w.r.t. the renderer flows
backward — the residual is treated as a constant target direction.

Dual-expert switch (Wan2.2 wan_i2v_A14B.py:36 boundary=0.900):
    high_noise_model  if  tau * num_train_timesteps >= 900
    low_noise_model   otherwise

Memory:
    - Wan VAE encode + backward is the expensive step (CHORD §D
      Limitations explicitly acknowledges this).
    - Wan DiT forwards (cond + uncond) run under torch.no_grad so we don't
      retain activations for 40-layer DiT * 2 forwards.
    - On a 464x832 frame x 21 frames, the latent is [16, 6, 58, 104] which
      is ~5MB fp32; grad through the VAE encoder peaks around ~10 GB for
      A14B. The bookkeeping is contained inside this module.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from .config import (
    F_FRAMES,
    F_LATENT,
    H_LATENT,
    H_PIXEL,
    W_LATENT,
    W_PIXEL,
    WAN_BOUNDARY_NORMALIZED,
    WAN_LATENT_CH,
    WAN_NUM_TRAIN_TIMESTEPS,
)


# =============================================================================
# Wan2.2 component loader
# =============================================================================

def _ensure_wan_on_sys_path(repo_root: str) -> None:
    """Insert ``<repo_root>/Wan2.2`` on ``sys.path`` (same as Stage A)."""
    wan_root = os.path.join(repo_root, "Wan2.2")
    if not os.path.isdir(wan_root):
        raise RuntimeError(
            f"Stage D W-RFSDS requires vendored Wan2.2 at {wan_root!r}; "
            "directory not found."
        )
    if wan_root not in sys.path:
        sys.path.insert(0, wan_root)


@dataclass
class WanRFSDSContext:
    """Holds the frozen Wan2.2 components and a few invariants.

    Loaded once at Stage D init and passed by reference into every
    ``w_rfsds_loss`` call. All weights are frozen and on the same device.

    Note on ``sample_shift``: Wan's scheduler (fm_solvers_unipc.py:109-119)
    applies a shift transformation to the sigma schedule:
        sigma_shifted = shift * sigma / (1 + (shift - 1) * sigma)
    The W-RFSDS forward ``z_sigma = (1-sigma)*z + sigma*eps`` must use the
    SHIFTED sigma so the noisy latent matches Wan's training distribution.
    Default shift=5.0 for 480p I2V; see Stage A's ``sample_shift`` arg.
    """
    wan_vae: Any                 # Wan2_1_VAE; .encode([Tensor [3, T, H, W] in [-1,1]])
    low_noise_model: Any         # WanModel; tau < 0.9 expert
    high_noise_model: Any        # WanModel; tau >= 0.9 expert
    boundary_in_t_units: float   # = WAN_BOUNDARY_NORMALIZED * (num_train_timesteps)
    num_train_timesteps: int
    sample_shift: float          # ★ C2 fix: Wan scheduler shift parameter (default 5.0)
    device: torch.device
    dtype: torch.dtype


def load_wan_for_rfsds(
    wan_ckpt_dir: str,
    repo_root: str,
    device: torch.device,
    convert_model_dtype: bool = True,
    device_id: int = 0,
    sample_shift: float = 5.0,
) -> WanRFSDSContext:
    """Load the three Wan2.2 components we need for W-RFSDS.

    Uses the same ``wan.image2video.WanI2V`` class that Stage A uses, but
    keeps everything resident on GPU (no offload) and freezes all
    parameters. The T5 text encoder is NOT loaded here because the
    ``wan_cond_cached`` dict from Bootstrap already contains the encoded
    text embeddings (context / context_null) as tensors.

    Parameters
    ----------
    wan_ckpt_dir : str
        Local directory containing the Wan2.2-I2V-A14B weights (same as
        what Stage A uses).
    repo_root : str
        Absolute path to ``mine/`` (so we can locate ``mine/Wan2.2/``).
    device : torch.device
    convert_model_dtype : bool
        If True, casts the DiT and VAE to ``config.param_dtype`` (bf16 by
        default for A14B). Matches Stage A's setting.
    device_id : int
        CUDA device index (matches the ``device`` arg).
    """
    # Same offline guards as Stage A; must precede HF imports.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    _ensure_wan_on_sys_path(repo_root)

    from wan.configs.wan_i2v_A14B import i2v_A14B as wan_cfg
    from wan.image2video import WanI2V

    # Construct WanI2V; we'll only retain vae / low_noise_model / high_noise_model.
    i2v = WanI2V(
        config=wan_cfg,
        checkpoint_dir=wan_ckpt_dir,
        device_id=device_id,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=False,
        convert_model_dtype=convert_model_dtype,
        init_on_cpu=False,
    )

    vae = i2v.vae
    low_noise = i2v.low_noise_model
    high_noise = i2v.high_noise_model

    # Freeze everything.
    for m in (low_noise, high_noise):
        for p in m.parameters():
            p.requires_grad_(False)
        m.eval()

    # vae may be a Wan2_1_VAE wrapper holding ``vae.model`` + ``vae.scale``;
    # the inner nn.Module is what carries parameters.
    if hasattr(vae, "model"):
        for p in vae.model.parameters():
            p.requires_grad_(False)
        vae.model.eval()

    # Drop T5 text encoder reference (we use cached context tensors).
    if hasattr(i2v, "text_encoder"):
        del i2v.text_encoder
    if hasattr(i2v, "t5_cpu"):
        i2v.t5_cpu = True   # ensure no accidental .to(device) on encoder

    # ``boundary * num_train_timesteps`` gives the cutoff in [0, 1000) scale
    # (matches wan/image2video.py:341).
    num_train_ts = int(wan_cfg.num_train_timesteps)
    if num_train_ts != WAN_NUM_TRAIN_TIMESTEPS:
        raise RuntimeError(
            f"Wan config num_train_timesteps={num_train_ts} differs from "
            f"our WAN_NUM_TRAIN_TIMESTEPS={WAN_NUM_TRAIN_TIMESTEPS}; "
            "constants need to be re-checked."
        )
    boundary = float(getattr(wan_cfg, "boundary", WAN_BOUNDARY_NORMALIZED))
    if abs(boundary - WAN_BOUNDARY_NORMALIZED) > 1.0e-6:
        raise RuntimeError(
            f"Wan config boundary={boundary} differs from our "
            f"WAN_BOUNDARY_NORMALIZED={WAN_BOUNDARY_NORMALIZED}; "
            "constants need to be re-checked."
        )
    boundary_in_t_units = boundary * float(num_train_ts)

    # Param dtype (bf16 for A14B by default after convert_model_dtype).
    sample_param = next(low_noise.parameters())
    param_dtype = sample_param.dtype

    return WanRFSDSContext(
        wan_vae=vae,
        low_noise_model=low_noise,
        high_noise_model=high_noise,
        boundary_in_t_units=boundary_in_t_units,
        num_train_timesteps=num_train_ts,
        sample_shift=float(sample_shift),          # ★ C2 fix
        device=device,
        dtype=param_dtype,
    )


def _shifted_sigma(tau: float, shift: float) -> float:
    """Apply Wan scheduler's sigma shift transformation.

    Wan2.2 ``fm_solvers_unipc.py:109-119``:
        if not use_dynamic_shifting:
            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
    The model was trained on shifted sigmas, so the W-RFSDS forward sample
    ``z_sigma = (1-sigma)*z + sigma*eps`` must use the same shifted sigma
    to stay in-distribution. ``shift=1`` is the identity (no shift).
    """
    if shift == 1.0:
        return float(tau)
    return float(shift * tau / (1.0 + (shift - 1.0) * tau))


# =============================================================================
# Wan VAE encode helper ([-1, 1] normalization, list interface)
# =============================================================================

def _to_wan_vae_input(video_3FHW_float01: torch.Tensor) -> torch.Tensor:
    """Convert rendered RGB in [0, 1] to Wan VAE expected [-1, 1].

    Wan2.2's ``image2video.py:259`` documents this convention:
        ``img = TF.to_tensor(img).sub_(0.5).div_(0.5)``
    which is exactly ``x * 2 - 1`` after ``to_tensor`` (which divides by 255).
    """
    return video_3FHW_float01 * 2.0 - 1.0


def wan_vae_encode_grad(
    rgb_3FHW_float01: torch.Tensor,
    ctx: WanRFSDSContext,
) -> torch.Tensor:
    """Encode the rendered 21-frame video through Wan VAE WITH gradient.

    Wan VAE expects List[Tensor [3, T, H, W] in [-1, 1]] and returns
    List[Tensor [16, T_lat, H_lat, W_lat]] (one element per video).
    We strip the list and add the batch axis for downstream DiT input
    convenience.

    Parameters
    ----------
    rgb_3FHW_float01 : Tensor [3, F=21, H=464, W=832] in [0, 1]
        Direct render output from ``render_21_with_warp`` (permuted from
        ``[F, 3, H, W]`` to ``[3, F, H, W]``).
    ctx : WanRFSDSContext

    Returns
    -------
    z : Tensor [1, 16, F_lat=6, H_lat=58, W_lat=104]
        Grad-enabled VAE latent. ``z.requires_grad`` is True iff the input
        rgb has ``requires_grad`` (true for our differentiable renderer).
    """
    if rgb_3FHW_float01.shape != (3, F_FRAMES, H_PIXEL, W_PIXEL):
        raise ValueError(
            f"rgb_3FHW must be [3, {F_FRAMES}, {H_PIXEL}, {W_PIXEL}], "
            f"got {tuple(rgb_3FHW_float01.shape)}"
        )
    rgb_neg11 = _to_wan_vae_input(rgb_3FHW_float01).to(ctx.device)
    # Wan2_1_VAE.encode expects a Python list and returns a list.
    z_list = ctx.wan_vae.encode([rgb_neg11])
    if len(z_list) != 1:
        raise RuntimeError(
            f"Wan VAE encode returned {len(z_list)} entries; expected 1"
        )
    z = z_list[0]
    # Sanity-check the latent shape.
    expected = (WAN_LATENT_CH, F_LATENT, H_LATENT, W_LATENT)
    if tuple(z.shape) != expected:
        raise RuntimeError(
            f"Wan VAE latent shape mismatch: expected {expected}, "
            f"got {tuple(z.shape)}"
        )
    return z.unsqueeze(0)                              # [1, 16, F_lat, H_lat, W_lat]


# =============================================================================
# W-RFSDS loss
# =============================================================================

def _select_expert(
    tau: float,
    ctx: WanRFSDSContext,
) -> Any:
    """Return high_noise_model when ``tau * (num_train_timesteps - 1) >= boundary``.

    The comparison matches ``wan/image2video.py:341, 388-391`` which uses
    ``t.item() >= boundary`` with ``boundary = self.boundary *
    num_train_timesteps``.

    ★ Fix SD-1: scaling must match the t_wan tensor passed to the model
    forward (line ~367 below uses ``tau * (num_train_timesteps - 1)``).
    Without this fix, at tau=0.9 the selector saw ``0.9 * 1000 = 900 >= 900``
    (high expert) but the model received ``0.9 * 999 = 899.1`` which in Wan's
    scheduler corresponds to the low-expert regime — i.e. high_noise weights
    were applied to an input the low_noise expert should have seen. The
    practical effect was small (1 timestep boundary jitter) but spec demands
    one consistent scaling.
    """
    t_in_train_units = float(tau) * (float(ctx.num_train_timesteps) - 1.0)
    if t_in_train_units >= ctx.boundary_in_t_units:
        return ctx.high_noise_model
    return ctx.low_noise_model


def w_rfsds_loss(
    rgb_3FHW_float01: torch.Tensor,
    wan_cond: Dict[str, Any],
    ctx: WanRFSDSContext,
    tau: float,
    cfg_scale: float,
    seed: Optional[int] = None,
    return_z_theta: bool = False,
) -> "torch.Tensor | Tuple[torch.Tensor, torch.Tensor]":
    """Compute the W-RFSDS surrogate loss (CHORD Eq. 3) with Wan-shifted sigma.

    The loss is constructed so its gradient w.r.t. ``rgb_3FHW_float01``
    equals (in expectation over eps and tau) the W-RFSDS update direction.
    The two Wan DiT forwards run under ``torch.no_grad`` because the
    residual is detached before being multiplied by ``z``; only the VAE
    encode needs backward.

    ★ C2 fix: Wan's scheduler maps the integer timestep to a SHIFTED sigma
    (see ``_shifted_sigma``). The W-RFSDS forward ``z_sigma = (1-sigma)*z +
    sigma*eps`` must use the shifted sigma so the noisy latent is in the
    distribution Wan was trained on. The selector below uses the UN-shifted
    ``tau`` to pick the high/low-noise expert (matching Wan's boundary check
    on the raw timestep), and the shifted sigma for the noise mixture.

    Parameters
    ----------
    rgb_3FHW_float01 : Tensor [3, 21, 464, 832] in [0, 1]
        Differentiable render output (caller should already have clamped to
        [0, 1] to keep VAE input in-distribution).
    wan_cond : dict   from Bootstrap B11
    ctx : WanRFSDSContext  (includes ``sample_shift``)
    tau : float in (0, 1)  raw quantile from inverse-CDF
    cfg_scale : float       CFG (CHORD: linear 25 -> 12 over training)
    seed : int or None      deterministic eps for type vote
    return_z_theta : bool
        If True, returns ``(loss, z_theta)`` so the caller can reuse
        ``z_theta`` for ``latent_recon_loss`` without a second VAE encode
        (S5 fix; saves the most expensive backward pass per iter).

    Returns
    -------
    loss : Tensor scalar  (or (loss, z_theta) if return_z_theta=True)
    """
    # ---- 1) Encode rendered video to VAE latent (grad-enabled) ----
    z_theta = wan_vae_encode_grad(rgb_3FHW_float01, ctx)
    # z_theta: [1, 16, F_lat, H_lat, W_lat]

    # ---- 2) Forward sample z_sigma = (1-sigma)*z + sigma*eps  (★ C2) ----
    # CHORD section 3.2 + Wan scheduler shift: sigma_shifted is what Wan saw
    # at training time when its UniPC scheduler used the given timestep.
    sigma = _shifted_sigma(float(tau), ctx.sample_shift)
    with torch.no_grad():
        if seed is not None:
            gen = torch.Generator(device=z_theta.device).manual_seed(int(seed))
            eps = torch.randn(z_theta.shape, generator=gen,
                              device=z_theta.device, dtype=z_theta.dtype)
        else:
            eps = torch.randn_like(z_theta)
        z_sigma = (1.0 - sigma) * z_theta.detach() + sigma * eps

    # ---- 3) Pick expert by raw tau; build t_wan in [0, 1000) ----
    # The expert switch uses the integer timestep, NOT the shifted sigma.
    # Match wan/image2video.py:341,388-391 which compares timestep against
    # boundary = boundary_normalized * num_train_timesteps.
    wan_model = _select_expert(tau, ctx)
    t_wan = torch.tensor(
        [float(tau) * (float(ctx.num_train_timesteps) - 1.0)],
        device=ctx.device, dtype=torch.float32,
    )

    # ---- 4) Two DiT forwards (cond + uncond), both under no_grad ----
    x_input = [z_sigma.squeeze(0)]                     # List of [16, 6, 58, 104]
    with torch.no_grad():
        v_pred_cond = wan_model(
            x_input, t=t_wan,
            context=wan_cond["context"],
            seq_len=int(wan_cond["seq_len"]),
            y=wan_cond["y"],
        )[0].to(z_theta.dtype).unsqueeze(0)            # [1, 16, F_lat, H_lat, W_lat]

        v_pred_uncond = wan_model(
            x_input, t=t_wan,
            context=wan_cond["context_null"],
            seq_len=int(wan_cond["seq_len"]),
            y=wan_cond["y"],
        )[0].to(z_theta.dtype).unsqueeze(0)

        v_pred = v_pred_uncond + float(cfg_scale) * (v_pred_cond - v_pred_uncond)
        # CHORD Eq.3 residual; detached, treated as fixed target direction.
        residual = (v_pred - eps + z_theta.detach()).detach()

    # ---- 5) Form surrogate loss whose gradient w.r.t. theta equals SDS ----
    loss = (residual * z_theta).sum() / float(z_theta.numel())
    if return_z_theta:
        return loss, z_theta
    return loss


# =============================================================================
# Latent-reconstruction auxiliary loss (NOT in CHORD; kept for optional use)
# =============================================================================

def latent_recon_loss(
    rgb_3FHW_float01: torch.Tensor,
    z_wan_target: torch.Tensor,
    ctx: WanRFSDSContext,
    z_render_cached: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """L2 in Wan VAE latent space against Bootstrap's ``z_wan_target``.

    NOT in CHORD. method.md keeps this as an optional auxiliary loss
    (disabled by default in P1, enabled in P2). If enabled at large weight
    in P1, can bias optimization toward over-fitting to Wan video artifacts.

    ★ S5 fix: caller can pass ``z_render_cached`` from a prior
    ``w_rfsds_loss(..., return_z_theta=True)`` call to skip the (expensive)
    duplicate Wan VAE encode + backward. Halves the iter's peak VAE memory
    when both losses are active.

    Parameters
    ----------
    rgb_3FHW_float01 : Tensor [3, 21, 464, 832] in [0, 1]
    z_wan_target : Tensor [16, F_lat, H_lat, W_lat]
    ctx : WanRFSDSContext
    z_render_cached : Optional[Tensor [1, 16, F_lat, H_lat, W_lat]]
        Reuse of z from a prior w_rfsds_loss call this iter.

    Returns
    -------
    loss : Tensor scalar
    """
    if z_render_cached is not None:
        if z_render_cached.shape[0] != 1:
            raise ValueError(
                f"z_render_cached must have batch=1; got {tuple(z_render_cached.shape)}"
            )
        z_render = z_render_cached.squeeze(0)
    else:
        z_render = wan_vae_encode_grad(rgb_3FHW_float01, ctx).squeeze(0)
    if z_render.shape != z_wan_target.shape:
        raise ValueError(
            f"shape mismatch: z_render={tuple(z_render.shape)}, "
            f"z_wan_target={tuple(z_wan_target.shape)}"
        )
    return ((z_render - z_wan_target.detach()) ** 2).mean()


__all__ = [
    "WanRFSDSContext", "load_wan_for_rfsds",
    "wan_vae_encode_grad", "w_rfsds_loss", "latent_recon_loss",
]
