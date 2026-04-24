"""Stage B driver: Velocity-Guided Consensus Flow Stage-1 sampling.

Consumes K DINOv2 conditions (one per articulation state), runs the
VGCF sampler on TRELLIS Stage-1, decodes the per-state latents to 64^3
occupancy grids, and persists all intermediates under ``out_dir``.

The function does NOT instantiate the sampler itself; the caller is
expected to have already swapped ``pipe.sparse_structure_sampler`` to a
:class:`VGCFSampler`. This keeps the driver trivially compatible with
an ``enabled=false`` ablation.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from pipelines.utils.postprocessing import remove_disk
from pipelines.utils.voxel_io import save_voxel_grid
from pipelines.utils.voxel_viz import (
    save_diagnostics_curves_html,
    save_voxel_html,
    save_voxel_stack_html,
)


@dataclass
class VGCFResult:
    """Return payload of :func:`run_vgcf`.

    Attributes
    ----------
    O_stack : torch.Tensor
        Shape ``(K, 64, 64, 64)``, binary float (0/1) after threshold.
    O_stack_soft : torch.Tensor
        Shape ``(K, 64, 64, 64)``, post-sigmoid decoder output.
    z_final : torch.Tensor
        Shape ``(K, 8, 16, 16, 16)``, final Stage-1 latents.
    diagnostics : list
        Per-step logging dicts from the sampler.
    """
    O_stack: torch.Tensor
    O_stack_soft: torch.Tensor
    z_final: torch.Tensor
    diagnostics: List[Dict[str, Any]] = field(default_factory=list)


def run_vgcf(
    pipe: Any,
    cond: Dict[str, torch.Tensor],
    K: int,
    cfg_vgcf: Any,
    out_dir: str,
    device: Optional[str] = None,
    cfg_stage_b: Any = None,
) -> VGCFResult:
    """Run Stage B: VGCF Stage-1 sampling + VAE decode.

    Parameters
    ----------
    pipe : TrellisImageTo3DPipeline
        The TRELLIS pipeline. ``pipe.sparse_structure_sampler`` must already
        be a :class:`trellis.pipelines.samplers.VGCFSampler` instance (see
        ``run_v1.py``).
    cond : dict
        Output of ``pipe.get_cond(images)``; contains keys ``'cond'`` and
        ``'neg_cond'``, each shape ``(K, 1369, 768)``.
    K : int
        Number of articulation states.
    cfg_vgcf : OmegaConf / dict-like
        The ``vgcf:`` subtree of ``configs/v1.yaml``.
    out_dir : str
        Destination directory for persisted intermediates.
    device : str, optional
        Device to run on. Defaults to the device of ``cond['cond']``.

    Returns
    -------
    VGCFResult
    """
    os.makedirs(out_dir, exist_ok=True)
    if device is None:
        device = cond["cond"].device
    device = torch.device(device)

    flow_model = pipe.models["sparse_structure_flow_model"]
    decoder = pipe.models["sparse_structure_decoder"]

    # Shared-noise initialization across K tracks: preserves each track's
    # marginal exactly at t=1 and makes the coupled ODE well-defined.
    gen = torch.Generator(device=device).manual_seed(int(cfg_vgcf.seed))
    eps = torch.randn((1, 8, 16, 16, 16), device=device, generator=gen)
    noise = eps.repeat(K, 1, 1, 1, 1)

    # Merge sampler params. pipe.sparse_structure_sampler_params holds the
    # TRELLIS defaults; we override with v1 config values.
    sampler_params: Dict[str, Any] = dict(pipe.sparse_structure_sampler_params)
    sampler_params["steps"] = int(cfg_vgcf.steps)
    sampler_params["cfg_strength"] = float(cfg_vgcf.cfg_strength)
    sampler_params["cfg_interval"] = tuple(cfg_vgcf.cfg_interval)
    sampler_params["rescale_t"] = float(cfg_vgcf.rescale_t)

    # Sample K latents simultaneously.
    sampler = pipe.sparse_structure_sampler
    out = sampler.sample(
        flow_model,
        noise,
        cond=cond["cond"],
        neg_cond=cond["neg_cond"],
        verbose=True,
        **sampler_params,
    )
    z_final = out.samples  # (K, 8, 16, 16, 16)

    # Decode to 64^3 occupancy. The decoder emits logits; sigmoid is applied
    # here to keep both the soft and binary views for downstream stages.
    logits = decoder(z_final)                       # (K, 1, 64, 64, 64)
    soft = torch.sigmoid(logits).squeeze(1)          # (K, 64, 64, 64)
    binary = (soft > 0.5).to(soft.dtype)             # (K, 64, 64, 64)

    # Strip the grounding disk/carpet from the binary occupancy. The
    # postprocessing helper is a no-op when inputs do not contain a disk,
    # so it is safe to run unconditionally if the config flag is on.
    remove_disk_flag = True
    if cfg_stage_b is not None and hasattr(cfg_stage_b, "remove_disk"):
        remove_disk_flag = bool(cfg_stage_b.remove_disk)
    removed_voxels = 0
    if remove_disk_flag:
        binary_np = binary.detach().cpu().numpy()        # (K, 64, 64, 64)
        voxels_5d = binary_np[:, None].astype(np.float32)  # (K, 1, 64, 64, 64)
        before = float(voxels_5d.sum())
        voxels_5d = remove_disk(voxels_5d)               # in-place + return
        after = float(voxels_5d.sum())
        removed_voxels = int(before - after)
        binary = torch.from_numpy(voxels_5d[:, 0]).to(
            device=soft.device, dtype=soft.dtype,
        )
        # Align the soft mask so downstream diagnostics match the binary.
        soft = soft * binary

    # Persist intermediates.
    save_voxel_grid(os.path.join(out_dir, "O_stack.npy"),
                    binary.detach().cpu().numpy().astype(np.uint8))
    save_voxel_grid(os.path.join(out_dir, "O_stack_soft.npy"),
                    soft.detach().cpu().numpy().astype(np.float16))
    torch.save(z_final.detach().cpu(), os.path.join(out_dir, "z_final.pt"))

    diagnostics = list(
        getattr(out, "vgcf_diagnostics", None)
        or getattr(out, "bcac_diagnostics", None)
        or []
    )
    with open(os.path.join(out_dir, "vgcf_diagnostics.json"), "w") as f:
        json.dump(diagnostics, f, indent=2)

    # Stage B post-process summary.
    with open(os.path.join(out_dir, "postprocess.json"), "w") as f:
        json.dump({
            "remove_disk_applied": bool(remove_disk_flag),
            "voxels_removed_by_disk": int(removed_voxels),
        }, f, indent=2)

    # ---- Per-step Tweedie decode + cross-state consistency analysis ----
    # Decode each Euler step's Tweedie estimate (pred_x_0) through the VAE
    # decoder to see how 64^3 occupancy evolves and whether base is aligned.
    pred_x0_list = getattr(out, "pred_x_0", [])
    if pred_x0_list:
        step_dir = os.path.join(out_dir, "per_step")
        os.makedirs(step_dir, exist_ok=True)
        per_step_stats = []

        for step_idx, x0_hat in enumerate(pred_x0_list):
            # x0_hat: (K, 8, 16, 16, 16) — Tweedie clean estimate at this step
            with torch.no_grad():
                logits_step = decoder(x0_hat.to(device))        # (K, 1, 64, 64, 64)
                occ_step = (torch.sigmoid(logits_step).squeeze(1) > 0.5).float()  # (K, 64, 64, 64)

            occ_np = occ_step.detach().cpu().numpy()
            K_s = occ_np.shape[0]

            # 1) Per-state voxel count
            counts = [int(occ_np[k].sum()) for k in range(K_s)]

            # 2) Pairwise IoU
            ious = []
            for i in range(K_s):
                for j in range(i + 1, K_s):
                    inter = ((occ_np[i] > 0.5) & (occ_np[j] > 0.5)).sum()
                    union = ((occ_np[i] > 0.5) | (occ_np[j] > 0.5)).sum()
                    ious.append(float(inter / max(union, 1)))

            # 3) Always-occupied / ever-occupied ratio (base consistency proxy)
            all_occ = (occ_np > 0.5).all(axis=0)
            any_occ = (occ_np > 0.5).any(axis=0)
            always = int(all_occ.sum())
            ever = int(any_occ.sum())

            # 4) Cross-state std per voxel
            voxel_std = occ_np.std(axis=0)
            mean_std = float(voxel_std.mean())
            max_std = float(voxel_std.max())

            # 5) 16^3 latent-level cross-state variance (raw, before decode)
            x0_np = x0_hat.detach().cpu().float().numpy()       # (K, 8, 16, 16, 16)
            latent_var_per_voxel = x0_np.var(axis=0).sum(axis=0)  # (16, 16, 16)

            step_stat = {
                "step": step_idx,
                "t": float(diagnostics[step_idx].get("t", 0)) if step_idx < len(diagnostics) else None,
                "voxel_counts": counts,
                "pairwise_iou_mean": float(np.mean(ious)) if ious else 0.0,
                "pairwise_iou_min": float(np.min(ious)) if ious else 0.0,
                "pairwise_iou_max": float(np.max(ious)) if ious else 0.0,
                "always_occupied": always,
                "ever_occupied": ever,
                "always_ever_ratio": float(always / max(ever, 1)),
                "voxel_std_mean": mean_std,
                "latent_var_mean": float(latent_var_per_voxel.mean()),
                "latent_var_max": float(latent_var_per_voxel.max()),
            }
            per_step_stats.append(step_stat)

            # Save per-step voxels for HTML visualization
            np.save(os.path.join(step_dir, f"O_step_{step_idx:02d}.npy"),
                    occ_np.astype(np.uint8))

        with open(os.path.join(step_dir, "per_step_stats.json"), "w") as f:
            json.dump(per_step_stats, f, indent=2)

        # Per-step HTML: one file with dropdown over steps, showing state-0
        from pipelines.utils.voxel_viz import save_voxel_stack_html
        for step_idx in range(len(pred_x0_list)):
            occ_step_np = np.load(os.path.join(step_dir, f"O_step_{step_idx:02d}.npy")).astype(float)
            save_voxel_stack_html(
                occ_step_np,
                os.path.join(step_dir, f"O_step_{step_idx:02d}.html"),
                title=f"Step {step_idx} Tweedie decode",
            )

        # Summary curve: pairwise IoU over steps
        from pipelines.utils.voxel_viz import _PLOTLY_OK
        if _PLOTLY_OK:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                                subplot_titles=("Pairwise IoU", "Always/Ever Ratio", "Latent Variance"))
            steps = [s["step"] for s in per_step_stats]
            ts = [s.get("t", 0) or 0 for s in per_step_stats]

            fig.add_trace(go.Scatter(x=steps, y=[s["pairwise_iou_mean"] for s in per_step_stats],
                                     mode="lines+markers", name="IoU mean"), row=1, col=1)
            fig.add_trace(go.Scatter(x=steps, y=[s["pairwise_iou_min"] for s in per_step_stats],
                                     mode="lines", name="IoU min", line=dict(dash="dot")), row=1, col=1)

            fig.add_trace(go.Scatter(x=steps, y=[s["always_ever_ratio"] for s in per_step_stats],
                                     mode="lines+markers", name="always/ever"), row=2, col=1)

            fig.add_trace(go.Scatter(x=steps, y=[s["latent_var_mean"] for s in per_step_stats],
                                     mode="lines+markers", name="latent var mean"), row=3, col=1)
            fig.add_trace(go.Scatter(x=steps, y=[s["latent_var_max"] for s in per_step_stats],
                                     mode="lines", name="latent var max", line=dict(dash="dot")), row=3, col=1)

            fig.update_layout(title="Per-step cross-state consistency", height=750)
            fig.update_xaxes(title_text="step", row=3, col=1)
            html_path = os.path.join(step_dir, "per_step_curves.html")
            fig.write_html(html_path, include_plotlyjs="cdn")
    # ---- End per-step diagnostics ----

    # --- Visualization ---
    binary_np = binary.detach().cpu().numpy().astype(np.float32)
    viz_dir = os.path.join(out_dir, "viz")
    os.makedirs(viz_dir, exist_ok=True)
    # 1) Single HTML with a dropdown over the K states
    save_voxel_stack_html(
        binary_np,
        os.path.join(viz_dir, "O_stack.html"),
        title="Stage B VGCF: O_k",
    )
    # 2) Per-state HTML for deep inspection
    for k in range(binary_np.shape[0]):
        save_voxel_html(
            binary_np[k],
            os.path.join(viz_dir, f"O_k_{k:02d}.html"),
            title=f"O_{k}",
        )
    # 3) VGCF per-step diagnostics.
    # Layered so P0 keys (log_sigma2_median, tau_used, M_mean_active,
    # active_voxels) are visible alongside the legacy ones.
    save_diagnostics_curves_html(
        diagnostics,
        os.path.join(viz_dir, "vgcf_diagnostics.html"),
        title="VGCF per-step diagnostics",
        keys=(
            "lambda",
            "M_mean",
            "M_mean_active",
            "log_sigma2_median",
            "tau_used",
            "active_voxels",
            "sigma2_median",
            "sigma2_max",
        ),
    )

    return VGCFResult(
        O_stack=binary,
        O_stack_soft=soft,
        z_final=z_final,
        diagnostics=diagnostics,
    )
