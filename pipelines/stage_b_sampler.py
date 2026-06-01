"""Shared Stage B sampler construction.

Both the standalone Stage B CLI and Bootstrap must attach the same sampler to
the TRELLIS pipeline before calling ``run_scar``. Keeping this in one place
prevents Bootstrap from silently using the default TRELLIS sampler.
"""

from __future__ import annotations

from typing import Any, Optional


def attach_stage_b_sampler(
    pipe: Any,
    cfg: Any,
    total_steps: Optional[int] = None,
    force_sampler: Optional[str] = None,
    mix_space: Optional[str] = None,
) -> str:
    """Attach the configured Stage B sampler to ``pipe`` and return its name."""
    from trellis.pipelines.samplers import (
        BCACSampler,
        SCARSampler,
        VGCFSampler,
        generate_alpha_schedule,
    )

    if total_steps is None:
        sampler_params = dict(pipe.sparse_structure_sampler_params)
        total_steps = int(sampler_params.get("steps", 25))

    sampler_choice = (
        str(force_sampler).lower()
        if force_sampler is not None
        else str(cfg.get("stage_b", {}).get("sampler", "scar")).lower()
    )
    sigma_min = pipe.sparse_structure_sampler.sigma_min

    if sampler_choice == "scar":
        scar_cfg = cfg.get("scar", {}) or {}
        if "alpha_schedule" in scar_cfg and scar_cfg["alpha_schedule"] is not None:
            alpha_schedule = tuple(scar_cfg["alpha_schedule"])
        else:
            alpha_schedule = tuple(generate_alpha_schedule(
                peak=float(scar_cfg.get("alpha_peak", 0.0)),
                total_steps=int(total_steps),
                decay=str(scar_cfg.get("alpha_decay", "quadratic")),
            ))
        resolved_mix_space = (
            str(mix_space)
            if mix_space is not None
            else str(scar_cfg.get("mix_space", "z_t"))
        )
        pipe.sparse_structure_sampler = SCARSampler(
            sigma_min=sigma_min,
            alpha_schedule=alpha_schedule,
            active_fraction=float(scar_cfg.get("active_fraction", 0.1)),
            tau_percentile=float(scar_cfg.get("tau_percentile", 0.65)),
            eps_log=float(scar_cfg.get("eps_log", 1.0e-6)),
            eta=float(scar_cfg.get("eta", 0.5)),
            mix_steps=int(scar_cfg.get("mix_steps", 8)),
            mix_weights=tuple(scar_cfg.get("mix_weights", [0.3, 0.4, 0.3])),
            extreme_mix_mode=str(scar_cfg.get("extreme_mix_mode", "symmetric")),
            w_floor=float(scar_cfg.get("w_floor", 0.0)),
            scar_enabled=True,
            mix_space=resolved_mix_space,
        )
    elif sampler_choice == "bcac":
        bcac_cfg = cfg.get("bcac", {}) or {}
        pipe.sparse_structure_sampler = BCACSampler(
            sigma_min=sigma_min,
            block_start=int(bcac_cfg.get("block_start", 4)),
            block_end=int(bcac_cfg.get("block_end", 9)),
            t_full=float(bcac_cfg.get("t_full", 0.7)),
            t_release=float(bcac_cfg.get("t_release", 0.3)),
            alpha_max=float(bcac_cfg.get("alpha_max", 1.0)),
            tau_percentile=float(bcac_cfg.get("tau_percentile", 0.65)),
            active_fraction=float(bcac_cfg.get("active_fraction", 0.8)),
            eps_log=float(bcac_cfg.get("eps_log", 1e-6)),
            eta=float(bcac_cfg.get("eta", 0.5)),
        )
    elif sampler_choice == "vgcf":
        vgcf_cfg = cfg.get("vgcf", {}) or {}
        pipe.sparse_structure_sampler = VGCFSampler(
            sigma_min=sigma_min,
            lambda_max=float(vgcf_cfg.get("lambda_max", 1.0)),
            t_stop=float(vgcf_cfg.get("t_stop", 0.2)),
            eta=float(vgcf_cfg.get("eta", 0.5)),
            vgcf_enabled=bool(vgcf_cfg.get("enabled", True)),
            active_fraction=float(vgcf_cfg.get("active_fraction", 0.8)),
            tau_percentile=float(vgcf_cfg.get("tau_percentile", 0.65)),
            eps_log=float(vgcf_cfg.get("eps_log", 1.0e-6)),
            lambda_schedule=str(vgcf_cfg.get("lambda_schedule", "warmup")),
        )
    else:
        raise ValueError(f"unknown Stage B sampler: {sampler_choice!r}")

    return sampler_choice


__all__ = ["attach_stage_b_sampler"]
