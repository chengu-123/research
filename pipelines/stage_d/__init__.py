"""Stage D package.

The package import path is intentionally lightweight. Heavy renderer and model
dependencies such as diff_gauss are imported only by the submodules that run
Stage D, not by ``import pipelines.stage_d`` or CLI help.
"""
from __future__ import annotations

from .config import (
    CANONICAL_STATE_IDX,
    F_FRAMES,
    F_LATENT,
    H_LATENT,
    H_PIXEL,
    K_STATES,
    STATE_INDICES,
    StageDConfig,
    W_LATENT,
    W_PIXEL,
)


def __getattr__(name: str):
    if name == "StageDLearnable":
        from .learnable import StageDLearnable
        return StageDLearnable
    if name == "StageDCameraConfig":
        from .render import StageDCameraConfig
        return StageDCameraConfig
    if name in {"load_bootstrap_bundle", "load_trellis_modules", "run_stage_d_main"}:
        from .run_stage_d import (
            load_bootstrap_bundle,
            load_trellis_modules,
            run_stage_d_main,
        )
        return {
            "load_bootstrap_bundle": load_bootstrap_bundle,
            "load_trellis_modules": load_trellis_modules,
            "run_stage_d_main": run_stage_d_main,
        }[name]
    if name in {"BootstrapBundle", "TrellisModules", "train_stage_d_p1"}:
        from .train import BootstrapBundle, TrellisModules, train_stage_d_p1
        return {
            "BootstrapBundle": BootstrapBundle,
            "TrellisModules": TrellisModules,
            "train_stage_d_p1": train_stage_d_p1,
        }[name]
    if name == "U_ExpandRequired":
        from .silhouette_check import U_ExpandRequired
        return U_ExpandRequired
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "StageDConfig", "StageDCameraConfig",
    "CANONICAL_STATE_IDX",
    "F_FRAMES", "F_LATENT", "H_LATENT", "W_LATENT", "H_PIXEL", "W_PIXEL",
    "K_STATES", "STATE_INDICES",
    "StageDLearnable", "BootstrapBundle", "TrellisModules",
    "run_stage_d_main", "train_stage_d_p1",
    "load_bootstrap_bundle", "load_trellis_modules",
    "U_ExpandRequired",
]
