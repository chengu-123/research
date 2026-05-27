"""Stage C IO contract.

Single source of truth for what Stage C joint init produces. Stage C is
defined in method.md section 6 (Bootstrap step B6) as a one-shot helper
that consumes Stage B Bootstrap intermediates and produces the joint
initialization (psi_0, phi_0, anchors_object) that Stage D W-RFSDS
optimization warm-starts from.

This file holds the dataclasses; the algorithms live in sibling modules
(move_geometry / joint_type_detect / axis_fit / anchor_extract / phi_fit).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Inputs (consumed from Stage B v3.3.6)
# ---------------------------------------------------------------------------


@dataclass
class StageCInputs:
    """All Stage B Bootstrap intermediates that Stage C consumes.

    Per method.md B6 the spec input list is (z_final, M_attn_boot_64,
    O_init, is_carpet_mask, U_seed). v3.3.6 Stage B produces strictly more
    signals; Stage C uses the richer ones when available.
    """

    # Spec inputs (method.md B6) ------------------------------------------------
    z_final: torch.Tensor                    # (K, 8, 16, 16, 16) Pass-1 SS latent
    M_attn_boot_64: torch.Tensor             # (64, 64, 64) cross-state semantic agreement
    O_init: torch.Tensor                     # (1, 1, 64, 64, 64) sigmoid(decoder(z_s0)) soft occ
    is_carpet_mask: torch.Tensor             # (64^3,) bool, FreeArt3D carpet voxels (flat)
    U_seed: torch.Tensor                     # (N_seed, 3) int32 voxel coords (seed support)

    # v3.3.6 richer signals --------------------------------------------------
    # Hard primary masks (after Pass-2 K-share + post-processing)
    O_base_canonical: Optional[torch.Tensor] = None      # (64, 64, 64) uint8/bool
    O_move_per_state: Optional[torch.Tensor] = None      # (K, 64, 64, 64) uint8/bool

    # Soft secondary fields (for graded analysis)
    P_base_canonical: Optional[torch.Tensor] = None             # (64, 64, 64) float
    P_move_evidence_per_state: Optional[torch.Tensor] = None    # (K, 64, 64, 64) float

    # Motion corridor (Pass-1 swept volume, footprint * (1 - shared))
    M_motion_corridor_64: Optional[torch.Tensor] = None  # (64, 64, 64) float

    # Optional dit_hidden (block 14/16/18 averaged), if Stage B captured
    dit_hidden: Optional[Dict[int, torch.Tensor]] = None  # {block: (K, 4096, 1024)} fp16

    def device(self) -> torch.device:
        return self.z_final.device

    def K(self) -> int:
        return int(self.z_final.shape[0])

    def has_v336_signals(self) -> bool:
        """Whether the v3.3.6 enriched primary outputs are available."""
        return (
            self.O_base_canonical is not None
            and self.O_move_per_state is not None
            and self.M_motion_corridor_64 is not None
        )


# ---------------------------------------------------------------------------
# Outputs (delivered to Bootstrap B7 + Stage D)
# ---------------------------------------------------------------------------


@dataclass
class Psi:
    """Initial joint parameters in unpacked form.

    19-element packed encoding follows method.md section 4.3 H_joint contract:
    [axis(3), origin(3), type_logit(1), theta_limit_raw(1), disp_limit_raw(1),
     delta_u(5), reserve(5)] = 19.

    All values are in WORLD space ([-0.5, 0.5] cube), per method.md voxel-to-
    world convention (`voxel_to_world(u, res=64) = (u + 0.5) / 64 - 0.5`).
    """

    axis: torch.Tensor             # (3,) unit vector
    origin: torch.Tensor           # (3,) world space, on the axis line for revolute
    type_logit: float              # sigmoid(type_logit) = P(prismatic); negative = revolute-leaning
    theta_limit_raw: float         # softplus(theta_limit_raw) = revolute max angle (rad)
    disp_limit_raw: float          # softplus(disp_limit_raw) = prismatic max displacement (world unit)
    delta_u_init: torch.Tensor     # (5,) softplus-positive increments such that
                                   #     u_raw = cumsum([0, delta_u_init]) recovers
                                   #     a monotone progression matching the data

    def pack_19(self) -> torch.Tensor:
        """Pack into the 19-element vector that matches H_joint output ordering.

        Reserve slots (last 5) are zero-filled.
        """
        device = self.axis.device
        dtype = self.axis.dtype
        out = torch.zeros(19, device=device, dtype=dtype)
        out[0:3] = self.axis
        out[3:6] = self.origin
        out[6] = float(self.type_logit)
        out[7] = float(self.theta_limit_raw)
        out[8] = float(self.disp_limit_raw)
        out[9:14] = self.delta_u_init
        # out[14:19] reserve, stays zero.
        return out


@dataclass
class JointInit:
    """Full output of Stage C joint init.

    Returned by `pipelines.stage_c.run_stage_c_init.run_stage_c_joint_init`.
    Persisted into Bootstrap artifacts (`psi_0.json`, `phi_0.npy`,
    `anchors_object.npy`) for Stage D consumption.

    v3 (cardinal-cand + voxel-scoring) addition: `secondary` field holds the
    OTHER type's best candidate (e.g., if primary is prismatic, secondary is
    revolute). Stage D's dual-clone can use both inits as warm-starts when
    the type margin is small, instead of cloning the same state.
    Set to None when not applicable (e.g., when running unit tests without
    dual-branch evaluation).
    """

    psi: Psi                                # initial joint parameters (unpacked)
    phi_0: torch.Tensor                     # (K,) per-state canonical-shifted progress
                                            # (phi_0[c]=0, c=canonical_state_idx)
    anchors_object: torch.Tensor            # (N_a, 3) int32 voxel coords (NOT world)
    confidence: float                       # in [0, 1] overall init confidence

    # Diagnostic per-stage sub-confidences (for debugging / paper)
    sub_confidence: Dict[str, float] = field(default_factory=dict)

    # Diagnostic intermediate quantities (move centroids, fit residuals etc.)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    # ★ v3 addition: alternate-type candidate for Stage D dual-clone init
    secondary: Optional["JointInit"] = None

    def joint_type(self) -> str:
        """Hard-classified joint type from psi.type_logit."""
        return "prismatic" if self.psi.type_logit >= 0.0 else "revolute"

    def to_dict_serialisable(self) -> Dict[str, Any]:
        """For JSON dump (numpy/torch -> python list/float)."""
        out: Dict[str, Any] = {
            "joint_type": self.joint_type(),
            "psi": {
                "axis": self.psi.axis.detach().cpu().tolist(),
                "origin": self.psi.origin.detach().cpu().tolist(),
                "type_logit": float(self.psi.type_logit),
                "theta_limit_raw": float(self.psi.theta_limit_raw),
                "disp_limit_raw": float(self.psi.disp_limit_raw),
                "delta_u_init": self.psi.delta_u_init.detach().cpu().tolist(),
            },
            "phi_0": self.phi_0.detach().cpu().tolist(),
            "anchors_object_count": int(self.anchors_object.shape[0]),
            "confidence": float(self.confidence),
            "sub_confidence": {k: float(v) for k, v in self.sub_confidence.items()},
        }
        # diagnostics: only include scalars; tensors/arrays summarised
        diag_safe: Dict[str, Any] = {}
        for k, v in self.diagnostics.items():
            if isinstance(v, (int, float, str, bool)):
                diag_safe[k] = v
            elif isinstance(v, (list, tuple)) and all(
                isinstance(x, (int, float)) for x in v
            ):
                diag_safe[k] = list(v)
            else:
                diag_safe[k] = f"<{type(v).__name__}>"
        out["diagnostics_scalars"] = diag_safe
        # Embed secondary candidate recursively for downstream consumers
        if self.secondary is not None:
            out["secondary"] = self.secondary.to_dict_serialisable()
        return out
