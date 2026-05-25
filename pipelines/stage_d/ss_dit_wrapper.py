"""Composition wrapper that turns the frozen TRELLIS SparseStructureFlowModel
into a *grad-enabled one-step structural refiner* (method.md innovation 2).

The wrapper:
  1. Faithfully replicates ``SparseStructureFlowModel.forward`` (TRELLIS
     trellis/models/sparse_structure_flow.py:176-208), so its output matches
     the frozen backbone bit-for-bit when adapters are zero-initialized.
  2. Injects a zero-init residual adapter after each of blocks {14, 16, 18}.
     ``h = h + adapter_k(h)`` is the formulation; with adapter weights all
     zero at init, the wrapper is identical to the frozen model.
  3. Captures the post-adapter hidden state at the same blocks so downstream
     heads (H_sup / H_part / H_joint via sample_hidden_at_U) can read them.

Timestep convention: ``forward_capture`` accepts ``t_raw`` in [0, 1] (raw
rectified-flow time) and multiplies by 1000 internally before calling the
TRELLIS ``TimestepEmbedder`` (matches trellis/pipelines/samplers/flow_euler.py
line 38-42). TRELLIS ``share_mod`` is False for image-large (no global adaLN);
each block has its own ``adaLN_modulation``.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# The TRELLIS spatial helpers; available because run_v1.py inserts the TRELLIS
# package on sys.path before any stage import.
from trellis.modules.spatial import patchify, unpatchify

from .config import (
    TRELLIS_LATENT_GRID,
    TRELLIS_SS_IN_CH,
    TRELLIS_SS_MODEL_CH,
    TRELLIS_SS_NUM_BLOCKS,
    WAN_NUM_TRAIN_TIMESTEPS,
)


class SS_DiT_WithAdapters(nn.Module):
    """Frozen SS-DiT + learnable adapters + hidden capture.

    Parameters
    ----------
    base_ss_dit : SparseStructureFlowModel
        The frozen TRELLIS SS-DiT (e.g. obtained from
        ``TrellisImageTo3DPipeline.sparse_structure_flow_model``).
    adapters : nn.ModuleDict
        Dict mapping block index (as ``str``) to a ``ZeroInitAdapter``.
        Ownership stays with the caller (typically
        ``StageDLearnable.adapters``); this wrapper holds a *reference* so
        gradients and checkpoint state are routed correctly.
    capture_blocks : tuple of int
        Block indices whose post-adapter hidden state is returned in
        ``captured``. Must be the same set as ``adapters.keys()``.
    """

    def __init__(self,
                 base_ss_dit: nn.Module,
                 adapters: nn.ModuleDict,
                 capture_blocks: Tuple[int, ...] = (14, 16, 18)) -> None:
        super().__init__()
        self.base = base_ss_dit
        # Hard-freeze the backbone (no checkpointing parameters needed).
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.base.eval()  # disable any training-only behaviour (dropout, etc.)

        # Validate adapter / capture block consistency.
        adapter_keys = sorted(int(k) for k in adapters.keys())
        capture_sorted = sorted(int(b) for b in capture_blocks)
        if adapter_keys != capture_sorted:
            raise ValueError(
                f"adapter block set {adapter_keys} must match capture_blocks "
                f"{capture_sorted}"
            )
        for b in capture_sorted:
            if not (0 <= b < TRELLIS_SS_NUM_BLOCKS):
                raise ValueError(
                    f"capture block index {b} out of range "
                    f"[0, {TRELLIS_SS_NUM_BLOCKS})"
                )
        self.adapters = adapters
        self.capture_blocks = tuple(capture_sorted)

        # Read backbone meta needed inside forward (avoid attribute lookup
        # in the hot loop).
        self._resolution: int = int(self.base.resolution)
        self._patch_size: int = int(self.base.patch_size)
        self._dtype: torch.dtype = self.base.dtype
        if self.base.share_mod:
            raise NotImplementedError(
                "share_mod=True is not expected for TRELLIS-image-large "
                "(see ss_flow_img_dit_L_16l8_fp16.json). This wrapper "
                "implements share_mod=False only."
            )

    # ------------------------------------------------------------------ #
    # The forward replicates trellis/models/sparse_structure_flow.py     #
    # forward() lines 176-208, with adapter injection + capture after    #
    # blocks listed in capture_blocks. See method.md section 7.2.        #
    # ------------------------------------------------------------------ #
    def forward_capture(
        self,
        x: torch.Tensor,
        t_raw: float,
        cond: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        """Run a single grad-enabled SS-DiT forward and capture hidden states.

        Parameters
        ----------
        x : torch.Tensor   shape [B, in_channels=8, 16, 16, 16]
            Noisy SS-VAE latent ``z_t`` (raw fp32, gradient flows back into
            ``z_s_base = z_s0 + Delta_z_s``).
        t_raw : float
            Rectified-flow time in [0, 1]. Multiplied by 1000 internally
            before being passed to ``base.t_embedder`` (TRELLIS convention).
        cond : torch.Tensor   shape [B, N_dino, 1024]
            DINOv2 patch embeddings (the cached ``trellis_cond_can`` from
            Bootstrap, which is ``DINOv2(s_0_with_carpet)``).

        Returns
        -------
        pred_v : torch.Tensor   shape [B, out_channels=8, 16, 16, 16]
            Predicted velocity (RF target). We don't actually consume
            ``pred_v`` in the loss (we use Wan2.2's velocity prediction via
            W-RFSDS), but emitting it makes the wrapper a drop-in for
            sanity tests against the frozen backbone.
        captured : dict { block_idx -> Tensor [B, 4096, 1024] }
            Post-adapter hidden state at each listed block. Used by
            ``feature_sample.sample_hidden_at_U`` to build per-voxel
            features.
        """
        if x.shape[1] != TRELLIS_SS_IN_CH:
            raise ValueError(
                f"SS-DiT expects in_channels={TRELLIS_SS_IN_CH}, "
                f"got shape {tuple(x.shape)}"
            )
        if x.shape[2:] != (TRELLIS_LATENT_GRID,) * 3:
            raise ValueError(
                f"SS-DiT expects spatial grid {TRELLIS_LATENT_GRID}^3, "
                f"got shape {tuple(x.shape)}"
            )
        B = x.shape[0]

        # Match TRELLIS convention: model receives t in [0, 1000).
        t_model = torch.full(
            (B,), float(WAN_NUM_TRAIN_TIMESTEPS) * float(t_raw),
            device=x.device, dtype=torch.float32,
        )

        # 1) Patchify (patch_size=1 -> shape-preserving) and flatten.
        h = patchify(x, self._patch_size)
        # h shape after patchify: [B, in_ch * patch^3, R, R, R] = [B, 8, 16, 16, 16]
        h = h.view(*h.shape[:2], -1).permute(0, 2, 1).contiguous()
        # h shape: [B, 4096, 8]

        # 2) Input projection 8 -> model_channels=1024
        h = self.base.input_layer(h)
        # h shape: [B, 4096, 1024]

        # 3) Absolute positional embedding (pe_mode='ape' for image-large)
        h = h + self.base.pos_emb[None]

        # 4) Timestep embedding
        t_emb = self.base.t_embedder(t_model)        # [B, 1024]
        # share_mod=False for image-large -> blocks own their own adaLN.

        # 5) Dtype cast to backbone dtype (fp16 for image-large)
        t_emb = t_emb.type(self._dtype)
        h = h.type(self._dtype)
        cond = cond.type(self._dtype)

        # 6) Transformer blocks with adapter + capture injection
        captured: Dict[int, torch.Tensor] = {}
        for k, block in enumerate(self.base.blocks):
            h = block(h, t_emb, cond)
            if str(k) in self.adapters:
                # Adapter is fp32 by default; cast its input/output to match h dtype.
                adapter = self.adapters[str(k)]
                adapter_out = adapter(h.to(torch.float32)).to(h.dtype)
                h = h + adapter_out
                # Capture POST-adapter hidden state. Detach not used:
                # downstream heads need gradient through here.
                captured[k] = h

        # 7) Cast back, final layer_norm, output projection, unpatchify
        h = h.type(x.dtype)
        h = F.layer_norm(h, h.shape[-1:])
        h = self.base.out_layer(h)
        # h shape: [B, 4096, out_ch * patch^3] = [B, 4096, 8]
        h = h.permute(0, 2, 1).view(
            B, h.shape[-1],
            self._resolution // self._patch_size,
            self._resolution // self._patch_size,
            self._resolution // self._patch_size,
        )
        pred_v = unpatchify(h, self._patch_size).contiguous()
        return pred_v, captured


__all__ = ["SS_DiT_WithAdapters"]
