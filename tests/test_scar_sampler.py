import sys
import os

import numpy as np
import torch
import torch.nn as nn
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "TRELLIS"))

from trellis.pipelines.samplers.scar import SCARSampler


class MockFlowModel(nn.Module):
    """A minimal flow model that returns noise-shaped random velocities.

    The output is shaped like a TRELLIS Stage-1 DiT: (B, 8, 16, 16, 16),
    where B is the number of latents in the batch.
    """

    def __init__(self, seed: int = 0):
        super().__init__()
        self.seed = seed
        self.resolution = 16
        self.patch_size = 2

    def forward(self, x, t, cond=None, **kwargs):
        # `_inference_model` in flow_euler.py wraps t as a (B,) float tensor
        # with values 1000*t_float; extract a scalar for seeding.
        if torch.is_tensor(t):
            t_val = float(t.flatten()[0].item())
        else:
            t_val = float(t)
        g = torch.Generator(device=x.device).manual_seed(
            self.seed + int(t_val) + int(cond.float().mean().item() * 100)
        )
        return torch.randn(x.shape, device=x.device, generator=g)


def test_sampler_returns_correct_shapes():
    """SCARSampler.sample() must return K latents of correct shape."""
    K = 3
    device = "cpu"
    noise = torch.randn(K, 8, 16, 16, 16, device=device)
    cond = torch.randn(K, 1374, 1024, device=device)
    neg_cond = torch.zeros_like(cond)

    model = MockFlowModel(seed=42)
    sampler = SCARSampler(sigma_min=0.0)
    out = sampler.sample(
        model, noise,
        cond=cond, neg_cond=neg_cond,
        steps=12, cfg_strength=7.5, cfg_interval=(0.0, 1.0),
        verbose=False,
    )
    assert out.samples.shape == (K, 8, 16, 16, 16)
    assert len(out.pred_x_0) == 12
    assert len(out.scar_diagnostics) == 12


def test_sampler_disabled_equals_flow_euler_baseline():
    """With scar_enabled=False, SCARSampler == FlowEulerGuidanceIntervalSampler."""
    from trellis.pipelines.samplers.flow_euler import FlowEulerGuidanceIntervalSampler
    K = 2
    device = "cpu"
    torch.manual_seed(123)
    noise = torch.randn(K, 8, 16, 16, 16, device=device)
    cond = torch.randn(K, 1374, 1024, device=device)
    neg_cond = torch.zeros_like(cond)

    model_a = MockFlowModel(seed=0)
    model_b = MockFlowModel(seed=0)

    baseline = FlowEulerGuidanceIntervalSampler(sigma_min=0.0)
    scar = SCARSampler(sigma_min=0.0, scar_enabled=False)

    out_a = baseline.sample(model_a, noise, cond=cond, neg_cond=neg_cond,
                            steps=12, cfg_strength=7.5, cfg_interval=(0.0, 1.0),
                            verbose=False)
    out_b = scar.sample(model_b, noise, cond=cond, neg_cond=neg_cond,
                        steps=12, cfg_strength=7.5, cfg_interval=(0.0, 1.0),
                        verbose=False)
    torch.testing.assert_close(out_a.samples, out_b.samples)


def test_sampler_active_steps_have_mask_diagnostic():
    """With mix_steps=0 (legacy mode): push is active in steps 0..3; later plain."""
    K = 3
    device = "cpu"
    noise = torch.randn(K, 8, 16, 16, 16, device=device)
    cond = torch.randn(K, 1374, 1024, device=device)
    neg_cond = torch.zeros_like(cond)
    model = MockFlowModel(seed=0)
    sampler = SCARSampler(sigma_min=0.0)   # default mix_steps=0
    out = sampler.sample(model, noise, cond=cond, neg_cond=neg_cond,
                         steps=12, cfg_strength=7.5, cfg_interval=(0.0, 1.0),
                         verbose=False)
    for s in range(4):
        assert "M_mean" in out.scar_diagnostics[s], f"step {s} missing M_mean"
        assert "alpha" in out.scar_diagnostics[s]
        assert out.scar_diagnostics[s]["alpha"] > 0
        assert out.scar_diagnostics[s]["mixed"] is False  # no mixing when mix_steps=0
    for s in range(4, 12):
        assert out.scar_diagnostics[s]["alpha"] == 0.0, \
            f"step {s} should have alpha=0 but got {out.scar_diagnostics[s]['alpha']}"


def test_sampler_two_phase_mode_shifts_push_after_mix():
    """mix_steps=4 + alpha_schedule indexed DIRECTLY by step: mix at 0-3,
    push at 0-7 (overlaps mix in 0-3 and continues alone in 4-7), plain 8-11.

    With new semantic (alpha_schedule indexed directly, no mix offset),
    alpha_schedule=[0.7, 1.0, 1.0, 0.5] applied at step indices 0..3
    means push is active in the mix region by design — mask is most
    discriminative at step 0.
    """
    K = 3
    device = "cpu"
    noise = torch.randn(K, 8, 16, 16, 16, device=device)
    cond = torch.randn(K, 1374, 1024, device=device)
    neg_cond = torch.zeros_like(cond)
    model = MockFlowModel(seed=0)
    sampler = SCARSampler(sigma_min=0.0, mix_steps=4,
                          mix_weights=(0.3, 0.4, 0.3))
    out = sampler.sample(model, noise, cond=cond, neg_cond=neg_cond,
                         steps=12, cfg_strength=7.5, cfg_interval=(0.0, 1.0),
                         verbose=False)
    # Steps 0-3: mix active AND push active (both during early phase)
    expected_alphas = [0.7, 1.0, 1.0, 0.5]
    for s, expected_a in zip(range(4), expected_alphas):
        assert out.scar_diagnostics[s]["mixed"] is True, \
            f"step {s} should be mixed but got mixed={out.scar_diagnostics[s]['mixed']}"
        assert abs(out.scar_diagnostics[s]["alpha"] - expected_a) < 1e-6, \
            f"step {s}: alpha={out.scar_diagnostics[s]['alpha']}, expected {expected_a}"
    # Steps 4-11: no mix, no push (alpha_schedule length=4 exhausted)
    for s in range(4, 12):
        assert out.scar_diagnostics[s]["mixed"] is False
        assert out.scar_diagnostics[s]["alpha"] == 0.0


def test_sampler_mixing_reduces_cross_state_divergence():
    """After mix phase, cross-state variance of x_t should be lower than without mix."""
    K = 3
    device = "cpu"
    torch.manual_seed(0)
    noise = torch.randn(K, 8, 16, 16, 16, device=device)
    # Different conds cause per-state divergence if left alone
    cond = torch.randn(K, 1374, 1024, device=device) * torch.tensor(
        [[1.0], [2.0], [3.0]], device=device
    ).unsqueeze(-1)
    neg_cond = torch.zeros_like(cond)

    # Without mixing
    model_a = MockFlowModel(seed=0)
    sampler_plain = SCARSampler(sigma_min=0.0, mix_steps=0, scar_enabled=False)
    out_plain = sampler_plain.sample(model_a, noise, cond=cond, neg_cond=neg_cond,
                                     steps=12, cfg_strength=7.5, cfg_interval=(0.0, 1.0),
                                     verbose=False)
    # With mixing (4 early steps)
    model_b = MockFlowModel(seed=0)
    sampler_mix = SCARSampler(sigma_min=0.0, mix_steps=4,
                               mix_weights=(0.3, 0.4, 0.3),
                               alpha_schedule=(0.0, 0.0, 0.0, 0.0))  # disable push
    out_mix = sampler_mix.sample(model_b, noise, cond=cond, neg_cond=neg_cond,
                                  steps=12, cfg_strength=7.5, cfg_interval=(0.0, 1.0),
                                  verbose=False)
    # Cross-state variance of final latent: mix should give lower variance
    var_plain = out_plain.samples.var(dim=0).mean().item()
    var_mix = out_mix.samples.var(dim=0).mean().item()
    assert var_mix < var_plain, \
        f"mixing should reduce cross-state variance: plain={var_plain:.4f}, mix={var_mix:.4f}"


def test_sampler_extreme_mix_mode_mean_of_middles():
    """extreme_mix_mode='mean_of_middles' makes ALL K states receive the
    same uniform mixed latent: 0.3*s0 + 0.4*mean(s1..s_{K-2}) + 0.3*s_{K-1}."""
    K = 6
    device = "cpu"
    torch.manual_seed(0)
    # Construct a simple (K, C=1, 1, 1, 1) tensor so we can easily reason.
    sample = torch.arange(K, dtype=torch.float32).reshape(K, 1, 1, 1, 1)
    # state k latent = k (just a scalar for verification)

    sampler = SCARSampler(
        sigma_min=0.0, mix_steps=1,
        mix_weights=(0.3, 0.4, 0.3),
        extreme_mix_mode="mean_of_middles",
    )
    mixed, was_mixed = sampler._mix_x_t(sample, step_idx=0)
    assert was_mixed

    # Expected: uniform formula for ALL states
    #   middle_mean = mean(sample[1..4]) = (1+2+3+4)/4 = 2.5
    #   For every k: 0.3*0 + 0.4*2.5 + 0.3*5 = 0 + 1.0 + 1.5 = 2.5
    # All K=6 states should get value 2.5
    expected = torch.full((K, 1, 1, 1, 1), 2.5)
    torch.testing.assert_close(mixed, expected, atol=1e-6, rtol=0)

    # Post-condition: after this mix, all K states are IDENTICAL
    for k in range(1, K):
        torch.testing.assert_close(mixed[k], mixed[0], atol=1e-6, rtol=0)


def test_sampler_extreme_mix_mode_symmetric_matches_legacy():
    """extreme_mix_mode='symmetric' should match original behavior (k=0 and
    K-1 keep 70% self + 30% other extreme)."""
    K = 6
    torch.manual_seed(0)
    sample = torch.arange(K, dtype=torch.float32).reshape(K, 1, 1, 1, 1)

    sampler = SCARSampler(
        sigma_min=0.0, mix_steps=1,
        mix_weights=(0.3, 0.4, 0.3),
        extreme_mix_mode="symmetric",
    )
    mixed, _ = sampler._mix_x_t(sample, step_idx=0)

    # Symmetric:
    #   state k: 0.3*0 + 0.4*k + 0.3*5
    expected = 0.3 * 0 + 0.4 * torch.arange(K, dtype=torch.float32) + 0.3 * 5
    expected = expected.reshape(K, 1, 1, 1, 1)
    torch.testing.assert_close(mixed, expected, atol=1e-6, rtol=0)


def test_sampler_extreme_mix_mode_default_is_symmetric():
    """Default extreme_mix_mode should be 'symmetric' for backward compat."""
    sampler = SCARSampler(sigma_min=0.0)
    assert sampler.extreme_mix_mode == "symmetric"


def test_sampler_scar_changes_outputs_vs_baseline():
    """With scar_enabled=True, outputs should differ from baseline (diff states)."""
    from trellis.pipelines.samplers.flow_euler import FlowEulerGuidanceIntervalSampler
    K = 3
    device = "cpu"
    torch.manual_seed(7)
    noise = torch.randn(K, 8, 16, 16, 16, device=device)
    cond = torch.randn(K, 1374, 1024, device=device) * torch.tensor(
        [[1.0], [2.0], [3.0]], device=device
    ).unsqueeze(-1)
    neg_cond = torch.zeros_like(cond)

    baseline = FlowEulerGuidanceIntervalSampler(sigma_min=0.0)
    scar = SCARSampler(sigma_min=0.0, scar_enabled=True)

    model_a = MockFlowModel(seed=999)
    model_b = MockFlowModel(seed=999)

    out_a = baseline.sample(model_a, noise, cond=cond, neg_cond=neg_cond,
                            steps=12, cfg_strength=7.5, cfg_interval=(0.0, 1.0),
                            verbose=False)
    out_b = scar.sample(model_b, noise, cond=cond, neg_cond=neg_cond,
                        steps=12, cfg_strength=7.5, cfg_interval=(0.0, 1.0),
                        verbose=False)
    assert not torch.allclose(out_a.samples, out_b.samples, atol=1e-4)
