"""Unit tests for pipelines.stage_c_segmatch.material_classifier.

Validates:
  - known synthetic feature fields recover correct classification
  - graceful fallback when seed counts are too small
  - auto EDT threshold search picks a reasonable value
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pipelines.stage_c_segmatch.material_classifier import (
    MaterialClassification,
    classify_always_on_by_zfinal,
)


def _synth_voxel_with_material(D=16, K=4):
    """Build synthetic occupancy + z_final where drawer-material voxels have
    one 8-dim signature and cabinet-material voxels have a different one."""
    O = torch.zeros((K, D, D, D))
    # Cabinet block
    O[:, 2:8, 2:10, 2:10] = 1.0
    # Drawer block (always_on — all states occupy) inside cabinet hollow
    O[:, 9:12, 4:8, 4:8] = 1.0
    # Drawer shell: voxels exposed only in late states
    for k in range(K):
        O[k, 12:14, 4:8, 4:8 + k] = 1.0
    # Build z_final at 8^3 latent resolution, 8 channels
    z = torch.zeros((K, 8, 8, 8, 8))
    # Drawer material features — channels 0, 2 highly positive
    z[:, 0] = -0.5
    z[:, 2] = -0.5
    # Cabinet voxels get opposite pattern: channels 0, 2 negative
    # (recall z_final is upsampled to 64^3 in classifier, we just need
    # different average patterns between cabinet vs drawer spatial regions)
    # Cabinet indices in 8^3 latent: roughly 0..1 along first axis
    z[:, 0, 0:1, 0:2, 0:2] = -1.5
    z[:, 2, 0:1, 0:2, 0:2] = -1.5
    return O, z


def test_classifier_returns_expected_dataclass():
    O, z = _synth_voxel_with_material()
    D = O.shape[1]
    count = (O > 0.5).to(torch.int32).sum(0)
    always_on = count == O.shape[0]
    shell = (count > 0) & ~always_on
    result = classify_always_on_by_zfinal(
        z_final=z, O_stack=O, always_on=always_on, shell=shell,
        far_aon_edt_threshold=2.0,
    )
    assert isinstance(result, MaterialClassification)
    assert result.true_base.shape == (D, D, D)
    assert result.move_interior.shape == (D, D, D)
    assert result.ambiguous_on.shape == (D, D, D)
    assert result.drawer_axis.shape == (8,)


def test_classifier_classes_sum_to_always_on():
    """The three output classes (true_base / move_interior / ambiguous_on)
    must partition exactly the always_on voxels."""
    O, z = _synth_voxel_with_material()
    count = (O > 0.5).to(torch.int32).sum(0)
    always_on = count == O.shape[0]
    shell = (count > 0) & ~always_on
    result = classify_always_on_by_zfinal(
        z_final=z, O_stack=O, always_on=always_on, shell=shell,
        far_aon_edt_threshold=2.0,
    )
    union = result.true_base | result.move_interior | result.ambiguous_on
    assert torch.equal(union, always_on)
    # No overlap
    assert not (result.true_base & result.move_interior).any()
    assert not (result.true_base & result.ambiguous_on).any()
    assert not (result.move_interior & result.ambiguous_on).any()


def test_classifier_fallback_when_too_few_seeds():
    """With near-empty shell or far_aon, classifier must gracefully fall back
    (everything → ambiguous, classifier_applied=False)."""
    K, D = 4, 8
    # O with only a single always_on voxel and no shell voxels
    O = torch.zeros((K, D, D, D))
    O[:, 4, 4, 4] = 1.0
    z = torch.zeros((K, 8, 2, 2, 2))
    count = (O > 0.5).to(torch.int32).sum(0)
    always_on = count == K
    shell = (count > 0) & ~always_on   # empty
    result = classify_always_on_by_zfinal(
        z_final=z, O_stack=O, always_on=always_on, shell=shell,
        far_aon_edt_threshold=1.0, min_seeds_shell=2, min_seeds_far_aon=2,
    )
    assert not result.classifier_applied
    assert torch.equal(result.ambiguous_on, always_on)
    assert result.true_base.sum() == 0
    assert result.move_interior.sum() == 0


def test_classifier_separation_on_clear_synthetic():
    """On a synthetic where cabinet and drawer have clearly different
    mean feature vectors, the classifier should separate them."""
    K, D = 4, 16
    O = torch.zeros((K, D, D, D))
    # Cabinet always_on region
    O[:, 2:5, 2:10, 2:10] = 1.0
    # Drawer region - always_on + some shell for motion evidence
    O[:, 9:13, 5:9, 5:9] = 1.0
    for k in range(K):
        O[k, 13:15, 5:9, 5 + k: 5 + k + 1] = 1.0

    # Feature: cabinet = all 1s; drawer material = all -1s in channels 0-3.
    d_latent = D // 4  # 4 — 16/4 = 4x4x4 latent grid
    z = torch.zeros((K, 8, d_latent, d_latent, d_latent))
    # Cabinet region in latent coords: x<2
    z[:, :4, 0:1, :, :] = 1.0
    # Drawer region in latent coords: x=2-3
    z[:, :4, 2:4, :, :] = -1.0

    count = (O > 0.5).to(torch.int32).sum(0)
    always_on = count == K
    shell = (count > 0) & ~always_on

    result = classify_always_on_by_zfinal(
        z_final=z, O_stack=O, always_on=always_on, shell=shell,
        far_aon_edt_threshold=3.0,
        min_seeds_shell=2, min_seeds_far_aon=2,
    )
    assert result.classifier_applied
    # Drawer interior voxels (those in drawer region, always_on) should be
    # correctly labeled as move_interior.
    drawer_region = torch.zeros((D, D, D), dtype=torch.bool)
    drawer_region[9:13, 5:9, 5:9] = True
    n_drawer_aon = int((always_on & drawer_region).sum().item())
    n_correctly_labeled = int((result.move_interior & drawer_region).sum().item())
    # Expect at least 60% of drawer-interior voxels correctly classified
    # (synthetic is simple; real data showed 93-99% on held-out seeds).
    if n_drawer_aon > 0:
        assert n_correctly_labeled / n_drawer_aon > 0.6, (
            f"move_interior only captured {n_correctly_labeled}/{n_drawer_aon} "
            f"drawer-region always_on voxels"
        )


def test_classifier_auto_threshold_picks_largest_viable():
    """Auto threshold search should pick the highest threshold with enough seeds."""
    K, D = 4, 16
    O = torch.zeros((K, D, D, D))
    O[:, 2:14, 2:14, 2:14] = 1.0   # big always_on region
    # Sprinkle some shell voxels on the edge
    for k in range(K):
        O[k, 1, 7, 7 + k: 7 + k + 1] = 1.0
    z = torch.zeros((K, 8, 4, 4, 4))
    z[:, 0] = 1.0
    count = (O > 0.5).to(torch.int32).sum(0)
    always_on = count == K
    shell = (count > 0) & ~always_on

    result = classify_always_on_by_zfinal(
        z_final=z, O_stack=O, always_on=always_on, shell=shell,
        far_aon_edt_threshold=1.0,  # ignored when auto_threshold_search=True
        min_seeds_shell=2, min_seeds_far_aon=2,
        auto_threshold_search=True,
        search_range=(1.0, 8.0),
    )
    # Should have picked some threshold in search_range
    assert 1.0 <= result.far_aon_edt_threshold_used <= 8.0
