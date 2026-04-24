# Stage B Implementation Plan: SCAR + Post-hoc ICP

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Status (2026-04-18):** Initial implementation complete; mechanism has since evolved through empirical iteration. Tasks 1-11 below reflect the INITIAL plan; the "Implementation Evolution" section immediately after this header captures what actually landed.

**Goal:** Produce K=6 base-consistent 64^3 occupancy grids from 6 segmented input images with grounding carpet, via a modified TRELLIS Stage-1 sampling (SCAR: cross-state latent mixing + Tweedie-space gradient push + post-hoc ICP).

**Architecture:** (1) `SCARSampler` class subclassing `FlowEulerGuidanceIntervalSampler`; (2) translation-only rigid ICP post-decode; (3) Stage-B driver wiring them together with visualization. VGCF/BCAC kept as legacy ablation baselines.

**Tech Stack:** PyTorch, NumPy, SciPy (ndimage for trilinear resample), frozen TRELLIS pipeline, pytest for unit tests.

**Spec reference:** `record/design.md` §3 (Stage B, kept current).

**Input test data:** `mine/outputs/30857/rendering_joint_00_state_{00..05}.png` (6 images with carpet).

---

## Implementation Evolution (2026-04-18 — current state)

The plan below (Tasks 1-11) captures the initial "SCAR v3" design. The
actual implementation landed the same code skeleton but several parameters
and mechanisms were iterated empirically. Current state:

### Final Stage B mechanism (authoritative — see `design.md` §3)

**Total Euler steps: 25** (TRELLIS-image-large default, matches FreeArt3D
baseline; initial spec said 12 but that was a spec error).

**Mix phase — steps 0..7 (mix_steps=8), `extreme_mix_mode=mean_of_middles`:**

```
middle_mean = (1/(K-2)) * sum_{j=1..K-2} z_t^(j)
mixed = 0.3 * z_t^(0) + 0.4 * middle_mean + 0.3 * z_t^(K-1)
for all k: z_t^(k) <- mixed
```

ALL K states get the SAME mixed latent. Per-state differentiation comes
from subsequent DiT forward (c_k driven) and push. Initial plan used
per-state `0.3*s_0 + 0.4*s_k + 0.3*s_{K-1}` (symmetric); `mean_of_middles`
was added to eliminate latent-init bias that caused state-0 to drift.

**Push phase — all 25 steps, quadratic decay:**

```
alpha(s) = alpha_peak * (1 - s/(N-1))^2    with alpha_peak = 0.5, N = 25
v_aug^(k) = v_cfg^(k) + (alpha(s)/t) * w(x)^2 * (x_0^(k) - x0_bar)
```

Initial plan had 4 fixed push steps at specific positions; final has
continuous 25-step coverage with decaying strength so effective push
(alpha/t) decays linearly and is strongest when mask is most reliable.

**Mask — 16^3 latent-space, soft gate with floor:**

```
sigma2(x) = (1/K) * sum_k ||x_0^(k)(x) - x0_bar(x)||^2       # latent-space variance
energy(x) = ||x0_bar(x)||^2
active = energy >= percentile_{1-0.1}(energy)                 # top 10% by energy
tau = percentile_65(log(sigma2 + 1e-6) over active)
M = sigmoid((tau - log_sigma2) / 0.5) * active
w = [M * (1 - w_floor) + w_floor] * active                    # soft floor, w_floor=0.2
```

Mask evolution:
- Initial: latent variance + `active_fraction=0.8` (kept 80% voxels, mostly air)
- Tried: decode to 64^3, variance in occupancy space, pool back to 16^3
- Final: latent variance + `active_fraction=0.1` (top 10% by energy = object core)

The decoder-based detour was a dead-end: decode-pool introduces blurring,
decoder saturation adds noise, and SS VAE's near-zero KL means latent
variance already approximates occupancy variance. Staying in latent space
is cleaner and 3-5s faster per run.

Soft floor `w_floor = 0.2` guarantees all object voxels receive
`w_floor^2 = 4%` push even when misclassified as move, rescuing
base-interior generation noise and state-drift boundaries.

**Post-hoc ICP**: **DISABLED** (`icp_enabled: false` in v1.yaml).

Reasons for removing ICP from the main Stage B path:
1. Rigid translation on binary 64^3 via trilinear resample + rebinarize is
   LOSSY — the artifact from resampling can exceed the sub-voxel drift it
   tries to correct.
2. ICP target (state 0) may itself be the outlier (closed-state image has
   weaker spatial constraint, cabinet floats). Aligning 1..K-1 to state 0
   pulls the majority toward the outlier.
3. With `mean_of_middles` mix eliminating latent-init bias, residual rigid
   drift after SCAR should be minimal.
4. Remaining cross-state misalignment (if any) is non-rigid (surface
   jitter, state-0 topology difference) which rigid ICP cannot fix anyway.
5. Stage C's canonical-aggregation EM handles cross-state base alignment
   directly and more robustly.

Code kept (`pipelines/utils/icp.py`, ICP branch in `stage_b_scar.py`) for
legacy ablation via `icp_enabled: true`.

### Key config changes vs. initial plan (configs/v1.yaml)

| Key | Initial | Final |
|---|---|---|
| `scar.steps` | (inherits 12 from spec) | (inherits 25 from pipeline default) |
| `scar.mix_steps` | 4 | 8 |
| `scar.extreme_mix_mode` | n/a | `mean_of_middles` |
| `scar.alpha_schedule` | `[0.7, 1.0, 1.0, 0.5]` | generated from `alpha_peak` + `alpha_decay` |
| `scar.alpha_peak` | n/a | `0.5` |
| `scar.alpha_decay` | n/a | `quadratic` |
| `scar.active_fraction` | `0.8` | `0.1` |
| `scar.w_floor` | n/a | `0.2` |
| `scar.icp_enabled` | `true` | **`false`** (removed from Stage B main path) |

### SCARSampler API changes vs. initial plan

| Parameter | Initial | Final |
|---|---|---|
| `alpha_schedule` | fixed 4-entry tuple | full-length (25 entries) list, usually generated |
| `mix_steps` | 0 | 0 by default |
| `mix_weights` | n/a | `(0.3, 0.4, 0.3)` |
| `extreme_mix_mode` | n/a | `"symmetric"` default (`"mean_of_middles"` via config) |
| `w_floor` | n/a | `0.0` default (enabled via config) |
| `decoder` | n/a | added, then removed (dead-end) |
| `occupancy_threshold` | n/a | added, then removed |

### Files actually created vs plan

All planned files landed:
- `mine/pipelines/utils/icp.py` ✓
- `mine/TRELLIS/trellis/pipelines/samplers/scar.py` ✓
- `mine/pipelines/stage_b_scar.py` ✓
- `mine/tests/test_icp.py`, `test_scar_formula.py`, `test_scar_sampler.py`, `test_stage_b_scar_driver.py`, `test_stage_b_e2e_30857.py` ✓
- `mine/scripts/compare_stage_b_samplers.py` ✓
- `mine/scripts/plain_stage_b_per_step.py` ✓ (added for ablation)

No files removed; legacy VGCF/BCAC kept for ablation.

---

## File Structure

**Create:**
- `mine/pipelines/utils/icp.py` — translation-only rigid ICP (pure Python/NumPy)
- `mine/TRELLIS/trellis/pipelines/samplers/scar.py` — SCARSampler class
- `mine/pipelines/stage_b_scar.py` — Stage B driver
- `mine/tests/__init__.py` — empty
- `mine/tests/test_icp.py` — ICP unit tests
- `mine/tests/test_scar_formula.py` — SCAR gradient-push formula unit tests
- `mine/tests/test_scar_sampler.py` — SCARSampler integration tests (with mock model)
- `mine/tests/test_stage_b_smoke.py` — end-to-end smoke test on 30857

**Modify:**
- `mine/TRELLIS/trellis/pipelines/samplers/__init__.py` — export `SCARSampler`
- `mine/configs/v1.yaml` — add `scar:` block, change default sampler to `scar`
- `mine/run_v1.py` — add `scar` sampler branch

**Keep unchanged (legacy baselines):**
- `mine/pipelines/stage_b_vgcf.py`
- `mine/TRELLIS/trellis/pipelines/samplers/vgcf.py`
- `mine/TRELLIS/trellis/pipelines/samplers/bcac.py`

---

### Task 1: Setup test infrastructure

**Files:**
- Create: `mine/tests/__init__.py`

- [ ] **Step 1: Verify pytest is installed in the mine environment**

Run: `conda activate mine && python -c "import pytest; print(pytest.__version__)"`
Expected: pytest version prints, e.g. `8.x.x`. If `ModuleNotFoundError`, run `pip install pytest` and retry.

- [ ] **Step 2: Create tests directory with empty __init__**

Create file `mine/tests/__init__.py` with a single blank line.

- [ ] **Step 3: Verify pytest discovers the directory**

Run: `cd mine && python -m pytest tests/ -v --collect-only`
Expected: `no tests ran` (empty directory collected successfully, no errors).

- [ ] **Step 4: Commit**

```bash
cd mine
git add tests/__init__.py
git commit -m "test: add tests directory"
```

---

### Task 2: ICP — compute translation offset

**Files:**
- Create: `mine/pipelines/utils/icp.py`
- Create: `mine/tests/test_icp.py`

- [ ] **Step 1: Write failing test for `compute_translation_offset`**

Create `mine/tests/test_icp.py`:

```python
import numpy as np
import pytest

from pipelines.utils.icp import compute_translation_offset


def _make_cube(center, size, grid=64):
    """Return a (grid, grid, grid) binary occupancy of a cube centered at `center`."""
    g = np.zeros((grid, grid, grid), dtype=np.float32)
    cx, cy, cz = center
    h = size // 2
    g[cx - h: cx + h, cy - h: cy + h, cz - h: cz + h] = 1.0
    return g


def test_zero_offset_returns_zero():
    target = _make_cube((32, 32, 32), 10)
    source = target.copy()
    offset = compute_translation_offset(source, target, mask=None)
    np.testing.assert_allclose(offset, [0.0, 0.0, 0.0], atol=1e-6)


def test_known_translation_recovered():
    target = _make_cube((32, 32, 32), 10)
    source = _make_cube((32 + 2, 32 - 1, 32), 10)  # source is target shifted (+2, -1, 0)
    offset = compute_translation_offset(source, target, mask=None)
    # offset should bring source to target: target_centroid - source_centroid = (-2, +1, 0)
    np.testing.assert_allclose(offset, [-2.0, 1.0, 0.0], atol=0.05)


def test_mask_restricts_region():
    """Mask should exclude the shifted cube's move-region from centroid computation."""
    grid = 64
    target = np.zeros((grid, grid, grid), dtype=np.float32)
    source = np.zeros((grid, grid, grid), dtype=np.float32)
    # Base region: a cube at (20, 32, 32), same in both
    target[15:25, 27:37, 27:37] = 1.0
    source[15:25, 27:37, 27:37] = 1.0
    # Move region: differs between states
    target[45:55, 27:37, 27:37] = 1.0
    source[40:50, 27:37, 27:37] = 1.0  # move shifted by -5 along x
    # Mask covers only the base region
    mask = np.zeros((grid, grid, grid), dtype=bool)
    mask[15:25, 27:37, 27:37] = True
    offset = compute_translation_offset(source, target, mask=mask)
    np.testing.assert_allclose(offset, [0.0, 0.0, 0.0], atol=0.05)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd mine && python -m pytest tests/test_icp.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipelines.utils.icp'`.

- [ ] **Step 3: Implement `compute_translation_offset`**

Create `mine/pipelines/utils/icp.py`:

```python
"""Translation-only rigid ICP for cross-state occupancy grid alignment.

Used by Stage B to correct residual rigid drift after SCAR sampling.
Operates on 64^3 binary or soft occupancy grids. The problem is greatly
simplified compared to full ICP because: (a) the shapes are already
near-aligned (< 2 voxel drift), (b) we only search translations (no
rotation), and (c) weighted centroid alignment is optimal for
translation-only alignment.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def compute_translation_offset(
    source: np.ndarray,
    target: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute translation to bring `source` centroid onto `target` centroid.

    Parameters
    ----------
    source : np.ndarray
        Shape ``(D, H, W)``, float occupancy in [0, 1] or binary.
    target : np.ndarray
        Same shape as `source`.
    mask : np.ndarray, optional
        Shape ``(D, H, W)``, bool or 0/1 mask restricting which voxels
        contribute to centroid computation. If ``None``, all occupied
        voxels contribute.

    Returns
    -------
    np.ndarray
        Shape ``(3,)``, translation ``[dx, dy, dz]`` in voxel units that
        should be added to `source` to align with `target`.
        Returns zeros if either side has no mass.
    """
    assert source.shape == target.shape, \
        f"shape mismatch: {source.shape} vs {target.shape}"
    assert source.ndim == 3, f"expected 3D grid, got {source.ndim}D"

    s = source.astype(np.float64)
    t = target.astype(np.float64)
    if mask is not None:
        m = mask.astype(np.float64)
        s = s * m
        t = t * m

    s_sum = s.sum()
    t_sum = t.sum()
    if s_sum < 1e-8 or t_sum < 1e-8:
        return np.zeros(3, dtype=np.float32)

    # Weighted centroids in voxel index units.
    dd, hh, ww = np.meshgrid(
        np.arange(source.shape[0], dtype=np.float64),
        np.arange(source.shape[1], dtype=np.float64),
        np.arange(source.shape[2], dtype=np.float64),
        indexing="ij",
    )
    s_centroid = np.array([
        (dd * s).sum() / s_sum,
        (hh * s).sum() / s_sum,
        (ww * s).sum() / s_sum,
    ])
    t_centroid = np.array([
        (dd * t).sum() / t_sum,
        (hh * t).sum() / t_sum,
        (ww * t).sum() / t_sum,
    ])
    return (t_centroid - s_centroid).astype(np.float32)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd mine && python -m pytest tests/test_icp.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
cd mine
git add tests/test_icp.py pipelines/utils/icp.py
git commit -m "feat(icp): add translation-only centroid alignment"
```

---

### Task 3: ICP — apply translation with trilinear resample

**Files:**
- Modify: `mine/pipelines/utils/icp.py`
- Modify: `mine/tests/test_icp.py`

- [ ] **Step 1: Append failing tests for `apply_translation`**

Append to `mine/tests/test_icp.py`:

```python
from pipelines.utils.icp import apply_translation


def test_apply_translation_zero_is_identity():
    grid = _make_cube((32, 32, 32), 10)
    shifted = apply_translation(grid, np.array([0.0, 0.0, 0.0]))
    np.testing.assert_allclose(shifted, grid, atol=1e-6)


def test_apply_translation_integer_shift():
    grid = _make_cube((32, 32, 32), 10)
    shifted = apply_translation(grid, np.array([2.0, 0.0, 0.0]))
    # After +2 shift along dim 0, the cube should be at (34, 32, 32).
    expected = _make_cube((34, 32, 32), 10)
    np.testing.assert_allclose(shifted, expected, atol=1e-4)


def test_apply_translation_roundtrip():
    grid = _make_cube((32, 32, 32), 10)
    shifted = apply_translation(grid, np.array([1.3, -0.7, 0.5]))
    back = apply_translation(shifted, np.array([-1.3, 0.7, -0.5]))
    # Trilinear is a bit lossy; check mean absolute error is small.
    assert np.abs(back - grid).mean() < 0.02
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd mine && python -m pytest tests/test_icp.py -v`
Expected: 3 new tests FAIL with `ImportError: cannot import name 'apply_translation'`.

- [ ] **Step 3: Implement `apply_translation` using SciPy**

Append to `mine/pipelines/utils/icp.py`:

```python
from scipy.ndimage import shift as _scipy_shift


def apply_translation(
    grid: np.ndarray,
    offset: np.ndarray,
    mode: str = "constant",
    cval: float = 0.0,
) -> np.ndarray:
    """Shift a 3D grid by a sub-voxel translation via trilinear interpolation.

    Parameters
    ----------
    grid : np.ndarray
        Shape ``(D, H, W)``.
    offset : np.ndarray
        Shape ``(3,)``, translation in voxel units.
    mode : str, default 'constant'
        Boundary handling for ``scipy.ndimage.shift``.
    cval : float, default 0.0
        Fill value when ``mode='constant'``.

    Returns
    -------
    np.ndarray
        Shape ``(D, H, W)``, dtype float32.
    """
    shifted = _scipy_shift(
        grid.astype(np.float32),
        shift=offset,
        order=1,  # trilinear
        mode=mode,
        cval=cval,
    )
    return shifted.astype(np.float32)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd mine && python -m pytest tests/test_icp.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
cd mine
git add tests/test_icp.py pipelines/utils/icp.py
git commit -m "feat(icp): add trilinear apply_translation"
```

---

### Task 4: ICP — full alignment pipeline with translation cap

**Files:**
- Modify: `mine/pipelines/utils/icp.py`
- Modify: `mine/tests/test_icp.py`

- [ ] **Step 1: Append failing test for `align_to_reference`**

Append to `mine/tests/test_icp.py`:

```python
from pipelines.utils.icp import align_to_reference


def test_align_to_reference_basic():
    """Verify O_k gets aligned to O_0 within cap."""
    target = _make_cube((32, 32, 32), 10)
    source = _make_cube((32 + 1, 32, 32), 10)  # shifted +1 along x
    aligned, offset, capped = align_to_reference(
        source, target, base_mask=None, max_translation=1.5,
    )
    np.testing.assert_allclose(offset, [-1.0, 0.0, 0.0], atol=0.05)
    assert not capped
    # Aligned source should match target (up to trilinear precision)
    assert np.abs(aligned - target).mean() < 0.02


def test_align_to_reference_caps_large_translation():
    """If raw offset exceeds cap, output is capped and flagged."""
    target = _make_cube((32, 32, 32), 10)
    source = _make_cube((32 + 5, 32, 32), 10)  # shifted +5 along x (exceeds 1.5)
    _, offset, capped = align_to_reference(
        source, target, base_mask=None, max_translation=1.5,
    )
    # Capped: magnitude = 1.5, direction preserved
    assert np.isclose(np.linalg.norm(offset), 1.5, atol=0.01)
    assert capped


def test_align_to_reference_uses_base_mask():
    """With a mask excluding move region, alignment uses only base."""
    grid = 64
    target = np.zeros((grid, grid, grid), dtype=np.float32)
    source = np.zeros((grid, grid, grid), dtype=np.float32)
    # Base: identical
    target[20:30, 27:37, 27:37] = 1.0
    source[20:30, 27:37, 27:37] = 1.0
    # Move: different
    target[50:60, 27:37, 27:37] = 1.0
    source[45:55, 27:37, 27:37] = 1.0
    base_mask = np.zeros((grid, grid, grid), dtype=bool)
    base_mask[20:30, 27:37, 27:37] = True
    _, offset, _ = align_to_reference(
        source, target, base_mask=base_mask, max_translation=1.5,
    )
    np.testing.assert_allclose(offset, [0.0, 0.0, 0.0], atol=0.05)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd mine && python -m pytest tests/test_icp.py -v`
Expected: 3 new tests FAIL with `ImportError: cannot import name 'align_to_reference'`.

- [ ] **Step 3: Implement `align_to_reference`**

Append to `mine/pipelines/utils/icp.py`:

```python
def align_to_reference(
    source: np.ndarray,
    target: np.ndarray,
    base_mask: Optional[np.ndarray] = None,
    max_translation: float = 1.5,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Align `source` to `target` via translation-only centroid ICP.

    Parameters
    ----------
    source : np.ndarray
        Shape ``(D, H, W)``.
    target : np.ndarray
        Same shape as `source`.
    base_mask : np.ndarray, optional
        If provided, restrict centroid computation to base voxels.
    max_translation : float, default 1.5
        Clip translation magnitude to this many voxels to avoid large jumps
        from imperfect mask / mismatched move regions.

    Returns
    -------
    aligned : np.ndarray
        Shape ``(D, H, W)``, the translated source grid.
    offset : np.ndarray
        Shape ``(3,)``, applied offset (after capping).
    capped : bool
        True if the raw offset magnitude exceeded `max_translation`.
    """
    raw_offset = compute_translation_offset(source, target, mask=base_mask)
    magnitude = float(np.linalg.norm(raw_offset))
    capped = magnitude > max_translation
    if capped:
        offset = raw_offset * (max_translation / magnitude)
    else:
        offset = raw_offset
    aligned = apply_translation(source, offset)
    return aligned.astype(source.dtype), offset.astype(np.float32), capped
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd mine && python -m pytest tests/test_icp.py -v`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
cd mine
git add tests/test_icp.py pipelines/utils/icp.py
git commit -m "feat(icp): add align_to_reference with translation cap"
```

---

### Task 5: SCAR — gradient push formula (pure math, no DiT)

**Files:**
- Create: `mine/tests/test_scar_formula.py`
- Create: `mine/TRELLIS/trellis/pipelines/samplers/scar.py` (partial)

- [ ] **Step 1: Write failing tests for the core gradient push formula**

Create `mine/tests/test_scar_formula.py`:

```python
import sys
import os

import numpy as np
import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "TRELLIS"))

from trellis.pipelines.samplers.scar import (
    compute_tweedie_variance_mask,
    apply_scar_gradient_push,
)


def test_mask_zero_variance_is_all_base():
    """Identical x_0 across K states -> sigma2 = 0 -> M close to 1."""
    K, C, D, H, W = 3, 8, 4, 4, 4
    x_0 = torch.ones(K, C, D, H, W) * 2.0
    M, active = compute_tweedie_variance_mask(
        x_0, tau_percentile=0.65, eta=0.5,
        active_fraction=0.8, eps_log=1e-6,
    )
    assert M.shape == (D, H, W)
    # With zero variance, all active voxels should have high M
    assert M[active].mean().item() > 0.9


def test_mask_bimodal_distribution_separates():
    """Half low-variance, half high-variance -> mask close to binary."""
    K, C, D, H, W = 5, 8, 4, 4, 4
    x_0 = torch.zeros(K, C, D, H, W)
    # "base" half: all K same value (low variance)
    x_0[:, :, :2, :, :] = 1.0
    # "move" half: K different random values (high variance)
    torch.manual_seed(0)
    x_0[:, :, 2:, :, :] = torch.randn(K, C, 2, H, W) * 5.0
    M, active = compute_tweedie_variance_mask(
        x_0, tau_percentile=0.65, eta=0.5,
        active_fraction=0.8, eps_log=1e-6,
    )
    # Base half should have higher M than move half on average
    assert M[:2].mean() > M[2:].mean() + 0.3


def test_gradient_push_zero_alpha_is_identity():
    """alpha=0 -> v_aug == v_cfg."""
    K, C, D, H, W = 3, 8, 4, 4, 4
    v_cfg = torch.randn(K, C, D, H, W)
    x_0 = torch.randn(K, C, D, H, W)
    M = torch.ones(D, H, W) * 0.8
    v_aug = apply_scar_gradient_push(
        v_cfg=v_cfg, x_0=x_0, M=M, alpha=0.0, t=1.0,
    )
    torch.testing.assert_close(v_aug, v_cfg)


def test_gradient_push_mask_zero_is_identity():
    """M=0 everywhere -> v_aug == v_cfg regardless of alpha."""
    K, C, D, H, W = 3, 8, 4, 4, 4
    v_cfg = torch.randn(K, C, D, H, W)
    x_0 = torch.randn(K, C, D, H, W)
    M = torch.zeros(D, H, W)
    v_aug = apply_scar_gradient_push(
        v_cfg=v_cfg, x_0=x_0, M=M, alpha=1.0, t=1.0,
    )
    torch.testing.assert_close(v_aug, v_cfg)


def test_gradient_push_pulls_toward_consensus():
    """At base voxels (M=1), v_aug should be closer to v_bar than v_cfg."""
    torch.manual_seed(42)
    K, C, D, H, W = 6, 8, 4, 4, 4
    v_cfg = torch.randn(K, C, D, H, W)
    # Construct x_0 consistent with v_cfg via x_0 = z - t*v where z is shared noise
    t = 0.92
    z_shared = torch.randn(1, C, D, H, W).repeat(K, 1, 1, 1, 1)
    x_0 = z_shared - t * v_cfg
    M = torch.ones(D, H, W)  # treat all as base
    alpha = 1.0
    v_aug = apply_scar_gradient_push(v_cfg=v_cfg, x_0=x_0, M=M, alpha=alpha, t=t)
    v_bar = v_cfg.mean(dim=0, keepdim=True)
    # With alpha=1 and t=t, the push should nearly achieve full alignment.
    pre_distance = (v_cfg - v_bar).norm(dim=1).mean().item()
    post_distance = (v_aug - v_bar).norm(dim=1).mean().item()
    assert post_distance < pre_distance * 0.2, \
        f"push should reduce distance to v_bar by >80%, got {pre_distance:.4f} -> {post_distance:.4f}"


def test_mask_M_squared_selectivity():
    """M^2 should be more selective than M: mid-range (M=0.5) -> M^2=0.25."""
    K, C, D, H, W = 3, 2, 2, 2, 2
    v_cfg = torch.ones(K, C, D, H, W)
    # Set one state's v differently so diff is nonzero
    v_cfg[1] = 3.0
    x_0 = torch.zeros(K, C, D, H, W)
    x_0[1] = 2.0  # large diff at state 1
    M_full = torch.ones(D, H, W)
    M_half = torch.ones(D, H, W) * 0.5
    v_aug_full = apply_scar_gradient_push(v_cfg, x_0, M_full, alpha=1.0, t=1.0)
    v_aug_half = apply_scar_gradient_push(v_cfg, x_0, M_half, alpha=1.0, t=1.0)
    diff_full = (v_aug_full - v_cfg).abs().mean().item()
    diff_half = (v_aug_half - v_cfg).abs().mean().item()
    # Ratio should be ~4 (since M^2: 1.0 vs 0.25)
    assert 3.5 < diff_full / max(diff_half, 1e-9) < 4.5, \
        f"M^2 scaling wrong: diff_full/diff_half = {diff_full / max(diff_half, 1e-9):.2f}, expected ~4"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd mine && python -m pytest tests/test_scar_formula.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'trellis.pipelines.samplers.scar'`.

- [ ] **Step 3: Implement mask and gradient-push functions**

Create `mine/TRELLIS/trellis/pipelines/samplers/scar.py`:

```python
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


def compute_tweedie_variance_mask(
    x_0: torch.Tensor,
    tau_percentile: float = 0.65,
    eta: float = 0.5,
    active_fraction: float = 0.8,
    eps_log: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute a spatial mask where low cross-state variance -> base (M~1).

    Parameters
    ----------
    x_0 : torch.Tensor
        Shape ``(K, C, D, H, W)``, Tweedie estimates for K states.
    tau_percentile : float
        Percentile of log-variance (over active voxels) used as threshold.
    eta : float
        Sigmoid sharpness (log-space).
    active_fraction : float
        Fraction of voxels considered "active" by x0_bar energy.
        Bottom (1 - active_fraction) by energy are excluded as air.
    eps_log : float
        Added to sigma2 before log for numerical stability.

    Returns
    -------
    M : torch.Tensor
        Shape ``(D, H, W)``, mask in [0, 1], 1 = base.
    active : torch.Tensor
        Shape ``(D, H, W)``, bool mask of non-air voxels.
    """
    x0_bar = x_0.mean(dim=0, keepdim=True)                       # (1,C,D,H,W)
    diff = x_0 - x0_bar                                           # (K,C,D,H,W)
    sigma2 = (diff * diff).sum(dim=1).mean(dim=0)                 # (D,H,W)

    energy = (x0_bar[0] ** 2).sum(dim=0)                          # (D,H,W)
    energy_flat = energy.flatten()
    cut_q = max(0.0, 1.0 - active_fraction)
    energy_thresh = torch.quantile(energy_flat, cut_q)
    active = energy > energy_thresh

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
) -> torch.Tensor:
    """Apply SCAR v3 gradient push to the per-state CFG velocity.

    v_aug^(k) = v_cfg^(k) + (alpha / t) * M(x)^2 * (x_0^(k) - x0_bar)

    Under shared-noise initialization, this is mathematically equivalent
    to velocity-space interpolation v_cfg + alpha * M^2 * (v_bar - v_cfg)
    (see pipeline-redesign spec section 3.1 derivation). We use the
    Tweedie form for continuity with VGCF and DPS guidance structure.

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

    Returns
    -------
    v_aug : torch.Tensor
        Same shape as `v_cfg`.
    """
    x0_bar = x_0.mean(dim=0, keepdim=True)        # (1,C,D,H,W)
    diff = x_0 - x0_bar                            # (K,C,D,H,W)
    M_sq = (M * M).to(diff.dtype)                  # (D,H,W)
    M5 = M_sq[None, None]                          # broadcast to (1,1,D,H,W)
    push = (alpha / t) * M5 * diff                 # (K,C,D,H,W)
    return v_cfg + push
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd mine && python -m pytest tests/test_scar_formula.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
cd mine
git add tests/test_scar_formula.py TRELLIS/trellis/pipelines/samplers/scar.py
git commit -m "feat(scar): add Tweedie-variance mask and gradient-push formula"
```

---

### Task 6: SCAR — SCARSampler class

**Files:**
- Modify: `mine/TRELLIS/trellis/pipelines/samplers/scar.py`
- Modify: `mine/TRELLIS/trellis/pipelines/samplers/__init__.py`
- Create: `mine/tests/test_scar_sampler.py`

- [ ] **Step 1: Write failing sampler integration test with mock model**

Create `mine/tests/test_scar_sampler.py`:

```python
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
        # Deterministic noise as a function of x and cond for reproducibility.
        g = torch.Generator(device=x.device).manual_seed(
            self.seed + int(1e3 * float(t)) + int(cond.float().mean().item() * 100)
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
    """Early steps (0..3) should emit M_mean in diagnostics; late steps should not."""
    K = 3
    device = "cpu"
    noise = torch.randn(K, 8, 16, 16, 16, device=device)
    cond = torch.randn(K, 1374, 1024, device=device)
    neg_cond = torch.zeros_like(cond)
    model = MockFlowModel(seed=0)
    sampler = SCARSampler(sigma_min=0.0)
    out = sampler.sample(model, noise, cond=cond, neg_cond=neg_cond,
                         steps=12, cfg_strength=7.5, cfg_interval=(0.0, 1.0),
                         verbose=False)
    for s in range(4):
        assert "M_mean" in out.scar_diagnostics[s], f"step {s} missing M_mean"
        assert "alpha" in out.scar_diagnostics[s]
        assert out.scar_diagnostics[s]["alpha"] > 0
    for s in range(4, 12):
        assert out.scar_diagnostics[s]["alpha"] == 0.0, \
            f"step {s} should have alpha=0 but got {out.scar_diagnostics[s]['alpha']}"


def test_sampler_scar_changes_outputs_vs_baseline():
    """With scar_enabled=True, outputs should differ from baseline (diff states)."""
    from trellis.pipelines.samplers.flow_euler import FlowEulerGuidanceIntervalSampler
    K = 3
    device = "cpu"
    torch.manual_seed(7)
    noise = torch.randn(K, 8, 16, 16, 16, device=device)
    # Make conditions differ across K to induce cross-state divergence
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
    # Outputs should differ due to SCAR intervention
    assert not torch.allclose(out_a.samples, out_b.samples, atol=1e-4)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd mine && python -m pytest tests/test_scar_sampler.py -v`
Expected: FAIL with `ImportError: cannot import name 'SCARSampler'`.

- [ ] **Step 3: Implement SCARSampler class**

Append to `mine/TRELLIS/trellis/pipelines/samplers/scar.py`:

```python
class SCARSampler(FlowEulerGuidanceIntervalSampler):
    """Stage-1 sampler with early-step Tweedie gradient push for base consensus.

    Drop-in replacement for VGCFSampler. Differences:
      - Schedule: early-peak alpha = [0.7, 1.0, 1.0, 0.5] for steps 0..3
        (VGCF was sin(pi*progress), peaked mid-trajectory).
      - Scaling: lambda = alpha / t (no 1/K factor). At alpha=1 this
        achieves full consensus at base voxels (VGCF achieved 1/K ~= 0.17).
      - Steps 4..11 run plain K-parallel sampling.

    Mask mechanism is unchanged from VGCF (Tweedie variance + log
    percentile + sigmoid + active filter).
    """

    DEFAULT_ALPHA_SCHEDULE = (0.7, 1.0, 1.0, 0.5)

    def __init__(
        self,
        sigma_min: float,
        alpha_schedule: Tuple[float, ...] = DEFAULT_ALPHA_SCHEDULE,
        active_fraction: float = 0.8,
        tau_percentile: float = 0.65,
        eps_log: float = 1e-6,
        eta: float = 0.5,
        scar_enabled: bool = True,
    ) -> None:
        super().__init__(sigma_min=sigma_min)
        self.alpha_schedule = tuple(alpha_schedule)
        self.active_fraction = float(active_fraction)
        self.tau_percentile = float(tau_percentile)
        self.eps_log = float(eps_log)
        self.eta = float(eta)
        self.scar_enabled = bool(scar_enabled)

    def _alpha_for_step(self, step_idx: int) -> float:
        if step_idx < len(self.alpha_schedule):
            return float(self.alpha_schedule[step_idx])
        return 0.0

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

        t_seq = np.linspace(1.0, 0.0, steps + 1)
        t_seq = rescale_t * t_seq / (1.0 + (rescale_t - 1.0) * t_seq)
        t_pairs = list(zip(t_seq[:-1], t_seq[1:]))

        ret = edict({
            "samples": None,
            "pred_x_t": [],
            "pred_x_0": [],
            "scar_diagnostics": [],
        })

        pbar = tqdm(t_pairs, desc="SCAR sampling", disable=not verbose)
        for step_idx, (t, t_prev) in enumerate(pbar):
            t = float(t)
            t_prev = float(t_prev)
            dt = t - t_prev

            pred_x_0, _pred_eps, pred_v = self._get_model_prediction(
                model, sample, t,
                cond=cond, neg_cond=neg_cond,
                cfg_strength=cfg_strength,
                cfg_interval=cfg_interval,
                **kwargs,
            )

            alpha = self._alpha_for_step(step_idx) if self.scar_enabled else 0.0
            v_aug = pred_v
            diag: Dict[str, Any] = {"step": step_idx, "t": t, "alpha": alpha}

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
                )
                diag.update({
                    "M_mean": float(M.mean().item()),
                    "M_mean_active": float(M[active].mean().item() if active.any() else 0.0),
                    "active_voxels": int(active.sum().item()),
                    "push_norm": float((v_aug - pred_v).norm().item()),
                })

            pred_x_prev = sample - dt * v_aug
            sample = pred_x_prev

            ret.pred_x_t.append(pred_x_prev)
            ret.pred_x_0.append(pred_x_0)
            ret.scar_diagnostics.append(diag)

        ret.samples = sample
        return ret
```

- [ ] **Step 4: Export SCARSampler from the samplers module**

Read current `mine/TRELLIS/trellis/pipelines/samplers/__init__.py`. Append:

```python
from .scar import SCARSampler
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd mine && python -m pytest tests/test_scar_sampler.py -v`
Expected: 4 passed.

- [ ] **Step 6: Run the formula tests again to ensure no regression**

Run: `cd mine && python -m pytest tests/test_scar_formula.py tests/test_scar_sampler.py -v`
Expected: 10 passed total.

- [ ] **Step 7: Commit**

```bash
cd mine
git add tests/test_scar_sampler.py \
        TRELLIS/trellis/pipelines/samplers/scar.py \
        TRELLIS/trellis/pipelines/samplers/__init__.py
git commit -m "feat(scar): add SCARSampler class with early-peak schedule"
```

---

### Task 7: Stage B driver with ICP integration

**Files:**
- Create: `mine/pipelines/stage_b_scar.py`
- Create: `mine/tests/test_stage_b_scar_driver.py`

- [ ] **Step 1: Write failing driver test with mock pipeline**

Create `mine/tests/test_stage_b_scar_driver.py`:

```python
import os
import json
import tempfile

import numpy as np
import torch
import torch.nn as nn
import pytest

from pipelines.stage_b_scar import run_scar, SCARResult


class MockDecoder(nn.Module):
    def forward(self, z):
        # z: (K, 8, 16, 16, 16) -> (K, 1, 64, 64, 64) logits
        K = z.shape[0]
        # Make a simple cube in the center to simulate occupancy
        logits = torch.full((K, 1, 64, 64, 64), -5.0)
        logits[:, :, 20:44, 20:44, 20:44] = 5.0
        return logits


class MockFlowModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.resolution = 16
        self.patch_size = 2

    def forward(self, x, t, cond=None, **kwargs):
        return torch.zeros_like(x)


class MockPipeline:
    def __init__(self, device="cpu"):
        self.device = device
        self.models = {
            "sparse_structure_flow_model": MockFlowModel(),
            "sparse_structure_decoder": MockDecoder(),
        }
        from trellis.pipelines.samplers import SCARSampler
        self.sparse_structure_sampler = SCARSampler(sigma_min=0.0)
        self.sparse_structure_sampler_params = {
            "steps": 12,
            "cfg_strength": 7.5,
            "cfg_interval": (0.0, 1.0),
            "rescale_t": 1.0,
        }


def test_run_scar_returns_correct_shapes():
    """End-to-end driver produces K x 64^3 occupancy grids."""
    K = 3
    cond_tensors = {
        "cond": torch.randn(K, 1374, 1024),
        "neg_cond": torch.zeros(K, 1374, 1024),
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_scar(
            pipe=MockPipeline(),
            cond=cond_tensors,
            K=K,
            out_dir=tmpdir,
            cfg_scar={
                "alpha_schedule": [0.7, 1.0, 1.0, 0.5],
                "icp_enabled": True,
                "icp_max_translation": 1.5,
            },
            seed=42,
            remove_disk_flag=False,
        )
        assert result.O_stack.shape == (K, 64, 64, 64)
        assert result.O_stack_soft.shape == (K, 64, 64, 64)
        assert result.z_final.shape == (K, 8, 16, 16, 16)
        # ICP offsets should be recorded
        assert len(result.icp_offsets) == K
        assert result.icp_offsets[0].shape == (3,)
        # Persisted files exist
        assert os.path.exists(os.path.join(tmpdir, "O_stack.npy"))
        assert os.path.exists(os.path.join(tmpdir, "scar_diagnostics.json"))
        assert os.path.exists(os.path.join(tmpdir, "icp_report.json"))


def test_run_scar_with_shifted_input_corrects_offset():
    """Driver's ICP should detect and correct an injected shift."""
    K = 2
    cond = {"cond": torch.randn(K, 1374, 1024),
            "neg_cond": torch.zeros(K, 1374, 1024)}

    class ShiftedDecoder(MockDecoder):
        """State 1 is shifted by (+1, 0, 0) relative to state 0."""
        def forward(self, z):
            logits = torch.full((z.shape[0], 1, 64, 64, 64), -5.0)
            # state 0: cube at (20..44, 20..44, 20..44)
            logits[0, :, 20:44, 20:44, 20:44] = 5.0
            # state 1: cube shifted by +1 along x
            logits[1, :, 21:45, 20:44, 20:44] = 5.0
            return logits

    pipe = MockPipeline()
    pipe.models["sparse_structure_decoder"] = ShiftedDecoder()

    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_scar(
            pipe=pipe, cond=cond, K=K, out_dir=tmpdir,
            cfg_scar={"alpha_schedule": [0.7, 1.0, 1.0, 0.5],
                      "icp_enabled": True, "icp_max_translation": 1.5},
            seed=0, remove_disk_flag=False,
        )
        # ICP should have applied approximately (-1, 0, 0) to state 1 to align.
        offset_1 = result.icp_offsets[1]
        assert offset_1[0] < -0.5 and offset_1[0] > -1.5
        assert abs(offset_1[1]) < 0.2
        assert abs(offset_1[2]) < 0.2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd mine && python -m pytest tests/test_stage_b_scar_driver.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipelines.stage_b_scar'`.

- [ ] **Step 3: Implement Stage B driver**

Create `mine/pipelines/stage_b_scar.py`:

```python
"""Stage B driver: SCAR v3 sampling + post-hoc rigid ICP alignment.

Replaces `stage_b_vgcf.py` as the main driver for base-consistent Stage-1
sampling. The VGCF driver is preserved in the tree as a legacy ablation.

Pipeline:
  1. Run SCARSampler on K conditions (early-step gradient push for base
     consensus).
  2. Decode latents to 64^3 occupancy via SS VAE decoder.
  3. Optional: remove grounding disk/carpet voxels.
  4. Post-hoc rigid ICP: align each O_k to O_0 using the
     "intersection of all states" as a base-region mask.
  5. Persist O_stack, diagnostics, per-step Tweedie decodes,
     visualizations.

See spec section 3 (Stage B) for the full design.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from pipelines.utils.icp import align_to_reference
from pipelines.utils.postprocessing import remove_disk
from pipelines.utils.voxel_io import save_voxel_grid
from pipelines.utils.voxel_viz import (
    save_diagnostics_curves_html,
    save_voxel_html,
    save_voxel_stack_html,
)


@dataclass
class SCARResult:
    O_stack: torch.Tensor           # (K, 64, 64, 64) binary
    O_stack_soft: torch.Tensor      # (K, 64, 64, 64) sigmoid outputs
    z_final: torch.Tensor           # (K, 8, 16, 16, 16)
    diagnostics: List[Dict[str, Any]] = field(default_factory=list)
    icp_offsets: List[np.ndarray] = field(default_factory=list)
    icp_capped: List[bool] = field(default_factory=list)


def _compute_intersection_mask(
    O_stack: torch.Tensor, vote_threshold: float = 0.83,
) -> np.ndarray:
    """Base mask = voxels occupied in >= vote_threshold fraction of states.

    0.83 with K=6 means "occupied in >= 5 of 6 states".
    """
    K = O_stack.shape[0]
    min_votes = int(np.ceil(vote_threshold * K))
    votes = (O_stack > 0.5).sum(dim=0)               # (D, H, W) int
    return (votes >= min_votes).cpu().numpy().astype(bool)


def run_scar(
    pipe: Any,
    cond: Dict[str, torch.Tensor],
    K: int,
    out_dir: str,
    cfg_scar: Dict[str, Any],
    seed: int = 0,
    remove_disk_flag: bool = True,
    device: Optional[str] = None,
) -> SCARResult:
    """Run Stage B: SCAR sampling + VAE decode + ICP.

    Parameters
    ----------
    pipe : TrellisImageTo3DPipeline
        Must have ``pipe.sparse_structure_sampler`` set to a
        :class:`SCARSampler` instance.
    cond : dict
        ``{'cond': (K, N_tok, d), 'neg_cond': (K, N_tok, d)}``.
    K : int
        Number of articulation states.
    out_dir : str
        Destination directory.
    cfg_scar : dict
        Keys: ``alpha_schedule``, ``icp_enabled``, ``icp_max_translation``.
    seed : int
        Shared-noise seed.
    remove_disk_flag : bool
        Whether to run ``remove_disk`` post-decoding.

    Returns
    -------
    SCARResult
    """
    os.makedirs(out_dir, exist_ok=True)
    if device is None:
        device = cond["cond"].device
    device = torch.device(device)

    flow_model = pipe.models["sparse_structure_flow_model"]
    decoder = pipe.models["sparse_structure_decoder"]

    # Shared noise
    gen = torch.Generator(device=device).manual_seed(int(seed))
    eps = torch.randn(
        (1, 8, flow_model.resolution, flow_model.resolution, flow_model.resolution),
        device=device, generator=gen,
    )
    noise = eps.repeat(K, 1, 1, 1, 1)

    # Sampler params
    sampler_params: Dict[str, Any] = dict(pipe.sparse_structure_sampler_params)
    sampler = pipe.sparse_structure_sampler
    out = sampler.sample(
        flow_model, noise,
        cond=cond["cond"], neg_cond=cond["neg_cond"],
        verbose=True,
        **sampler_params,
    )
    z_final = out.samples

    # Decode
    logits = decoder(z_final)
    soft = torch.sigmoid(logits).squeeze(1)
    binary = (soft > 0.5).to(soft.dtype)

    # Remove disk (carpet voxels)
    if remove_disk_flag:
        binary_np = binary.detach().cpu().numpy()
        voxels_5d = binary_np[:, None].astype(np.float32)
        voxels_5d = remove_disk(voxels_5d)
        binary = torch.from_numpy(voxels_5d[:, 0]).to(
            device=soft.device, dtype=soft.dtype,
        )
        soft = soft * binary

    # Post-hoc ICP: align each O_k to O_0 using cross-state intersection mask
    icp_offsets: List[np.ndarray] = []
    icp_capped_flags: List[bool] = []
    if cfg_scar.get("icp_enabled", True) and K >= 2:
        base_mask = _compute_intersection_mask(binary, vote_threshold=0.83)
        target = binary[0].cpu().numpy()
        aligned_stack = [target]
        icp_offsets.append(np.zeros(3, dtype=np.float32))
        icp_capped_flags.append(False)
        max_t = float(cfg_scar.get("icp_max_translation", 1.5))
        for k in range(1, K):
            source = binary[k].cpu().numpy()
            aligned, offset, capped = align_to_reference(
                source, target,
                base_mask=base_mask,
                max_translation=max_t,
            )
            aligned_stack.append(aligned)
            icp_offsets.append(offset)
            icp_capped_flags.append(capped)
        binary = torch.from_numpy(np.stack(aligned_stack, axis=0)).to(
            device=device, dtype=soft.dtype,
        )
    else:
        icp_offsets = [np.zeros(3, dtype=np.float32) for _ in range(K)]
        icp_capped_flags = [False] * K

    # Persist
    save_voxel_grid(os.path.join(out_dir, "O_stack.npy"),
                    binary.detach().cpu().numpy().astype(np.uint8))
    save_voxel_grid(os.path.join(out_dir, "O_stack_soft.npy"),
                    soft.detach().cpu().numpy().astype(np.float16))
    torch.save(z_final.detach().cpu(), os.path.join(out_dir, "z_final.pt"))

    diag = list(getattr(out, "scar_diagnostics", None) or [])
    with open(os.path.join(out_dir, "scar_diagnostics.json"), "w") as f:
        json.dump(diag, f, indent=2)

    icp_report = {
        "offsets": [o.tolist() for o in icp_offsets],
        "capped": icp_capped_flags,
        "max_translation": float(cfg_scar.get("icp_max_translation", 1.5)),
    }
    with open(os.path.join(out_dir, "icp_report.json"), "w") as f:
        json.dump(icp_report, f, indent=2)

    # Visualizations
    viz_dir = os.path.join(out_dir, "viz")
    os.makedirs(viz_dir, exist_ok=True)
    binary_np = binary.detach().cpu().numpy().astype(np.float32)
    save_voxel_stack_html(
        binary_np,
        os.path.join(viz_dir, "O_stack.html"),
        title="Stage B SCAR: O_k (post-ICP)",
    )
    for k in range(binary_np.shape[0]):
        save_voxel_html(
            binary_np[k],
            os.path.join(viz_dir, f"O_k_{k:02d}.html"),
            title=f"O_{k}",
        )
    save_diagnostics_curves_html(
        diag,
        os.path.join(viz_dir, "scar_diagnostics.html"),
        title="SCAR per-step diagnostics",
        keys=("alpha", "M_mean", "M_mean_active", "push_norm", "active_voxels"),
    )

    return SCARResult(
        O_stack=binary,
        O_stack_soft=soft,
        z_final=z_final,
        diagnostics=diag,
        icp_offsets=icp_offsets,
        icp_capped=icp_capped_flags,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd mine && python -m pytest tests/test_stage_b_scar_driver.py -v`
Expected: 2 passed.

- [ ] **Step 5: Run full test suite to check no regression**

Run: `cd mine && python -m pytest tests/ -v`
Expected: 15 passed (3 icp + 3 icp-apply + 3 icp-align + 6 scar formula + 4 scar sampler + 2 driver).

- [ ] **Step 6: Commit**

```bash
cd mine
git add tests/test_stage_b_scar_driver.py pipelines/stage_b_scar.py
git commit -m "feat(stage_b): add SCAR + ICP driver"
```

---

### Task 8: Config and runner integration

**Files:**
- Modify: `mine/configs/v1.yaml`
- Modify: `mine/run_v1.py`

- [ ] **Step 1: Read current config to understand structure**

Run: `cat mine/configs/v1.yaml`
Note: `stage_b.sampler` currently is `bcac`, `vgcf` block exists. Must add `scar` block and change sampler default.

- [ ] **Step 2: Update `mine/configs/v1.yaml` — add scar block**

Modify the file by:
1. Changing `stage_b.sampler: bcac` to `stage_b.sampler: scar`
2. After the `bcac:` block, add:

```yaml
scar:
  # SCAR v3: Tweedie-space gradient push with alpha/t scaling,
  # early-peak schedule. See record/2026-04-17-pipeline-redesign.md sec 3.
  alpha_schedule: [0.7, 1.0, 1.0, 0.5]   # one entry per early step (s=0..3)
  active_fraction: 0.8                    # exclude bottom air voxels by energy
  tau_percentile: 0.65                    # log-variance percentile threshold
  eps_log: 1.0e-6                         # numerical stability in log
  eta: 0.5                                # sigmoid sharpness in log-variance
  # Post-hoc ICP
  icp_enabled: true
  icp_max_translation: 1.5                # voxels; residual drift is typically <1
  icp_vote_threshold: 0.83                # K=6 -> occupied in >= 5 of 6 is base
  # Shared-noise seed is taken from io.seed (top-level config)
  remove_disk: true
```

- [ ] **Step 3: Update `mine/run_v1.py` — add scar sampler branch**

Find the block starting with `if sampler_choice == "bcac":` around line 86. After the `bcac`/`vgcf` blocks, add a new `scar` block before the final `else:`.

The resulting sampler-selection block should read:

```python
sampler_choice = str(cfg.get("stage_b", {}).get("sampler", "scar")).lower()
sigma_min = pipe.sparse_structure_sampler.sigma_min

if sampler_choice == "scar":
    scar_cfg = cfg.get("scar", {})
    pipe.sparse_structure_sampler = SCARSampler(
        sigma_min=sigma_min,
        alpha_schedule=tuple(scar_cfg.get("alpha_schedule", [0.7, 1.0, 1.0, 0.5])),
        active_fraction=float(scar_cfg.get("active_fraction", 0.8)),
        tau_percentile=float(scar_cfg.get("tau_percentile", 0.65)),
        eps_log=float(scar_cfg.get("eps_log", 1.0e-6)),
        eta=float(scar_cfg.get("eta", 0.5)),
        scar_enabled=True,
    )
elif sampler_choice == "bcac":
    # (existing bcac block unchanged)
    ...
elif sampler_choice == "vgcf":
    # (existing vgcf block unchanged)
    ...
else:
    raise ValueError(f"unknown stage_b.sampler: {sampler_choice}")
```

Also add at the top imports: `from trellis.pipelines.samplers import VGCFSampler, BCACSampler, SCARSampler`.

And replace the call to `run_vgcf(...)` with a conditional on sampler_choice: if `scar`, call `run_scar(...)`. This means importing `from pipelines.stage_b_scar import run_scar, SCARResult` and branching in the Stage B section of main():

```python
# ------------------ Stage B ------------------
stage_b_dir = os.path.join(args.output_dir, "stage_b")

need_stage_b = args.stage in ("all", "b") or \
               not os.path.exists(os.path.join(stage_b_dir, "O_stack.npy"))
if need_stage_b:
    if sampler_choice == "scar":
        from pipelines.stage_b_scar import run_scar
        scar_cfg_with_io = dict(cfg.scar)
        scar_cfg_with_io["remove_disk"] = bool(cfg.scar.get("remove_disk", True))
        stage_b_res = run_scar(
            pipe=pipe, cond=cond, K=int(cfg.io.K),
            out_dir=stage_b_dir, cfg_scar=scar_cfg_with_io,
            seed=int(cfg.io.seed),
            remove_disk_flag=scar_cfg_with_io["remove_disk"],
        )
    else:
        stage_b_res = run_vgcf(
            pipe=pipe, cond=cond, K=int(cfg.io.K),
            cfg_vgcf=cfg.vgcf, out_dir=stage_b_dir,
            cfg_stage_b=cfg.get("stage_b"),
        )
else:
    # (existing skip-reload block)
    ...
```

Replace all downstream references to `vgcf_res` with `stage_b_res` (they share the same attribute names `O_stack`, `O_stack_soft`, `z_final`, `diagnostics`).

Rename the hard-coded `stage_b_vgcf` directory to `stage_b` so the directory is sampler-agnostic.

- [ ] **Step 4: Smoke-test the Python imports**

Run: `cd mine && python -c "from run_v1 import *; print('imports OK')"`
Expected: `imports OK` printed with no errors.

- [ ] **Step 5: Commit**

```bash
cd mine
git add configs/v1.yaml run_v1.py
git commit -m "feat(config): wire SCAR sampler into run_v1 and v1.yaml"
```

---

### Task 9: End-to-end smoke test on 30857 real data

**Files:**
- Create: `mine/tests/test_stage_b_e2e_30857.py` (slow test, marked skip if data missing)

- [ ] **Step 1: Write the e2e test**

Create `mine/tests/test_stage_b_e2e_30857.py`:

```python
"""End-to-end smoke test: SCAR + ICP on real 30857 input.

This test is marked slow and is skipped if:
  - CUDA is not available
  - TRELLIS checkpoints are not cached
  - The 30857 input directory is missing

Intended to be run manually or in CI with the mine environment set up.
"""

import json
import os
import sys
import shutil

import numpy as np
import pytest


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "30857")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "30857_scar_smoke")

requires_cuda = pytest.mark.skipif(
    not __import__("torch").cuda.is_available(),
    reason="needs CUDA",
)
requires_data = pytest.mark.skipif(
    not os.path.isdir(DATA_DIR) or
    not os.path.exists(os.path.join(DATA_DIR, "rendering_joint_00_state_00.png")),
    reason=f"30857 data missing in {DATA_DIR}",
)


@requires_cuda
@requires_data
@pytest.mark.slow
def test_scar_on_30857_produces_consistent_base():
    """SCAR should produce K=6 grids with high pairwise IoU in the base region."""
    import torch
    from omegaconf import OmegaConf
    from trellis.pipelines import TrellisImageTo3DPipeline
    from trellis.pipelines.samplers import SCARSampler
    from PIL import Image

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from pipelines.stage_b_scar import run_scar

    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR, exist_ok=True)

    # Build pipeline
    pipe = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
    pipe.cuda()

    # Load 6 images
    K = 6
    images = [
        Image.open(os.path.join(DATA_DIR, f"rendering_joint_00_state_{i:02d}.png")).convert("RGBA")
        for i in range(K)
    ]
    preprocessed = [pipe.preprocess_image(im) for im in images]
    cond = pipe.get_cond(preprocessed)

    # Swap in SCAR sampler
    pipe.sparse_structure_sampler = SCARSampler(
        sigma_min=pipe.sparse_structure_sampler.sigma_min,
    )
    pipe.sparse_structure_sampler_params["steps"] = 12
    pipe.sparse_structure_sampler_params["cfg_strength"] = 7.5
    pipe.sparse_structure_sampler_params["cfg_interval"] = (0.0, 1.0)

    # Run
    result = run_scar(
        pipe=pipe, cond=cond, K=K,
        out_dir=OUT_DIR,
        cfg_scar={
            "alpha_schedule": [0.7, 1.0, 1.0, 0.5],
            "icp_enabled": True,
            "icp_max_translation": 1.5,
            "icp_vote_threshold": 0.83,
        },
        seed=0, remove_disk_flag=True,
    )

    # Checks
    assert result.O_stack.shape == (K, 64, 64, 64)

    O_np = result.O_stack.detach().cpu().numpy().astype(bool)

    # Base-region (intersection of >=5 states) IoU should be very high (~0.9+)
    votes = O_np.sum(axis=0)
    base_mask = votes >= 5
    if base_mask.sum() > 0:
        # For each state, IoU of its base-region part with the base_mask itself
        ious = []
        for k in range(K):
            k_base = O_np[k] & base_mask
            iou = k_base.sum() / max(base_mask.sum(), 1)
            ious.append(iou)
        mean_iou = float(np.mean(ious))
        assert mean_iou > 0.9, f"base-region IoU only {mean_iou:.3f}; expected > 0.9"

    # Load diagnostics and verify SCAR was active for early steps
    with open(os.path.join(OUT_DIR, "scar_diagnostics.json")) as f:
        diag = json.load(f)
    for s in range(4):
        assert diag[s]["alpha"] > 0, f"step {s} alpha=0 (SCAR not active)"
    for s in range(4, 12):
        assert diag[s]["alpha"] == 0.0, f"step {s} alpha != 0 (SCAR still active)"

    # Load ICP report and verify all offsets are within cap
    with open(os.path.join(OUT_DIR, "icp_report.json")) as f:
        icp_rep = json.load(f)
    for offs in icp_rep["offsets"]:
        assert np.linalg.norm(offs) <= 1.5 + 1e-4, f"ICP offset exceeds cap: {offs}"

    # Visualization files exist
    assert os.path.exists(os.path.join(OUT_DIR, "viz", "O_stack.html"))
    assert os.path.exists(os.path.join(OUT_DIR, "viz", "scar_diagnostics.html"))
```

- [ ] **Step 2: Run the e2e test**

Run: `cd mine && python -m pytest tests/test_stage_b_e2e_30857.py -v -s -m slow --no-header`

Expected (if CUDA + checkpoints + data all available): 1 passed, prints SCAR sampling progress and final IoU/offset stats.

If skipped: all 3 of {CUDA, checkpoints, data} present? If so, debug the failure. If any missing, note the skip reason — this test is intended for manual validation.

- [ ] **Step 3: Inspect visualizations**

Open in browser:
- `mine/outputs/30857_scar_smoke/viz/O_stack.html` — dropdown over 6 states, verify base looks consistent
- `mine/outputs/30857_scar_smoke/viz/scar_diagnostics.html` — verify curves: alpha is [0.7,1.0,1.0,0.5] for steps 0-3 then 0; M_mean_active is meaningful (not 0 or 1 flat); push_norm is nonzero for steps 0-3

Manual check: compare visually against `outputs/30857_step_diag/` (baseline from `diagnose_trellis_steps.py`) to confirm SCAR reduces surface jitter.

- [ ] **Step 4: Commit**

```bash
cd mine
git add tests/test_stage_b_e2e_30857.py
git commit -m "test(stage_b): add end-to-end smoke test on 30857 with real TRELLIS"
```

---

### Task 10: Stage B vs VGCF ablation comparison script

**Files:**
- Create: `mine/scripts/compare_stage_b_samplers.py`

- [ ] **Step 1: Write the comparison script**

Create `mine/scripts/compare_stage_b_samplers.py`:

```python
"""Compare SCAR v3, VGCF (legacy), BCAC (legacy), and plain K-parallel
sampling on a given input directory, producing a side-by-side metric table.

Usage:
    conda activate mine
    python scripts/compare_stage_b_samplers.py \
        --input_dir outputs/30857 \
        --output_dir outputs/30857_ablation \
        --K 6

Outputs under `<output_dir>/`:
    scar/        # SCAR v3 result + diagnostics
    vgcf/        # legacy VGCF (unchanged)
    bcac/        # legacy BCAC (unchanged)
    plain/       # plain K-parallel (SCAR with scar_enabled=False)
    summary.json # per-sampler: pairwise_iou_mean, base_iou, icp_magnitude, wall_time
    summary.html # bar chart visualization
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from PIL import Image
import torch


def run_one_sampler(name, pipe, cond, K, out_dir):
    from pipelines.stage_b_scar import run_scar
    from pipelines.stage_b_vgcf import run_vgcf
    from trellis.pipelines.samplers import SCARSampler, VGCFSampler, BCACSampler
    from omegaconf import OmegaConf

    sigma_min = pipe.sparse_structure_sampler.sigma_min
    t0 = time.time()
    if name == "scar":
        pipe.sparse_structure_sampler = SCARSampler(sigma_min=sigma_min)
        pipe.sparse_structure_sampler_params["steps"] = 12
        res = run_scar(
            pipe=pipe, cond=cond, K=K, out_dir=out_dir,
            cfg_scar={"alpha_schedule": [0.7, 1.0, 1.0, 0.5],
                      "icp_enabled": True, "icp_max_translation": 1.5,
                      "icp_vote_threshold": 0.83},
            seed=0, remove_disk_flag=True,
        )
        O = res.O_stack
        icp_offsets = res.icp_offsets
    elif name == "plain":
        pipe.sparse_structure_sampler = SCARSampler(sigma_min=sigma_min, scar_enabled=False)
        pipe.sparse_structure_sampler_params["steps"] = 12
        res = run_scar(
            pipe=pipe, cond=cond, K=K, out_dir=out_dir,
            cfg_scar={"alpha_schedule": [0.0, 0.0, 0.0, 0.0],
                      "icp_enabled": True, "icp_max_translation": 1.5,
                      "icp_vote_threshold": 0.83},
            seed=0, remove_disk_flag=True,
        )
        O = res.O_stack
        icp_offsets = res.icp_offsets
    elif name == "vgcf":
        pipe.sparse_structure_sampler = VGCFSampler(sigma_min=sigma_min)
        vgcf_cfg = OmegaConf.create({
            "enabled": True, "steps": 12, "cfg_strength": 7.5,
            "cfg_interval": [0.0, 1.0], "rescale_t": 1.0,
            "lambda_max": 1.0, "t_stop": 0.2, "eta": 0.5, "seed": 0,
            "active_fraction": 0.8, "tau_percentile": 0.65,
            "eps_log": 1.0e-6, "lambda_schedule": "warmup",
        })
        res = run_vgcf(pipe=pipe, cond=cond, K=K, cfg_vgcf=vgcf_cfg, out_dir=out_dir)
        O = res.O_stack
        icp_offsets = [np.zeros(3)] * K
    elif name == "bcac":
        pipe.sparse_structure_sampler = BCACSampler(sigma_min=sigma_min)
        vgcf_cfg = OmegaConf.create({
            "enabled": True, "steps": 12, "cfg_strength": 7.5,
            "cfg_interval": [0.0, 1.0], "rescale_t": 1.0,
            "lambda_max": 1.0, "t_stop": 0.2, "eta": 0.5, "seed": 0,
            "active_fraction": 0.8, "tau_percentile": 0.65,
            "eps_log": 1.0e-6, "lambda_schedule": "warmup",
        })
        res = run_vgcf(pipe=pipe, cond=cond, K=K, cfg_vgcf=vgcf_cfg, out_dir=out_dir)
        O = res.O_stack
        icp_offsets = [np.zeros(3)] * K
    else:
        raise ValueError(f"unknown sampler: {name}")
    wall = time.time() - t0

    O_np = O.detach().cpu().numpy().astype(bool)

    # Metrics
    ious = []
    for i in range(K):
        for j in range(i + 1, K):
            inter = (O_np[i] & O_np[j]).sum()
            union = (O_np[i] | O_np[j]).sum()
            ious.append(float(inter / max(union, 1)))
    pairwise_iou_mean = float(np.mean(ious))

    votes = O_np.sum(axis=0)
    base_mask = votes >= max(5, int(0.83 * K))
    if base_mask.sum() > 0:
        base_ious = [float((O_np[k] & base_mask).sum() / max(base_mask.sum(), 1)) for k in range(K)]
        base_iou_mean = float(np.mean(base_ious))
    else:
        base_iou_mean = 0.0

    icp_magnitudes = [float(np.linalg.norm(o)) for o in icp_offsets]

    return {
        "name": name,
        "pairwise_iou_mean": pairwise_iou_mean,
        "base_iou_mean": base_iou_mean,
        "icp_magnitude_mean": float(np.mean(icp_magnitudes)),
        "icp_magnitude_max": float(np.max(icp_magnitudes)),
        "wall_time_sec": wall,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--K", type=int, default=6)
    p.add_argument("--samplers", nargs="+",
                   default=["scar", "plain", "vgcf", "bcac"])
    p.add_argument("--pattern", default="rendering_joint_00_state_{i:02d}.png")
    args = p.parse_args()

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "TRELLIS"))

    from trellis.pipelines import TrellisImageTo3DPipeline

    os.makedirs(args.output_dir, exist_ok=True)

    # Load images and encode once
    images = [
        Image.open(os.path.join(args.input_dir, args.pattern.format(i=i))).convert("RGBA")
        for i in range(args.K)
    ]
    pipe = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
    pipe.cuda()
    preprocessed = [pipe.preprocess_image(im) for im in images]
    cond = pipe.get_cond(preprocessed)

    summary = []
    for name in args.samplers:
        out = os.path.join(args.output_dir, name)
        row = run_one_sampler(name, pipe, cond, args.K, out)
        print(f"[{name}] pairwise_iou={row['pairwise_iou_mean']:.3f}  "
              f"base_iou={row['base_iou_mean']:.3f}  "
              f"icp_mag_mean={row['icp_magnitude_mean']:.3f}  "
              f"wall={row['wall_time_sec']:.1f}s")
        summary.append(row)

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to {args.output_dir}/summary.json")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify script import**

Run: `cd mine && python -c "import scripts.compare_stage_b_samplers" 2>&1 | head -5`

If `scripts` dir is not a package, verify it's still runnable via `python scripts/compare_stage_b_samplers.py --help`:

Run: `cd mine && python scripts/compare_stage_b_samplers.py --help`
Expected: argparse help text prints.

- [ ] **Step 3: Run the comparison on 30857 (manual, if GPU + checkpoints available)**

Run: `cd mine && python scripts/compare_stage_b_samplers.py --input_dir outputs/30857 --output_dir outputs/30857_ablation --K 6`

Expected: Prints 4 rows, one per sampler, with IoU metrics. Takes ~1-2 min on H800.

Inspect `outputs/30857_ablation/summary.json` to confirm SCAR has highest `base_iou_mean` (expected > 0.9) and VGCF likely lower (< 0.85), BCAC in between, plain as baseline.

- [ ] **Step 4: Commit**

```bash
cd mine
git add scripts/compare_stage_b_samplers.py
git commit -m "test(stage_b): add ablation comparison script for SCAR/VGCF/BCAC/plain"
```

---

### Task 11: Documentation update in spec

**Files:**
- Modify: `record/2026-04-17-pipeline-redesign.md`

- [ ] **Step 1: Mark Stage B as implemented in the spec**

Find the section `## 9. 代码改动全景` in `record/2026-04-17-pipeline-redesign.md`. For each Stage B-related row (scar.py, stage_b_scar.py, icp.py), prepend `(DONE)` to the state column.

Example: the row for `mine/pipelines/utils/icp.py` should read:
```
| `mine/pipelines/utils/icp.py` | 不存在 | (DONE 2026-04-17) **新增**：rigid translation-only ICP |
```

- [ ] **Step 2: Commit**

```bash
cd ..
git add record/2026-04-17-pipeline-redesign.md
git commit -m "docs(stage_b): mark Stage B tasks as implemented"
```

---

## Self-Review

Spec coverage check:

| Spec §3 requirement | Task implementing |
|---|---|
| SCAR v3 mechanism (Tweedie gradient push, α/t scaling) | Task 5 (formula), Task 6 (sampler class) |
| Early schedule α=[0.7, 1.0, 1.0, 0.5] | Task 6 (DEFAULT_ALPHA_SCHEDULE), Task 8 (config) |
| Tweedie-variance mask with M² selectivity | Task 5 |
| active voxel filter by energy | Task 5 |
| log-space percentile threshold | Task 5 |
| Late steps plain (s≥4) | Task 6 (_alpha_for_step returns 0) |
| Remove disk | Task 7 (run_scar) |
| Post-hoc rigid ICP | Tasks 2-4 (icp module), Task 7 (wiring) |
| ICP uses base-region mask | Task 7 (_compute_intersection_mask) |
| ICP max translation 1.5 voxels | Task 4 (align_to_reference), Task 8 (config) |
| SCAR M as diagnostic | Task 6 (scar_diagnostics in sample return), Task 7 (persisted to JSON) |
| Ablation: Option A / plain / VGCF / BCAC | Task 10 (compare script) |
| End-to-end validation on 30857 | Task 9 |

All §3 requirements covered.

Placeholder scan: no "TBD", "TODO", "implement later" left in the plan. Each step has complete code.

Type consistency check: `SCARResult` fields (`O_stack`, `O_stack_soft`, `z_final`, `diagnostics`, `icp_offsets`, `icp_capped`) are used consistently across Task 7 (definition), Task 9 (test assertions), Task 10 (script consumes). `compute_translation_offset`, `apply_translation`, `align_to_reference` signatures match between definition and call sites.

Scope check: Stage B only. Stage C/D/E/F plans come later.

---

## Execution Handoff

Plan complete and saved to `record/2026-04-17-stage-b-plan.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
