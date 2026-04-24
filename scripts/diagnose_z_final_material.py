"""Diagnostic: can z_final (8-dim SS latent) distinguish cabinet vs drawer-interior
at always_on voxels for 30857_b?

Run:
    python scripts/diagnose_z_final_material.py
or
    python scripts/diagnose_z_final_material.py --sample 7201_b

Outputs three decision metrics:
  1. Prototype cosine similarity (shell vs far-aon) — lower is better
  2. Between-class distance / within-class std — higher is better
  3. K-means(k=2) cluster purity on shell and far-aon seeds
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt


def run(sample: str, base_root: str, edt_far_threshold: float) -> None:
    base = f"{base_root}/{sample}/stage_b"
    print(f"Reading from: {base}\n")

    O = torch.from_numpy(np.load(f"{base}/O_stack.npy")).float()
    z_final = torch.load(f"{base}/z_final.pt", map_location="cpu")

    K = O.shape[0]
    print(f"O_stack shape: {tuple(O.shape)}")
    print(f"z_final shape: {tuple(z_final.shape)}\n")

    # 1. Upsample z to 64^3
    z = F.interpolate(z_final, size=(64, 64, 64), mode="trilinear",
                       align_corners=True)
    z_mean = z.mean(dim=0)                                   # (C, 64, 64, 64)
    C = z_mean.shape[0]

    # 2. Count-based classes
    count = (O > 0.5).to(torch.int32).sum(0)
    always_on = count == K
    shell = (count > 0) & ~always_on

    print(f"Classes:")
    print(f"  always_on: {int(always_on.sum().item())}")
    print(f"  shell:     {int(shell.sum().item())}\n")

    # 3. EDT (distance from every voxel to the nearest shell voxel)
    edt = distance_transform_edt(~shell.numpy())
    far_aon = always_on & (torch.from_numpy(edt) > edt_far_threshold)
    print(f"far_aon (EDT > {edt_far_threshold}): {int(far_aon.sum().item())}")

    if far_aon.sum() == 0 or shell.sum() == 0:
        print("ERROR: empty far_aon or shell; cannot compute prototypes")
        sys.exit(1)

    # 4. Gather features
    # z_mean shape (C, D, H, W). Indexing with a boolean (D, H, W) returns
    # (C, N) — transpose to (N, C).
    shell_f = z_mean[:, shell].T.numpy()                     # (N_shell, C)
    far_f = z_mean[:, far_aon].T.numpy()                     # (N_far, C)
    aon_f = z_mean[:, always_on].T.numpy()                   # (N_aon, C)

    print(f"\nshell feature stats (N={len(shell_f)}):")
    print(f"  mean: {shell_f.mean(0)}")
    print(f"  std:  {shell_f.std(0)}")
    print(f"\nfar_aon feature stats (N={len(far_f)}):")
    print(f"  mean: {far_f.mean(0)}")
    print(f"  std:  {far_f.std(0)}")

    # 5. Prototype cosine similarity (lower = more distinguishable)
    from scipy.spatial.distance import cosine
    prototype_cos = 1.0 - cosine(shell_f.mean(0), far_f.mean(0))
    print(f"\n=== Metric 1: Prototype cosine ===")
    print(f"  drawer(shell) vs cabinet(far_aon): {prototype_cos:.4f}")
    if prototype_cos < 0.7:
        verdict1 = "STRONG (drawer/cabinet well separated)"
    elif prototype_cos < 0.9:
        verdict1 = "WEAK (borderline)"
    else:
        verdict1 = "FAIL (prototypes too similar)"
    print(f"  verdict: {verdict1}")

    # 6. Between-class / within-class ratio
    between_dist = np.linalg.norm(shell_f.mean(0) - far_f.mean(0))
    within_std = 0.5 * (shell_f.std(0).mean() + far_f.std(0).mean())
    ratio = between_dist / max(within_std, 1e-8)
    print(f"\n=== Metric 2: Distance ratio ===")
    print(f"  between-class dist: {between_dist:.4f}")
    print(f"  within-class std:   {within_std:.4f}")
    print(f"  ratio:              {ratio:.3f}")
    if ratio > 3.0:
        verdict2 = "STRONG (clear clustering)"
    elif ratio > 1.0:
        verdict2 = "WEAK"
    else:
        verdict2 = "FAIL (clusters overlap)"
    print(f"  verdict: {verdict2}")

    # 7. K-means(k=2) cluster purity on seeded classes
    try:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(aon_f)
        labels = km.labels_

        # Which cluster is "drawer"? — the one whose center is closer to shell_f.mean
        c0 = km.cluster_centers_[0]
        c1 = km.cluster_centers_[1]
        cos_c0_shell = 1.0 - cosine(c0, shell_f.mean(0))
        cos_c1_shell = 1.0 - cosine(c1, shell_f.mean(0))
        drawer_cluster = 0 if cos_c0_shell > cos_c1_shell else 1

        n_drawer_cluster = int((labels == drawer_cluster).sum())
        n_cabinet_cluster = int((labels != drawer_cluster).sum())
        print(f"\n=== Metric 3: K-means(k=2) on always_on features ===")
        print(f"  drawer-cluster (center closer to shell): {n_drawer_cluster} voxels")
        print(f"  cabinet-cluster:                         {n_cabinet_cluster} voxels")
        print(f"  cluster center distance: {np.linalg.norm(c0 - c1):.4f}")
        print(f"  center 0: {c0}")
        print(f"  center 1: {c1}")

        # Test purity: of shell and far_aon (both in always_on is false for shell, so we re-cluster to verify)
        # Verify on far_aon seeds: what fraction ends up in cabinet cluster?
        # Since far_aon ⊂ always_on, their labels are directly in `labels`.
        # Build a mapping from (i,j,k) index to label index.
        aon_indices = torch.argwhere(always_on).numpy()      # (N_aon, 3)
        far_indices = torch.argwhere(far_aon).numpy()        # (N_far, 3)

        # Fast lookup: build dict from (i,j,k) → label index
        aon_lookup = {tuple(idx): labels[i] for i, idx in enumerate(aon_indices)}
        far_labels = np.array([aon_lookup[tuple(idx)] for idx in far_indices])
        far_in_cabinet = int((far_labels != drawer_cluster).sum())
        far_in_drawer = int((far_labels == drawer_cluster).sum())
        far_purity = far_in_cabinet / max(len(far_labels), 1)
        print(f"\n  far_aon seeds → cabinet cluster: {far_in_cabinet}/{len(far_labels)} ({100*far_purity:.1f}%)")
        print(f"  far_aon seeds → drawer cluster:  {far_in_drawer}/{len(far_labels)}")
        if far_purity > 0.85:
            verdict3 = "STRONG (clustering respects seeds)"
        elif far_purity > 0.65:
            verdict3 = "WEAK"
        else:
            verdict3 = "FAIL (clustering disagrees with EDT seeds)"
        print(f"  verdict: {verdict3}")
    except ImportError:
        print("\n(skipping K-means metric: sklearn not installed)")
        verdict3 = "SKIPPED"

    # 7.5. Supervised prototype classifier — validate on held-out seeds
    # This is the ACTUAL test: can we classify unseen shell/far_aon voxels
    # using ONLY the projection onto the drawer axis?
    print(f"\n=== Metric 4: Supervised prototype classifier (held-out) ===")
    rng = np.random.default_rng(0)
    shell_perm = rng.permutation(len(shell_f))
    far_perm = rng.permutation(len(far_f))
    n_shell_train = int(0.8 * len(shell_f))
    n_far_train = int(0.8 * len(far_f))
    shell_train = shell_f[shell_perm[:n_shell_train]]
    shell_val = shell_f[shell_perm[n_shell_train:]]
    far_train = far_f[far_perm[:n_far_train]]
    far_val = far_f[far_perm[n_far_train:]]

    drawer_proto = shell_train.mean(0)
    cabinet_proto = far_train.mean(0)
    drawer_axis = drawer_proto - cabinet_proto
    drawer_axis = drawer_axis / np.linalg.norm(drawer_axis)

    shell_proj_train = shell_train @ drawer_axis
    far_proj_train = far_train @ drawer_axis
    threshold = 0.5 * (shell_proj_train.mean() + far_proj_train.mean())
    print(f"  drawer proj (train): mean={shell_proj_train.mean():.3f}, "
          f"std={shell_proj_train.std():.3f}")
    print(f"  cabinet proj (train): mean={far_proj_train.mean():.3f}, "
          f"std={far_proj_train.std():.3f}")
    print(f"  decision threshold: {threshold:.3f}")

    shell_val_proj = shell_val @ drawer_axis
    far_val_proj = far_val @ drawer_axis
    shell_val_correct = int((shell_val_proj > threshold).sum())
    far_val_correct = int((far_val_proj <= threshold).sum())
    shell_acc = shell_val_correct / max(len(shell_val), 1)
    far_acc = far_val_correct / max(len(far_val), 1)
    avg_acc = 0.5 * (shell_acc + far_acc)
    print(f"  shell (drawer) val acc:    {shell_val_correct}/{len(shell_val)} "
          f"({100*shell_acc:.1f}%)")
    print(f"  far_aon (cabinet) val acc: {far_val_correct}/{len(far_val)} "
          f"({100*far_acc:.1f}%)")
    print(f"  balanced accuracy:         {100*avg_acc:.1f}%")

    if avg_acc > 0.9:
        verdict4 = "STRONG (classifier works)"
    elif avg_acc > 0.75:
        verdict4 = "WEAK (marginal classifier)"
    else:
        verdict4 = "FAIL (classifier unreliable)"
    print(f"  verdict: {verdict4}")

    # Apply classifier to all always_on voxels and report distribution
    aon_proj = aon_f @ drawer_axis
    n_aon_drawer = int((aon_proj > threshold).sum())
    n_aon_cabinet = int((aon_proj <= threshold).sum())
    print(f"\n  always_on classified as drawer_interior: {n_aon_drawer}")
    print(f"  always_on classified as true_base:        {n_aon_cabinet}")
    print(f"  always_on total: {len(aon_f)}")

    # 8. Combined verdict
    print(f"\n=== Combined verdict ===")
    print(f"  Metric 1 (prototype cos):     {verdict1}")
    print(f"  Metric 2 (distance ratio):    {verdict2}")
    print(f"  Metric 3 (K-means purity):    {verdict3}  [expected to fail — K-means finds wrong axis]")
    print(f"  Metric 4 (supervised hold-out): {verdict4}  [THIS is the real test]")

    # Metric 4 is the authoritative signal — it directly answers
    # "can we build a classifier from z_final?".
    if "STRONG" in verdict4:
        print("\n  >>> z_final IS USABLE — supervised prototype classifier works")
    elif "WEAK" in verdict4:
        print("\n  >>> z_final MARGINAL — try anyway, fall back to DiT if pipeline output poor")
    else:
        print("\n  >>> z_final INSUFFICIENT — escalate to DiT hidden feature (Stage B hook)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", default="30857_b",
                        help="Sample folder under base_root")
    parser.add_argument("--base_root", default="outputs",
                        help="Stage B output root directory")
    parser.add_argument("--edt_far_threshold", type=float, default=12.0,
                        help="EDT distance above which always_on is treated as cabinet seed")
    args = parser.parse_args()
    run(args.sample, args.base_root, args.edt_far_threshold)
