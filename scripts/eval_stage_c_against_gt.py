"""
Rigorous GT-vs-prediction evaluation for Stage C articulation pipeline.

For each of four PartNet-Mobility objects, we:
  1. Parse the URDF to identify the moving link and its mesh-file stems.
  2. Load the 6 per-state glb meshes, split each scene into {moving, base} using
     URDF stems (with a defensive "moved-between-states" fallback for glb sub-
     geoms that the exporter renamed away from their obj stems).
  3. Normalize every mesh into a single [-0.5, 0.5] frame using ONE union
     bounding-box across all states and both parts (critical: the bounding box
     must be shared across k so that phi-range is expressed in consistent units).
  4. Voxelize each mesh at 64^3 on a canonical grid where cell center (i) is
     located at world coordinate (i - 31.5)/63.
  5. Rigidly align the GT base voxels to the TRELLIS O_stack base (6-state
     intersection of O_stack>0.5) using 48 axis-aligned rotations + FPFH-RANSAC
     + point-to-point ICP. The resulting SE(3) transform is applied to the
     moving-part voxels and to the GT joint axis.
  6. Compute joint-type, axis-angle, pivot-line-distance, range, base and
     per-state move IoU metrics. Emit a Markdown table and a JSON dump.

Everything lives in a single file so it can run unchanged from the command line.
"""

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
import trimesh
import open3d as o3d


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(r"C:/Users/管晨皓/Desktop/temp/standard/mine")
REF_DIR = ROOT / "outputs" / "reference"
STAGE_B_DIR = ROOT / "outputs" / "experiment_b"
STAGE_C_DIR = ROOT / "outputs" / "experiment_c_v8_dit"

OBJECTS = [
    ("30857", "Table"),
    ("7201", "Oven"),
    ("7128", "Microwave"),
    ("26525", "Table"),
]

OUT_JSON = ROOT / "scripts" / "eval_results.json"


# ---------------------------------------------------------------------------
# Step 1: URDF parsing
# ---------------------------------------------------------------------------

def parse_urdf_moving_link(urdf_path: Path, joint_info: dict):
    """Find the URDF joint that matches joint_info and enumerate its mesh stems.

    Matching policy:
      * Joint types must match exactly.
      * The absolute axis directions must be within 5 degrees.
      * The joint range (upper - lower) must match |range[1]-range[0]| within 1e-3.
    If multiple joints satisfy these, the closest range match wins.
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    target_type = joint_info["type"]
    target_dir = np.asarray(joint_info["axis"]["direction"], dtype=np.float64)
    target_dir = target_dir / (np.linalg.norm(target_dir) + 1e-12)
    target_range = abs(joint_info["range"][1] - joint_info["range"][0])

    candidates = []
    for joint in root.findall("joint"):
        jtype = joint.get("type")
        if jtype != target_type:
            continue
        axis_el = joint.find("axis")
        if axis_el is None:
            continue
        axis = np.asarray([float(x) for x in axis_el.get("xyz").split()], dtype=np.float64)
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        # Axis is signed but ± equivalent for joint direction.
        cos = abs(float(np.dot(axis, target_dir)))
        cos = min(1.0, max(-1.0, cos))
        ang_err = np.degrees(np.arccos(cos))
        if ang_err > 5.0:
            continue
        limit_el = joint.find("limit")
        if limit_el is None:
            continue
        lo = float(limit_el.get("lower"))
        hi = float(limit_el.get("upper"))
        rng = abs(hi - lo)
        range_err = abs(rng - target_range)
        if range_err > 1e-3:
            continue
        child_el = joint.find("child")
        if child_el is None:
            continue
        child_link_name = child_el.get("link")
        candidates.append((range_err, ang_err, joint.get("name"), child_link_name))

    if not candidates:
        raise RuntimeError(f"No URDF joint matches joint_info for {urdf_path}")
    candidates.sort()
    _, _, joint_name, link_name = candidates[0]

    # Enumerate mesh filenames on that link.
    mesh_stems = set()
    for link in root.findall("link"):
        if link.get("name") != link_name:
            continue
        for visual in link.findall("visual"):
            geom = visual.find("geometry")
            if geom is None:
                continue
            mesh = geom.find("mesh")
            if mesh is None:
                continue
            fn = mesh.get("filename")
            if fn:
                mesh_stems.add(Path(fn).stem)
    return joint_name, link_name, mesh_stems


# ---------------------------------------------------------------------------
# Step 2: GT mesh loading + per-link split
# ---------------------------------------------------------------------------

def _concat_or_empty(mesh_list):
    if not mesh_list:
        # sentinel: single-vertex mesh so voxelize doesn't crash
        return trimesh.Trimesh(vertices=[[0, 0, 0]], faces=[])
    return trimesh.util.concatenate(mesh_list)


def load_glb_split(glb_path: Path, moving_stems: set):
    """Return (moving_mesh, base_mesh, moving_node_names).

    Two-pass split:
      * Pass A: URDF stem match on the scene-graph NODE name (nodes retain
        the original obj stem; geometry keys often have .020 suffixes).
      * Pass B: for the motion-inference fallback, the caller may later re-
        classify a node by its translation delta; see split_with_motion_fallback.
    """
    scene = trimesh.load(glb_path, process=False, force="scene")
    if not isinstance(scene, trimesh.Scene):
        # already a concatenated Trimesh (unusual); treat as all base
        return trimesh.Trimesh(vertices=[[0, 0, 0]], faces=[]), scene, set()

    moving_nodes = set()
    node_transforms = {}
    node_meshes = {"move": [], "base": []}
    for node in scene.graph.nodes_geometry:
        T, gname = scene.graph[node]
        geom = scene.geometry[gname]
        # Apply the node transform so we work in scene coords.
        m = geom.copy()
        m.apply_transform(T)
        node_transforms[node] = T.copy()
        if node in moving_stems:
            moving_nodes.add(node)
            node_meshes["move"].append(m)
        else:
            node_meshes["base"].append(m)
    return node_meshes, node_transforms, moving_nodes


def _split_with_motion_fallback(per_state_nodes, per_state_transforms, urdf_moving_nodes):
    """Augment URDF-stem moving set with any node whose translation column
    differs between state 0 and any other state by > 1e-3 m. This captures glb
    sub-geoms (e.g. "mesh4/mesh4-geometry#...") that the exporter renamed away
    from the URDF obj stems but rigidly move with the same link.
    """
    K = len(per_state_transforms)
    all_nodes = set()
    for tr in per_state_transforms:
        all_nodes.update(tr.keys())

    moved = set()
    for n in all_nodes:
        if n not in per_state_transforms[0]:
            continue
        T0 = per_state_transforms[0][n]
        for k in range(1, K):
            if n not in per_state_transforms[k]:
                continue
            Tk = per_state_transforms[k][n]
            disp = np.linalg.norm(Tk[:3, 3] - T0[:3, 3])
            if disp > 1e-3:
                moved.add(n)
                break
    return urdf_moving_nodes | moved


# ---------------------------------------------------------------------------
# Step 3: Joint normalization
# ---------------------------------------------------------------------------

def compute_union_bounds(mesh_lists):
    lo = np.full(3, +np.inf)
    hi = np.full(3, -np.inf)
    for meshes in mesh_lists:
        for m in meshes:
            if m.vertices.shape[0] == 0:
                continue
            b = m.bounds
            lo = np.minimum(lo, b[0])
            hi = np.maximum(hi, b[1])
    center = (lo + hi) * 0.5
    extent = float((hi - lo).max())
    return lo, hi, center, extent


def normalize_mesh(mesh: trimesh.Trimesh, center: np.ndarray, extent: float) -> trimesh.Trimesh:
    out = mesh.copy()
    if out.vertices.shape[0] == 0:
        return out
    out.apply_translation(-center)
    out.apply_scale(1.0 / extent)
    return out


# ---------------------------------------------------------------------------
# Step 4: Voxelize on a canonical 64^3 grid
# ---------------------------------------------------------------------------

# Cell center world coord: (i - 31.5) / 63, so cell-center(0) = -0.5, cell-center(63) = +0.5.
GRID_N = 64
GRID_PITCH = 1.0 / 63.0


def mesh_to_voxels_64(mesh: trimesh.Trimesh) -> np.ndarray:
    """Voxelize at pitch 1/63, then embed into a canonical 64x64x64 bool grid
    where cell index i corresponds to world (i - 31.5)/63.

    Uses .voxelized + matrix placement via .transform so we never rely on the
    trimesh grid being exactly 64^3.
    """
    out = np.zeros((GRID_N, GRID_N, GRID_N), dtype=bool)
    if mesh.vertices.shape[0] < 4 or mesh.faces.shape[0] == 0:
        return out
    try:
        vg = mesh.voxelized(pitch=GRID_PITCH)
    except Exception:
        return out
    mat = vg.matrix.astype(bool)
    if mat.size == 0:
        return out
    # trimesh VoxelGrid.transform: voxel-index -> world, with translation
    # giving world coord of voxel (0,0,0) corner.
    tr = vg.transform
    world_origin = tr[:3, 3]  # corner of cell (0,0,0)
    # In our canonical grid, corner of cell (i,j,k) = ((i - 32)/63, (j-32)/63, (k-32)/63)
    # So i0 such that (i0 - 32)/63 ~= world_origin.x ->  i0 = round(world_origin.x*63 + 32)
    i0 = int(np.round(world_origin[0] * 63.0 + 32.0))
    j0 = int(np.round(world_origin[1] * 63.0 + 32.0))
    k0 = int(np.round(world_origin[2] * 63.0 + 32.0))

    # Compute intersection of [i0:i0+mat.shape[0], ...] with [0:64, ...]
    src_x0 = max(0, -i0)
    src_y0 = max(0, -j0)
    src_z0 = max(0, -k0)
    src_x1 = min(mat.shape[0], GRID_N - i0)
    src_y1 = min(mat.shape[1], GRID_N - j0)
    src_z1 = min(mat.shape[2], GRID_N - k0)
    if src_x0 >= src_x1 or src_y0 >= src_y1 or src_z0 >= src_z1:
        return out
    dst_x0 = src_x0 + i0
    dst_y0 = src_y0 + j0
    dst_z0 = src_z0 + k0
    out[dst_x0:dst_x0 + (src_x1 - src_x0),
        dst_y0:dst_y0 + (src_y1 - src_y0),
        dst_z0:dst_z0 + (src_z1 - src_z0)] |= mat[src_x0:src_x1, src_y0:src_y1, src_z0:src_z1]
    return out


# ---------------------------------------------------------------------------
# Step 5: Rigid alignment
# ---------------------------------------------------------------------------

def voxel_to_points(vox: np.ndarray) -> np.ndarray:
    """(64,64,64) bool -> (N, 3) world-coordinate points at cell centers."""
    idx = np.argwhere(vox)
    if idx.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    centers = (idx.astype(np.float64) - 31.5) / 63.0
    return centers


def iou_masks(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    if union == 0:
        return 0.0
    return inter / union


def _axis_aligned_rotations():
    """48 axis-permutation rotations with determinant +1 (full octahedral group
    including reflections, which we then filter for det=+1 = 24).
    To hedge against chirality flips in the exporter we ALSO include det=-1
    reflections -> 48 total. We pick the best regardless of det.
    """
    Rs = []
    perms = [
        (0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0),
    ]
    signs = [(sx, sy, sz) for sx in (+1, -1) for sy in (+1, -1) for sz in (+1, -1)]
    for p in perms:
        for s in signs:
            R = np.zeros((3, 3))
            for row, (col, sign) in enumerate(zip(p, s)):
                R[row, col] = sign
            Rs.append(R)
    # 6 * 8 = 48
    return Rs


def _warp_voxel(src_vox: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply SE(3) T to source voxel centers, then rasterize into 64^3 grid
    on the canonical layout. Nearest-neighbor.
    """
    pts = voxel_to_points(src_vox)
    if pts.shape[0] == 0:
        return np.zeros_like(src_vox, dtype=bool)
    pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)
    warped = pts_h @ T.T
    warped = warped[:, :3]
    idx = np.round(warped * 63.0 + 31.5).astype(np.int64)
    keep = np.all((idx >= 0) & (idx < GRID_N), axis=1)
    idx = idx[keep]
    out = np.zeros((GRID_N, GRID_N, GRID_N), dtype=bool)
    if idx.size == 0:
        return out
    out[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return out


def _pts_to_pcd(pts: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd


def _icp_refine(src_pts: np.ndarray, tgt_pts: np.ndarray, T_init: np.ndarray,
                max_corr: float = 3.0 / 63.0, max_iter: int = 80) -> np.ndarray:
    src_pcd = _pts_to_pcd(src_pts)
    tgt_pcd = _pts_to_pcd(tgt_pts)
    reg = o3d.pipelines.registration.registration_icp(
        src_pcd, tgt_pcd, max_corr, T_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter,
                                                         relative_fitness=1e-8,
                                                         relative_rmse=1e-8),
    )
    return np.asarray(reg.transformation)


def rigid_align_binary_voxels(src_mask: np.ndarray, tgt_mask: np.ndarray):
    """Find similarity transform (R, s, t) expressed as an SE(3) matrix with a
    uniform scale baked into R such that warp(src, T) ~= tgt. Strategy:
      * Axis-aligned rotation seeds (48) x scale seeds (sweep the ratio of
        src vs tgt point-cloud std) -> best by warped-IoU.
      * Refine the best seed with point-to-point ICP (which only refines
        rotation + translation, so scale is fixed at the seed scale).
    The TRELLIS O_stack frame and the union-bounds-normalized GT frame are
    related by a rigid transform up to a UNIFORM SCALE (TRELLIS normalizes
    per-image to its own canonical unit cube, while GT normalization uses
    the full union bounding box across all states). Treating the problem
    as similarity (not rigid) is necessary for honest evaluation.

    Returns (T 4x4, best_iou, seed_scale).
    """
    src_pts = voxel_to_points(src_mask)
    tgt_pts = voxel_to_points(tgt_mask)
    if src_pts.shape[0] == 0 or tgt_pts.shape[0] == 0:
        return np.eye(4), 0.0, 1.0

    src_c = src_pts.mean(axis=0)
    tgt_c = tgt_pts.mean(axis=0)

    # Estimate isotropic scale via RMS radii.
    src_std = float(np.sqrt(((src_pts - src_c) ** 2).sum(axis=1).mean()))
    tgt_std = float(np.sqrt(((tgt_pts - tgt_c) ** 2).sum(axis=1).mean()))
    s0 = tgt_std / max(src_std, 1e-8)
    # Sweep scale near s0 with a fine grid.
    scale_candidates = [s0 * c for c in (0.65, 0.75, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20, 1.35)]

    # Rank all (rotation, scale) seeds by IoU, keep top K for ICP refinement.
    seeds = []
    for scale in scale_candidates:
        for R in _axis_aligned_rotations():
            M = scale * R  # 3x3 rotation-with-scale
            T = np.eye(4)
            T[:3, :3] = M
            T[:3, 3] = tgt_c - M @ src_c
            warped = _warp_voxel(src_mask, T)
            s = iou_masks(warped, tgt_mask)
            seeds.append((s, scale, T))
    seeds.sort(key=lambda x: -x[0])

    best_T = seeds[0][2]
    best_iou = seeds[0][0]
    best_scale = seeds[0][1]

    # Try ICP from top-K seeds with varied correspondence thresholds.
    top_k = min(5, len(seeds))
    for s0_iou, s0_scale, s0_T in seeds[:top_k]:
        for max_corr in (3.0 / 63.0, 6.0 / 63.0, 10.0 / 63.0):
            try:
                T_icp = _icp_refine(src_pts, tgt_pts, s0_T, max_corr=max_corr, max_iter=150)
                warped_icp = _warp_voxel(src_mask, T_icp)
                icp_iou = iou_masks(warped_icp, tgt_mask)
                if icp_iou > best_iou:
                    best_iou = icp_iou
                    best_T = T_icp
                    best_scale = s0_scale
            except Exception:
                pass
    return best_T, float(best_iou), float(best_scale)


# ---------------------------------------------------------------------------
# Step 6: metrics
# ---------------------------------------------------------------------------

def axis_angle_error_deg(pred_omega: np.ndarray, gt_omega: np.ndarray) -> float:
    a = pred_omega / (np.linalg.norm(pred_omega) + 1e-12)
    b = gt_omega / (np.linalg.norm(gt_omega) + 1e-12)
    cos = abs(float(np.dot(a, b)))
    cos = min(1.0, max(-1.0, cos))
    return float(np.degrees(np.arccos(cos)))


def line_line_distance(p1: np.ndarray, d1: np.ndarray, p2: np.ndarray, d2: np.ndarray) -> float:
    """Shortest distance between two infinite 3D lines."""
    d1 = d1 / (np.linalg.norm(d1) + 1e-12)
    d2 = d2 / (np.linalg.norm(d2) + 1e-12)
    n = np.cross(d1, d2)
    nn = np.linalg.norm(n)
    if nn < 1e-8:
        # parallel: perpendicular distance from p2 to line (p1, d1)
        w = p2 - p1
        return float(np.linalg.norm(w - np.dot(w, d1) * d1))
    return float(abs(np.dot((p2 - p1), n)) / nn)


# ---------------------------------------------------------------------------
# Pipeline per-object evaluation
# ---------------------------------------------------------------------------

def eval_object(obj_id: str, category: str) -> dict:
    print(f"\n===== {obj_id} ({category}) =====")
    ref = REF_DIR / obj_id
    joint_info = json.loads((ref / "joint_info.json").read_text())[0]
    joint_name, link_name, mesh_stems = parse_urdf_moving_link(ref / "mobility.urdf", joint_info)
    print(f" Matched URDF joint={joint_name} link={link_name} with {len(mesh_stems)} mesh stems")

    # Load 6 states.
    per_state_nodes = []
    per_state_transforms = []
    for k in range(6):
        glb = ref / "gt_mesh" / f"0{k}.glb"
        nodes_per_group, node_tr, _ = load_glb_split(glb, mesh_stems)
        per_state_nodes.append(nodes_per_group)
        per_state_transforms.append(node_tr)

    # Motion-fallback augmentation: expand moving set using per-node translation
    # delta between state 0 and any other state.
    urdf_nodes = set(mesh_stems)  # URDF stems == glb node names in our data
    moving_nodes = _split_with_motion_fallback(per_state_nodes, per_state_transforms, urdf_nodes)
    print(f" Moving nodes after motion fallback: {len(moving_nodes)} (URDF-only: {len(urdf_nodes)})")

    # Re-split per state using augmented node set.
    per_state_meshes = []
    for k in range(6):
        scene = trimesh.load(ref / "gt_mesh" / f"0{k}.glb", process=False, force="scene")
        mv, bs = [], []
        for node in scene.graph.nodes_geometry:
            T, gname = scene.graph[node]
            geom = scene.geometry[gname].copy()
            geom.apply_transform(T)
            if node in moving_nodes:
                mv.append(geom)
            else:
                bs.append(geom)
        per_state_meshes.append({"move": mv, "base": bs})

    # Union bounding box across all states and both parts.
    all_groups = []
    for d in per_state_meshes:
        all_groups.append(d["move"])
        all_groups.append(d["base"])
    lo, hi, center, extent = compute_union_bounds(all_groups)
    print(f" Union bounds center={center} extent={extent:.4f}")

    # Voxelize.
    gt_move_voxels_k = np.zeros((6, GRID_N, GRID_N, GRID_N), dtype=bool)
    gt_base_voxels_k = np.zeros((6, GRID_N, GRID_N, GRID_N), dtype=bool)
    for k in range(6):
        mv = _concat_or_empty([normalize_mesh(m, center, extent) for m in per_state_meshes[k]["move"]])
        bs = _concat_or_empty([normalize_mesh(m, center, extent) for m in per_state_meshes[k]["base"]])
        gt_move_voxels_k[k] = mesh_to_voxels_64(mv)
        gt_base_voxels_k[k] = mesh_to_voxels_64(bs)
        print(f"  k={k}  gt_move={int(gt_move_voxels_k[k].sum())}  gt_base={int(gt_base_voxels_k[k].sum())}")

    # State-0 canonical base (spec: use state 0's base as canonical).
    gt_base = gt_base_voxels_k[0]

    # Load pipeline outputs.
    o_stack = np.load(STAGE_B_DIR / f"{obj_id}_b" / "stage_b" / "O_stack.npy").astype(bool)
    pipeline_base_target = np.all(o_stack, axis=0)  # TRELLIS always-on approximation
    pipeline_union = np.any(o_stack, axis=0)

    jp = torch.load(STAGE_C_DIR / obj_id / "stage_c" / "joint_params.pt",
                    map_location="cpu", weights_only=False)
    pred_type = str(jp["joint_type"])
    pred_omega = jp["omega"].cpu().numpy().astype(np.float64)
    pred_omega = pred_omega / (np.linalg.norm(pred_omega) + 1e-12)
    pred_q = jp["q"].cpu().numpy().astype(np.float64)
    pred_phi = jp["phi_k"].cpu().numpy().astype(np.float64)
    pred_phi_K = float(abs(pred_phi[-1] - pred_phi[0]))

    canonical_base = np.load(STAGE_C_DIR / obj_id / "stage_c" / "canonical_base.npy").astype(bool)
    canonical_move = np.load(STAGE_C_DIR / obj_id / "stage_c" / "canonical_move.npy").astype(bool)
    psa = np.load(STAGE_C_DIR / obj_id / "stage_c" / "per_state_assignment.npy")

    # Per-state pipeline move voxels: (O_stack > 0.5) & (per_state_assignment == 1)
    pipe_move_k = np.zeros_like(o_stack, dtype=bool)
    for k in range(6):
        pipe_move_k[k] = o_stack[k] & (psa[k] == 1)
    pipe_base_per_state = np.zeros_like(o_stack, dtype=bool)
    for k in range(6):
        pipe_base_per_state[k] = o_stack[k] & (psa[k] == 0)

    # --- Stage B base-alignment IoU (for sanity) ---
    stage_b_base_iou = iou_masks(gt_base, pipeline_base_target)
    print(f" Stage-B raw base IoU (GT vs always-on O_stack): {stage_b_base_iou:.4f}")

    # --- Frame alignment (similarity) on base voxels ---
    T_align, best_iou, seed_scale = rigid_align_binary_voxels(gt_base, pipeline_base_target)
    print(f" Alignment IoU (GT base -> O_stack always-on): {best_iou:.4f} (seed scale={seed_scale:.3f})")

    # Apply T to GT move voxels.
    aligned_move_k = np.zeros_like(gt_move_voxels_k, dtype=bool)
    for k in range(6):
        aligned_move_k[k] = _warp_voxel(gt_move_voxels_k[k], T_align)
    aligned_base = _warp_voxel(gt_base, T_align)

    # Apply T to GT joint axis/origin. T is a similarity: T[:3,:3] = s * R.
    # Axis direction transforms by the pure rotation R (undo scale by dividing
    # out its magnitude). Origin transforms as a point by the full similarity.
    M = T_align[:3, :3]
    t = T_align[:3, 3]
    M_col_norm = float(np.linalg.norm(M[:, 0]))
    R_only = M / max(M_col_norm, 1e-12)

    gt_dir = np.asarray(joint_info["axis"]["direction"], dtype=np.float64)
    gt_dir = gt_dir / (np.linalg.norm(gt_dir) + 1e-12)
    gt_origin_world = np.asarray(joint_info["axis"]["origin"], dtype=np.float64)
    gt_origin_norm = (gt_origin_world - center) / extent
    omega_aligned = R_only @ gt_dir
    omega_aligned = omega_aligned / (np.linalg.norm(omega_aligned) + 1e-12)
    origin_aligned = M @ gt_origin_norm + t

    # GT range in pipeline (TRELLIS) normalized units.
    # Prismatic: range is a length. GT-normalized length = (urdf_range)/extent.
    # After similarity (pipeline) alignment with scale s, the equivalent pipeline
    # length = (urdf_range / extent) * s.
    # Revolute: range is an angle, scale-invariant.
    if joint_info["type"] == "revolute":
        gt_range_norm = float(abs(joint_info["range"][1] - joint_info["range"][0]))
    else:
        gt_range_gt = float(abs(joint_info["range"][1] - joint_info["range"][0])) / extent
        gt_range_norm = gt_range_gt * M_col_norm

    # --- Metrics ---
    type_correct = (pred_type == joint_info["type"])
    axis_err = axis_angle_error_deg(pred_omega, omega_aligned)
    if joint_info["type"] == "revolute":
        pivot_err = line_line_distance(pred_q, pred_omega, origin_aligned, omega_aligned)
    else:
        pivot_err = float("nan")
    range_err = abs(pred_phi_K - gt_range_norm)
    per_state_iou = [iou_masks(pipe_move_k[k], aligned_move_k[k]) for k in range(6)]
    move_iou_mean = float(np.mean(per_state_iou))
    base_iou = iou_masks(canonical_base, aligned_base)

    # Also: "base in each state" iou vs aligned GT base (base should be static).
    per_state_base_iou = [iou_masks(pipe_base_per_state[k], aligned_base) for k in range(6)]

    row = {
        "obj": obj_id,
        "category": category,
        "gt_type": joint_info["type"],
        "pred_type": pred_type,
        "type_correct": bool(type_correct),
        "axis_err_deg": float(axis_err),
        "pivot_err": float(pivot_err),
        "pred_range": float(pred_phi_K),
        "gt_range_normalized": float(gt_range_norm),
        "range_err": float(range_err),
        "move_iou_mean": float(move_iou_mean),
        "per_state_move_iou": [float(x) for x in per_state_iou],
        "base_iou": float(base_iou),
        "per_state_base_iou": [float(x) for x in per_state_base_iou],
        "stage_b_base_iou_raw": float(stage_b_base_iou),
        "align_iou": float(best_iou),
        "align_scale": float(M_col_norm),
        "align_scale_seed": float(seed_scale),
        "extent": float(extent),
        "center": [float(x) for x in center.tolist()],
        "joint_name": joint_name,
        "link_name": link_name,
        "moving_node_count_urdf": int(len(urdf_nodes)),
        "moving_node_count_effective": int(len(moving_nodes)),
        "pipeline_base_always_on_count": int(pipeline_base_target.sum()),
        "pipeline_union_count": int(pipeline_union.sum()),
    }
    return row


def main():
    results = []
    for obj, cat in OBJECTS:
        try:
            r = eval_object(obj, cat)
        except Exception as e:
            print(f"ERROR on {obj}: {e!r}")
            import traceback
            traceback.print_exc()
            r = {"obj": obj, "category": cat, "error": repr(e)}
        results.append(r)

    # Markdown table.
    def fmt(v, p=3):
        if v is None:
            return "-"
        if isinstance(v, bool):
            return "yes" if v else "no"
        if isinstance(v, float):
            if np.isnan(v):
                return "-"
            return f"{v:.{p}f}"
        return str(v)

    print("\n\n" + "=" * 90)
    print("RESULTS TABLE")
    print("=" * 90)
    header = ("| Object | Category | GT Type | Pred Type | Correct | Axis err (deg) | "
              "Pivot err | Range err | Move IoU | Base IoU | StageB base align IoU |")
    sep = "|" + "|".join(["---"] * 11) + "|"
    print(header)
    print(sep)
    for r in results:
        if "error" in r:
            print(f"| {r['obj']} | {r['category']} | ERROR | | | | | | | | |")
            continue
        print("| {obj} | {cat} | {gtt} | {pt} | {tc} | {axe} | {pe} | {re} | {miou} | {biou} | {align} |".format(
            obj=r["obj"], cat=r["category"], gtt=r["gt_type"], pt=r["pred_type"],
            tc=fmt(r["type_correct"]),
            axe=fmt(r["axis_err_deg"], 2),
            pe=fmt(r["pivot_err"], 4),
            re=fmt(r["range_err"], 4),
            miou=fmt(r["move_iou_mean"], 3),
            biou=fmt(r["base_iou"], 3),
            align=fmt(r["align_iou"], 3),
        ))

    print("\nPer-state move IoU:")
    for r in results:
        if "error" in r:
            continue
        psi = "  ".join(f"{x:.3f}" for x in r["per_state_move_iou"])
        print(f" {r['obj']}: [{psi}]")

    # json.dumps defaults to emitting "NaN" which is not valid JSON; coerce.
    def _sanitize(o):
        if isinstance(o, float):
            if np.isnan(o) or np.isinf(o):
                return None
            return o
        if isinstance(o, dict):
            return {k: _sanitize(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_sanitize(v) for v in o]
        return o

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(_sanitize(results), indent=2))
    print(f"\nWrote {OUT_JSON}")


if __name__ == "__main__":
    main()
