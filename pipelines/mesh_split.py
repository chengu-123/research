"""Vertex-level mesh split by per-part soft masks.

Given a single TRELLIS mesh in the canonical world frame and the
``M_base`` / ``M_move`` soft masks from SAJO (on the 64^3 voxel grid),
this module produces two disjoint sub-meshes. Triangles straddling the
base/move boundary are cut along the ``m_b = m_m`` iso-surface via
per-edge linear interpolation.

The world-frame convention matches :mod:`pipelines.sajo.warp`:
``world = (i, j, k) / 63 - 0.5``. Mesh vertex coordinates are assumed
to already live in ``[-0.5, 0.5]``; TRELLIS emits meshes in that frame
(see ``trellis.utils.postprocessing_utils.to_glb``).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


def _trilinear_interp_mask(
    vertices_world: np.ndarray,
    mask: np.ndarray,
    resolution: int = 64,
) -> np.ndarray:
    """Sample a ``(R, R, R)`` scalar grid at world-coord vertices.

    Parameters
    ----------
    vertices_world : (V, 3)
        World coordinates in ``[-0.5, 0.5]``. Points outside the grid
        are clamped (instead of zero-padded) so that boundary vertices
        still get a well-defined classification.
    mask : (R, R, R)
        Scalar field, e.g. ``M_base`` or ``M_move``.
    """
    R = int(resolution)
    idx = (vertices_world + 0.5) * float(R - 1)
    idx = np.clip(idx, 0.0, R - 1 - 1e-6)                    # strictly inside

    i0 = np.floor(idx[:, 0]).astype(np.int64)
    j0 = np.floor(idx[:, 1]).astype(np.int64)
    k0 = np.floor(idx[:, 2]).astype(np.int64)
    i1 = np.minimum(i0 + 1, R - 1)
    j1 = np.minimum(j0 + 1, R - 1)
    k1 = np.minimum(k0 + 1, R - 1)

    fi = idx[:, 0] - i0
    fj = idx[:, 1] - j0
    fk = idx[:, 2] - k0

    def _g(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        return mask[a, b, c]

    c000 = _g(i0, j0, k0); c100 = _g(i1, j0, k0)
    c010 = _g(i0, j1, k0); c110 = _g(i1, j1, k0)
    c001 = _g(i0, j0, k1); c101 = _g(i1, j0, k1)
    c011 = _g(i0, j1, k1); c111 = _g(i1, j1, k1)

    c00 = c000 * (1 - fi) + c100 * fi
    c10 = c010 * (1 - fi) + c110 * fi
    c01 = c001 * (1 - fi) + c101 * fi
    c11 = c011 * (1 - fi) + c111 * fi

    c0 = c00 * (1 - fj) + c10 * fj
    c1 = c01 * (1 - fj) + c11 * fj

    return c0 * (1 - fk) + c1 * fk


def trilinear_interp_vertex_masks(
    vertices_world: np.ndarray,
    M_base: np.ndarray,
    M_move: np.ndarray,
    resolution: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return per-vertex ``(m_b, m_m)`` scalar arrays."""
    m_b = _trilinear_interp_mask(vertices_world, M_base, resolution)
    m_m = _trilinear_interp_mask(vertices_world, M_move, resolution)
    return m_b.astype(np.float32), m_m.astype(np.float32)


def _edge_cut_t(m_a: float, m_b_other: float, m_base_a: float, m_base_b: float,
                m_move_a: float, m_move_b: float) -> float:
    """Solve for the parameter ``t in [0, 1]`` along an edge where the
    iso-surface ``m_base(t) = m_move(t)`` crosses.

    Parametrization: ``f(t) = (m_base_a + t * (m_base_b - m_base_a))
                              - (m_move_a + t * (m_move_b - m_move_a))``
    Linear in t with root ``t = (m_move_a - m_base_a) / (delta_base - delta_move)``.
    """
    num = m_move_a - m_base_a
    den = (m_base_b - m_base_a) - (m_move_b - m_move_a)
    if abs(den) < 1e-12:
        return 0.5
    t = float(num / den)
    if t < 0.0:
        return 0.0
    if t > 1.0:
        return 1.0
    return t


def split_mesh_by_masks(
    mesh: "trimesh.Trimesh",
    M_base: np.ndarray,
    M_move: np.ndarray,
    resolution: int = 64,
) -> Tuple["trimesh.Trimesh", "trimesh.Trimesh"]:
    """Split a single mesh into two by per-vertex base/move classification.

    Triangles are partitioned as follows:
    - all-base -> goes into the base mesh
    - all-move -> goes into the move mesh
    - mixed (2-1 or 1-2) -> each edge crossing the iso-surface is cut at
      the interpolation parameter where ``m_base == m_move``; the
      resulting polygons are re-triangulated and distributed to both
      meshes.

    The output meshes share no vertices by construction (new vertices
    are inserted at the cut points, and per-part vertex arrays are
    independent).
    """
    import trimesh

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    m_b, m_m = trilinear_interp_vertex_masks(vertices, M_base, M_move, resolution)
    is_base = m_b >= m_m                                     # (V,)

    base_verts: List[np.ndarray] = [vertices.copy()]
    move_verts: List[np.ndarray] = [vertices.copy()]
    base_faces: List[List[int]] = []
    move_faces: List[List[int]] = []

    # Each mesh receives a COPY of all vertices so that the original face
    # indices remain valid; new cut-points are appended per-mesh.
    base_vidx_next = vertices.shape[0]
    move_vidx_next = vertices.shape[0]

    def _add_base_vertex(p: np.ndarray) -> int:
        nonlocal base_vidx_next
        base_verts.append(p.reshape(1, 3))
        idx = base_vidx_next
        base_vidx_next += 1
        return idx

    def _add_move_vertex(p: np.ndarray) -> int:
        nonlocal move_vidx_next
        move_verts.append(p.reshape(1, 3))
        idx = move_vidx_next
        move_vidx_next += 1
        return idx

    for tri in faces:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        tags = (is_base[a], is_base[b], is_base[c])
        nb = int(tags[0]) + int(tags[1]) + int(tags[2])

        if nb == 3:
            base_faces.append([a, b, c])
            continue
        if nb == 0:
            move_faces.append([a, b, c])
            continue

        # Mixed triangle: we need to cut. Rotate so that `a` is the odd vertex
        # and `b`, `c` are the two vertices of the same side.
        if nb == 1:
            odd_side_is_base = True
        else:
            odd_side_is_base = False  # nb == 2

        if odd_side_is_base:
            base_idx_set = [i for i, t in zip((a, b, c), tags) if t]
            move_idx_set = [i for i, t in zip((a, b, c), tags) if not t]
            odd = base_idx_set[0]
            same = move_idx_set                              # length 2
        else:
            base_idx_set = [i for i, t in zip((a, b, c), tags) if t]
            move_idx_set = [i for i, t in zip((a, b, c), tags) if not t]
            odd = move_idx_set[0]
            same = base_idx_set                              # length 2

        def _ensure_order(odd_v: int, same_v: Tuple[int, int]) -> Tuple[int, int, int]:
            # Preserve winding: find the two edges (odd, same[0]) and (odd, same[1])
            # in their original triangle order so the output normals match.
            cyc = [a, b, c]
            idx_odd = cyc.index(odd_v)
            s0, s1 = cyc[(idx_odd + 1) % 3], cyc[(idx_odd + 2) % 3]
            return odd_v, s0, s1

        odd_v, s0, s1 = _ensure_order(odd, tuple(same))

        # Compute cut points on edges (odd, s0) and (odd, s1).
        def _cut(u: int, v: int) -> Tuple[np.ndarray, float]:
            t = _edge_cut_t(
                0.0, 0.0,
                m_b[u], m_b[v],
                m_m[u], m_m[v],
            )
            p = vertices[u] * (1.0 - t) + vertices[v] * t
            return p, t

        p1, _ = _cut(odd_v, s0)
        p2, _ = _cut(odd_v, s1)

        # Register the two cut-points in both meshes so triangles in
        # each mesh have valid vertex indices.
        p1_base = _add_base_vertex(p1); p2_base = _add_base_vertex(p2)
        p1_move = _add_move_vertex(p1); p2_move = _add_move_vertex(p2)

        if odd_side_is_base:
            # The single base vertex (odd_v) + cut points form the base tri.
            base_faces.append([odd_v, p1_base, p2_base])
            # The move side is a quadrilateral (s0, s1, p2, p1); split into
            # two triangles preserving winding.
            move_faces.append([s0, s1, p2_move])
            move_faces.append([s0, p2_move, p1_move])
        else:
            move_faces.append([odd_v, p1_move, p2_move])
            base_faces.append([s0, s1, p2_base])
            base_faces.append([s0, p2_base, p1_base])

    base_vertices = np.concatenate(base_verts, axis=0) if base_verts else vertices
    move_vertices = np.concatenate(move_verts, axis=0) if move_verts else vertices
    base_face_arr = np.asarray(base_faces, dtype=np.int64) if base_faces else np.zeros((0, 3), dtype=np.int64)
    move_face_arr = np.asarray(move_faces, dtype=np.int64) if move_faces else np.zeros((0, 3), dtype=np.int64)

    base_mesh = trimesh.Trimesh(vertices=base_vertices, faces=base_face_arr, process=False)
    move_mesh = trimesh.Trimesh(vertices=move_vertices, faces=move_face_arr, process=False)

    # Drop orphaned vertices from each mesh so the output is compact.
    base_mesh.update_faces(base_mesh.unique_faces())
    base_mesh.remove_unreferenced_vertices()
    move_mesh.update_faces(move_mesh.unique_faces())
    move_mesh.remove_unreferenced_vertices()

    return base_mesh, move_mesh
