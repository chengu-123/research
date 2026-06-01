"""Differentiable Gaussian rendering of the warped canonical asset.

Per Stage D inner loop (pipeline.md section 9.1) the per-frame render is:

  base contribution:
    means_base    = xyz_canon                            (no warp)
    opacity_base  = opacity_canon * g * (1 - m)
    rotation_base = rot_canon
  move contribution:
    means_move    = R(psi, phi[t]) @ xyz_canon + T
    opacity_move  = opacity_canon * g * m
    rotation_move = quat_multiply(R_quat, rot_canon)     (wxyz convention)

Two-branch rendering (P1, before type commit):
  rgb_revolute   = rasterize(... using SE3_revolute(psi, phi_render_rev[t]))
  rgb_prismatic  = rasterize(... using SE3_prismatic(psi, phi_render_pri[t]))
  rgb[t]         = (1 - type_soft) * rgb_revolute + type_soft * rgb_prismatic

After type commit (P1 G1b end / P2): single-branch rendering with the
committed type_hard; ``type_soft`` does not appear, no soft blend.

The frozen TRELLIS D_GS produces ``Gaussian`` objects with ``.get_xyz``,
``.get_opacity`` (post-sigmoid + bias), ``.get_rotation``, ``.get_scaling``,
``._features_dc``. We do NOT mutate the Gaussian object; we call the
underlying ``diff_gaussian_rasterization.GaussianRasterizer`` directly with our warped
arrays as kwargs. This avoids any in-place edit on the frozen D_GS output
while keeping ``rendered_image`` differentiable w.r.t. all the warped
quantities (means3D, opacities, rotations).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Use TRELLIS/3DGS rasterization rather than FreeArt3D's diff_gauss package.
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

from .config import (
    H_PIXEL,
    TRELLIS_DGS_N_GAUSS_PER_VOXEL,
    W_PIXEL,
)
from .joint_ops import (
    JointParams,
    SE3_prismatic,
    SE3_revolute,
    quaternion_from_axis_angle,
    quaternion_multiply,
)


# =============================================================================
# Camera setup (locked-off; same for all 21 frames)
# =============================================================================

@dataclass
class StageDCameraConfig:
    """Locked-off camera for Stage D rendering.

    The canonical D_GS asset lives in world coordinates in [-0.5, 0.5]^3
    (TRELLIS convention; see decoder_gs.py:95 ``aabb=[-0.5,-0.5,-0.5,1,1,1]``).
    The camera is placed on a single fixed location and stays there across
    all 21 frames (Wan2.2 generates locked-off video; render must match).

    The camera MUST match the camera that produced ``s_0_pure`` (the
    user-input image). For PartNet inputs rendered by FreeArt3D's pipeline
    (``mine/pipelines/render.py``), use ``StageDCameraConfig.freeart3d_canonical()``.
    Mismatched camera makes L_first ill-defined and biases the W-RFSDS
    gradient toward 3D distortion that compensates for view error.

    fov_y_deg : Optional[float]
        Explicit vertical FoV in degrees. When ``None`` (default), the
        projection matrix computes ``fov_y`` from the image aspect ratio
        assuming square pixels. Set explicitly when the image pixel grid
        does NOT have a square-pixel aspect — e.g. the Stage A pipeline
        use the same calibrated FoV as the input renderer and pass the Stage A
        actual H/W.

    near / far must bracket the object (in [-0.5, 0.5]^3) plus camera
    distance; defaults 0.1 / 10.0 cover all reasonable poses including
    FreeArt3D's distance ~2.1.
    """
    pos: Tuple[float, float, float] = (0.0, 0.0, 2.0)
    look_at: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    up: Tuple[float, float, float] = (0.0, 1.0, 0.0)
    fov_x_deg: float = 30.0
    fov_y_deg: Optional[float] = None       # ★ explicit y FoV (None -> derive from aspect)
    image_h: int = H_PIXEL
    image_w: int = W_PIXEL
    near: float = 0.1
    far: float = 10.0
    bg_color: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    @classmethod
    def freeart3d_canonical(
        cls,
        image_h: int = H_PIXEL,
        image_w: int = W_PIXEL,
        object_scale: float = 1.0,
        object_center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> "StageDCameraConfig":
        """FreeArt3D's hardcoded rendering camera, expressed in TRELLIS world.

        Source: ``mine/pipelines/render.py:run_rendering()`` line 211-285 sets
            fov = 45 deg                    (Blender camera.angle, applies to both axes
                                             because original render is 800x800)
            azi = pi/8 = 22.5 deg           (training render azimuth)
            height = sin(pi/4) * distance   (elevation: 45 deg from horizontal)
            camera_distance = 2.1 * object_scale
        with the camera aimed at the object bbox center and
        ``object_scale = max(bbox extent)``. The TRELLIS-decoded asset occupies
        only a SUBSET of ``[-0.5, 0.5]^3`` (true max_extent < 1, centroid offset
        from origin), so the ``object_scale=1`` / ``object_center=origin``
        defaults render it too small and off-center. Callers that have the
        decoded canonical Gaussians should pass the measured bbox center and
        max_extent (see ``train.measure_canonical_object_bbox``) so the render
        matches the FreeArt3D-framed target.

        **TRELLIS canonical world up is +Z**, verified from:
            ``trellis/utils/render_utils.py:33``:
                ``extrinsics_look_at(orig, [0,0,0], [0, 0, 1])``   <-- up=+Z
            ``trellis/utils/render_utils.py:28-32`` (spherical orig):
                ``[sin(yaw)*cos(pitch), cos(yaw)*cos(pitch), sin(pitch)]``
                                                            ^-- third axis = height
        Same convention as Blender / FreeArt3D source. No axis swap needed.

        Image aspect: Stage A preserves the input aspect ratio. For the
        FreeArt3D default 800x800 render, the Wan 480P profile therefore
        produces a square output. Keep fov_x = fov_y = 45 deg and pass the
        Stage A actual H/W into this factory.
        """
        d = 2.1 * float(object_scale)           # distance scales with object extent
        azi = math.pi / 8.0                     # 22.5 deg azimuth
        cx, cy, cz = (
            float(object_center[0]),
            float(object_center[1]),
            float(object_center[2]),
        )
        # FreeArt3D's render.py: height = sin(45 deg) * distance,
        # x = sin(azi) * d, y = -cos(azi) * d, relative to the object center.
        # +Z up world (Blender = TRELLIS).
        x = cx + math.sin(azi) * d
        y = cy - math.cos(azi) * d
        z = cz + math.sin(math.pi / 4.0) * d

        return cls(
            pos=(x, y, z),
            look_at=(cx, cy, cz),
            up=(0.0, 0.0, 1.0),                 # ★ hardcoded +Z (TRELLIS canonical)
            fov_x_deg=45.0,
            fov_y_deg=45.0,
            image_h=int(image_h), image_w=int(image_w),
            near=0.1, far=10.0,
            bg_color=(0.0, 0.0, 0.0),
        )

    @classmethod
    def freeart3d_sampled(
        cls,
        azi_deg: float,
        ele_deg: float,
        image_h: int,
        image_w: int,
        object_scale: float,
        object_center: Tuple[float, float, float],
    ) -> "StageDCameraConfig":
        """Multi-view variant of ``freeart3d_canonical`` with explicit angles.

        Reproduces the EXACT same (non-spherical) position convention as
        ``freeart3d_canonical`` but with caller-chosen azimuth / elevation
        instead of the hardcoded 22.5 / 45 deg::

            d = 2.1 * object_scale
            x = cx + sin(azi) * d
            y = cy - cos(azi) * d
            z = cz + sin(ele) * d        # NOTE: x/y use the full d (no cos(ele))

        With ``azi_deg=22.5`` and ``ele_deg=45`` this returns a config EQUAL to
        ``freeart3d_canonical(image_h, image_w, object_scale, object_center)``
        (same pos, up, fov, near/far, bg, image size). near/far/bg constants are
        copied verbatim from ``freeart3d_canonical``.
        """
        d = 2.1 * float(object_scale)           # distance scales with object extent
        ar = math.radians(float(azi_deg))
        el = math.radians(float(ele_deg))
        cx, cy, cz = (
            float(object_center[0]),
            float(object_center[1]),
            float(object_center[2]),
        )
        # Same convention as freeart3d_canonical: x/y use the FULL d, the
        # elevation enters only z via sin(el) (there is NO cos(el) on x/y).
        x = cx + math.sin(ar) * d
        y = cy - math.cos(ar) * d
        z = cz + math.sin(el) * d

        return cls(
            pos=(x, y, z),
            look_at=(cx, cy, cz),
            up=(0.0, 0.0, 1.0),                 # ★ hardcoded +Z (TRELLIS canonical)
            fov_x_deg=45.0,
            fov_y_deg=45.0,
            image_h=int(image_h), image_w=int(image_w),
            near=0.1, far=10.0,
            bg_color=(0.0, 0.0, 0.0),
        )


def sample_camera_for_iter(
    rng: "torch.Generator",
    cfg,
    object_center: Tuple[float, float, float],
    object_scale: float,
    image_h: int,
    image_w: int,
    canonical_cfg: StageDCameraConfig,
) -> Tuple[StageDCameraConfig, bool]:
    """Pick the camera for one StageD iter: canonical or a random view.

    With probability ``cfg.mv_canonical_ratio`` returns the pre-built
    ``canonical_cfg`` (and ``is_canonical=True``); otherwise samples a fresh
    azimuth / elevation and returns ``freeart3d_sampled(...)`` with
    ``is_canonical=False``.

    Sampling ranges (degrees)::

        azi in [mv_azi_min_deg, mv_azi_max_deg]
        ele in [mv_ele_min_deg, mv_ele_max_deg]

    Parameters
    ----------
    rng : torch.Generator (cpu)
        Dedicated CPU generator so camera sampling is deterministic and
        decoupled from the model/training RNG.
    cfg : StageDConfig
        Reads ``mv_canonical_ratio``, ``mv_azi_min_deg``, ``mv_azi_max_deg``,
        ``mv_ele_min_deg``, ``mv_ele_max_deg``.
    canonical_cfg : StageDCameraConfig
        The pre-built canonical camera, returned as-is on canonical draws.

    Returns
    -------
    (camera_cfg, is_canonical) : Tuple[StageDCameraConfig, bool]
    """
    u = torch.rand(1, generator=rng).item()
    if u < float(cfg.mv_canonical_ratio):
        return canonical_cfg, True
    azi = float(cfg.mv_azi_min_deg) + torch.rand(1, generator=rng).item() * (
        float(cfg.mv_azi_max_deg) - float(cfg.mv_azi_min_deg)
    )
    ele = float(cfg.mv_ele_min_deg) + torch.rand(1, generator=rng).item() * (
        float(cfg.mv_ele_max_deg) - float(cfg.mv_ele_min_deg)
    )
    sampled_cfg = StageDCameraConfig.freeart3d_sampled(
        azi, ele, image_h, image_w, object_scale, object_center,
    )
    return sampled_cfg, False


def _build_extrinsics(camera_cfg: StageDCameraConfig,
                      device: torch.device,
                      dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """4x4 world-to-camera transform from (pos, look_at, up).

    Matches ``utils3d.torch.extrinsics_look_at`` used by TRELLIS:
    camera-space +Z points from the camera to the target. This is the
    convention expected by TRELLIS' ``intrinsics_to_projection`` and
    ``diff_gaussian_rasterization`` (projection ``w`` must be positive for visible points).
    Returned matrix transforms world points to camera-local coordinates as
    ``p_cam = R_wc @ p_world + t_wc``.
    """
    pos = torch.tensor(camera_cfg.pos, device=device, dtype=dtype)
    target = torch.tensor(camera_cfg.look_at, device=device, dtype=dtype)
    up = torch.tensor(camera_cfg.up, device=device, dtype=dtype)

    forward = target - pos
    forward = forward / forward.norm().clamp_min(1.0e-6)
    right = torch.linalg.cross(forward, up)
    right = right / right.norm().clamp_min(1.0e-6)
    up_corrected = torch.linalg.cross(forward, right)

    # Camera-axes-in-world rows; rotation is world -> camera.
    R_wc = torch.stack([right, up_corrected, forward], dim=0)         # [3, 3]
    t_wc = -R_wc @ pos                                                  # [3]

    extrinsics = torch.eye(4, device=device, dtype=dtype)
    extrinsics[:3, :3] = R_wc
    extrinsics[:3, 3] = t_wc
    return extrinsics


def _build_proj_matrix(camera_cfg: StageDCameraConfig,
                        device: torch.device,
                        dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """OpenGL-style perspective projection for diff_gauss.

    When ``camera_cfg.fov_y_deg is None`` (square-pixel default), computes
    ``fov_y`` from the image aspect via ``tan(fov_y/2) = tan(fov_x/2) * H/W``.
    When ``fov_y_deg`` is set explicitly, uses it directly — this is needed
    when the image grid does NOT have square pixels.

    Maps z in [near, far] to [0, 1] (TRELLIS rasterization convention; see
    trellis/renderers/gaussian_render.py:30-47).
    """
    fovx = math.radians(camera_cfg.fov_x_deg)
    tan_fov_x = math.tan(fovx * 0.5)
    if camera_cfg.fov_y_deg is None:
        # ★ Default: square-pixel assumption — derive fov_y from aspect.
        aspect_h_over_w = float(camera_cfg.image_h) / float(camera_cfg.image_w)
        tan_fov_y = tan_fov_x * aspect_h_over_w
        fovy = 2.0 * math.atan(tan_fov_y)
    else:
        # Explicit fov_y for calibrated non-square-pixel grids.
        fovy = math.radians(float(camera_cfg.fov_y_deg))
        tan_fov_y = math.tan(fovy * 0.5)

    near = float(camera_cfg.near)
    far = float(camera_cfg.far)

    P = torch.zeros(4, 4, device=device, dtype=dtype)
    # TRELLIS rasterization uses a non-standard layout where intrinsics is normalized
    # (fx, fy in [0, 1] of image dim). Here we go via FoV -> direct matrix.
    P[0, 0] = 1.0 / tan_fov_x
    P[1, 1] = 1.0 / tan_fov_y
    P[2, 2] = far / (far - near)
    P[2, 3] = -(far * near) / (far - near)
    P[3, 2] = 1.0
    return P, fovx, fovy


@dataclass
class LockedCamera:
    """Pre-built camera tensors used by the rasterizer.

    All fields are constructed once at Stage D init and reused for every
    iter / every frame (camera is locked-off).
    """
    image_h: int
    image_w: int
    bg_color: torch.Tensor                     # [3] on device
    tan_fov_x: float
    tan_fov_y: float
    view_matrix: torch.Tensor                  # [4, 4]  (extrinsics.T)
    proj_matrix_full: torch.Tensor             # [4, 4]  (P @ extrinsics).T
    camera_center: torch.Tensor                # [3]     world-space camera position


def build_locked_camera(camera_cfg: StageDCameraConfig,
                        device: torch.device,
                        dtype: torch.dtype = torch.float32) -> LockedCamera:
    """Construct the LockedCamera struct from a StageDCameraConfig.

    Matches the camera_dict assembly in TRELLIS
    ``GaussianRenderer.render`` (gaussian_render.py:216-236):

        world_view_transform = extrinsics.T
        full_proj_transform  = (perspective @ extrinsics).T
        camera_center        = inverse(extrinsics)[:3, 3]
    """
    extrinsics = _build_extrinsics(camera_cfg, device, dtype)
    P, fovx, fovy = _build_proj_matrix(camera_cfg, device, dtype)
    tan_fov_x = math.tan(fovx * 0.5)
    tan_fov_y = math.tan(fovy * 0.5)

    view_matrix = extrinsics.T.contiguous()
    proj_matrix_full = (P @ extrinsics).T.contiguous()
    # Camera center = inverse(extrinsics)[:3, 3] = -R_wc^T @ t_wc
    R_wc = extrinsics[:3, :3]
    t_wc = extrinsics[:3, 3]
    camera_center = -R_wc.T @ t_wc

    bg = torch.tensor(camera_cfg.bg_color, device=device, dtype=dtype)

    return LockedCamera(
        image_h=int(camera_cfg.image_h),
        image_w=int(camera_cfg.image_w),
        bg_color=bg,
        tan_fov_x=float(tan_fov_x),
        tan_fov_y=float(tan_fov_y),
        view_matrix=view_matrix,
        proj_matrix_full=proj_matrix_full,
        camera_center=camera_center,
    )


# =============================================================================
# D_GS wrapper that also returns parent_idx
# =============================================================================

class DGSWithParent(nn.Module):
    """Wrap the frozen TRELLIS D_GS to additionally return ``parent_idx``.

    ``decoder_gs.py:108-110`` lays out per-Gaussian outputs as
        ``_xyz = xyz.unsqueeze(1) + offset``   # [N_voxel, 32, 3]
        ``_xyz.flatten(0, 1)``                  # [N_voxel * 32, 3]
    so ``gauss_idx = voxel_idx * 32 + g``, which means
        ``parent_idx[i*32 + g] = i``
    i.e. ``parent_idx = arange(N_voxel).repeat_interleave(32)`` — trivial,
    no reverse-engineering from positions needed (method.md D-v3.8).
    """

    def __init__(self, d_gs_frozen: nn.Module,
                 n_gauss_per_voxel: int = TRELLIS_DGS_N_GAUSS_PER_VOXEL) -> None:
        super().__init__()
        self.d_gs = d_gs_frozen
        for p in self.d_gs.parameters():
            p.requires_grad_(False)
        self.d_gs.eval()
        self.n_gauss_per_voxel = int(n_gauss_per_voxel)

    def forward(self, sparse_in):
        """Run the frozen D_GS and emit (gauss, parent_idx).

        Parameters
        ----------
        sparse_in : sp.SparseTensor
            Coords [N_voxel, 4] (batch col first), feats [N_voxel, 8].

        Returns
        -------
        gauss : Gaussian
            TRELLIS Gaussian object (one batch element). Access via
            ``gauss.get_xyz``, ``gauss.get_opacity``, etc.
        parent_idx : Tensor [N_voxel * n_gauss_per_voxel] long
        """
        gauss_list = self.d_gs(sparse_in)
        if len(gauss_list) != 1:
            raise ValueError(
                f"DGSWithParent expects batch=1; got {len(gauss_list)} Gaussians"
            )
        gauss = gauss_list[0]
        n_voxel = sparse_in.coords[sparse_in.layout[0]].shape[0]
        parent_idx = (
            torch.arange(n_voxel, device=sparse_in.coords.device)
                 .repeat_interleave(self.n_gauss_per_voxel)
        )
        return gauss, parent_idx


# =============================================================================
# Single-frame rasterize given warped Gaussian state
# =============================================================================

def _build_raster_settings(
    camera: LockedCamera,
    sh_degree: int = 0,
    scale_modifier: float = 1.0,
) -> "GaussianRasterizationSettings":
    """Build a GaussianRasterizationSettings tuple.

    Pulled out so the train loop / render_21_with_warp can build it ONCE
    per render call instead of 42 times (★ S3 fix: rasterizer reuse).
    """
    return GaussianRasterizationSettings(
        image_height=camera.image_h,
        image_width=camera.image_w,
        tanfovx=camera.tan_fov_x,
        tanfovy=camera.tan_fov_y,
        bg=camera.bg_color,
        scale_modifier=float(scale_modifier),
        viewmatrix=camera.view_matrix,
        projmatrix=camera.proj_matrix_full,
        sh_degree=int(sh_degree),
        campos=camera.camera_center,
        prefiltered=False,
        debug=False,
    )


def _rasterize_one_frame(
    means3D: torch.Tensor,
    opacities: torch.Tensor,
    rotations: torch.Tensor,
    scales: torch.Tensor,
    sh_coeffs: torch.Tensor,
    rasterizer: "GaussianRasterizer",
) -> torch.Tensor:
    """Call a pre-built ``GaussianRasterizer`` with warped Gaussian state.

    ★ S3 fix: ``rasterizer`` is passed in (built once per
    ``render_21_with_warp`` call) instead of constructed per-frame. Per
    iter, this avoids 42 redundant constructions of identical objects.

    C3 caveat: rasterizer return tuple length varies across versions
    (3-tuple ``(rendered, radii, depth)`` in some, 4-tuple ``(rendered,
    depth, alpha, radii)`` in others). We unpack the rendered image as the
    first element (universal) and discard the rest with ``*_``.

    Parameters
    ----------
    means3D : Tensor [N, 3]     world-space positions (already warped if move).
    opacities : Tensor [N, 1]   in [0, 1], already gated.
    rotations : Tensor [N, 4]   wxyz quaternions.
    scales : Tensor [N, 3]      from gauss.get_scaling (canonical).
    sh_coeffs : Tensor [N, 1, 3]   SH0 (DC) only since sh_degree=0 default.
    rasterizer : GaussianRasterizer (pre-built).

    Returns
    -------
    rgb : Tensor [3, H, W]
        Raw rasterizer output. Caller is responsible for clamping to [0, 1]
        if needed (★ S1 fix: clamp moved to ``render_21_with_warp`` end).
    """
    screenspace = torch.zeros_like(means3D, requires_grad=True)

    out = rasterizer(
        means3D=means3D,
        means2D=screenspace,
        shs=sh_coeffs,
        colors_precomp=None,
        opacities=opacities,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None,
    )
    # The rasterizer returns a tuple; the first element is always the rendered
    # image. We accept any length to be version-tolerant (C3 caveat).
    rendered = out[0] if isinstance(out, tuple) else out
    return rendered                          # [3, H, W]


def render_static_gaussians(
    xyz: torch.Tensor,
    opacity: torch.Tensor,
    rotation: torch.Tensor,
    scale: torch.Tensor,
    sh: torch.Tensor,
    camera: LockedCamera,
    opacity_scale: Optional[torch.Tensor] = None,
    sh_degree: int = 0,
) -> torch.Tensor:
    """Render the canonical Gaussian set without gate split or joint warp."""
    if opacity_scale is not None:
        if opacity_scale.ndim != 1 or opacity_scale.shape[0] != opacity.shape[0]:
            raise ValueError(
                f"opacity_scale must be [{opacity.shape[0]}], got {tuple(opacity_scale.shape)}"
            )
        opacity = opacity * opacity_scale.to(device=opacity.device, dtype=opacity.dtype).unsqueeze(-1)
    raster_settings = _build_raster_settings(camera, sh_degree=sh_degree)
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    return _rasterize_one_frame(
        xyz,
        opacity,
        rotation,
        scale,
        sh,
        rasterizer,
    ).clamp(0.0, 1.0)


# =============================================================================
# 21-frame render with revolute / prismatic two-branch warp
# =============================================================================

@dataclass
class RenderInputs:
    """Constants for ``render_21_with_warp`` that don't change across frames.

    Built once per iter after D_GS decode + per-voxel gate computation.
    """
    xyz_canon: torch.Tensor                  # [N_gauss, 3]
    opacity_canon: torch.Tensor              # [N_gauss, 1]
    rot_canon: torch.Tensor                  # [N_gauss, 4] wxyz
    scale_canon: torch.Tensor                # [N_gauss, 3]
    sh_canon: torch.Tensor                   # [N_gauss, 1, 3]
    g_per_gauss: torch.Tensor                # [N_gauss]   in [0, 1] (post STE)
    m_per_gauss: torch.Tensor                # [N_gauss]   in [0, 1] (post STE)


def _warp_one_state(
    inputs: RenderInputs,
    R: torch.Tensor,
    t: torch.Tensor,
    R_quat_or_none: Optional[torch.Tensor],
    move_only: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build per-frame (means, opacities, rotations, scales, sh) for one state.

    Concatenates base and move contributions: each canonical Gaussian
    appears *twice* in the per-frame Gaussian set, gated by ``(1 - m)`` and
    ``m`` respectively. The base copy stays at ``xyz_canon`` with
    ``rot_canon``; the move copy is at ``R @ xyz_canon + t`` with
    ``quaternion_multiply(R_quat, rot_canon)``.

    For prismatic motion ``R`` is identity, ``R_quat_or_none`` may be
    ``None`` and we skip the quaternion multiplication entirely.
    """
    xyz = inputs.xyz_canon
    op_canon_1d = inputs.opacity_canon.squeeze(-1)         # [N_gauss]

    # Base
    means_base = xyz                                       # [N_gauss, 3]
    opacity_base = (
        op_canon_1d * inputs.g_per_gauss * (1.0 - inputs.m_per_gauss)
    ).unsqueeze(-1)                                        # [N_gauss, 1]
    if move_only:
        opacity_base = torch.zeros_like(opacity_base)
    rot_base = inputs.rot_canon

    # Move
    means_move = xyz @ R.T + t                             # [N_gauss, 3]
    opacity_move = (
        op_canon_1d * inputs.g_per_gauss * inputs.m_per_gauss
    ).unsqueeze(-1)
    if R_quat_or_none is not None:
        rot_move = quaternion_multiply(R_quat_or_none, inputs.rot_canon)
    else:
        rot_move = inputs.rot_canon

    means = torch.cat([means_base, means_move], dim=0)     # [2N_gauss, 3]
    opacities = torch.cat([opacity_base, opacity_move], dim=0)
    rotations = torch.cat([rot_base, rot_move], dim=0)
    scales = torch.cat([inputs.scale_canon, inputs.scale_canon], dim=0)
    sh = torch.cat([inputs.sh_canon, inputs.sh_canon], dim=0)
    return means, opacities, rotations, scales, sh


def render_21_with_warp(
    inputs: RenderInputs,
    joint: JointParams,
    phi_render_rev: torch.Tensor,            # [F=21] signed radians
    phi_render_pri: torch.Tensor,            # [F=21] signed world units
    camera: LockedCamera,
    type_soft: Optional[torch.Tensor],        # scalar in [0, 1], or None for committed
    committed_type: Optional[str] = None,     # 'revolute' / 'prismatic' / None
    sh_degree: int = 0,
) -> torch.Tensor:
    """Render 21 frames with the joint-warped move part.

    Two modes, controlled by ``committed_type``:

      * ``None`` (P1 before type vote): render both revolute and prismatic
        branches, blend per-frame by ``type_soft`` (sigmoid of psi.type_logit).
        ``rgb[t] = (1 - type_soft) * rev[t] + type_soft * pri[t]``.

      * ``'revolute'`` / ``'prismatic'`` (after type commit): only render the
        committed branch. Saves ~50% rasterize calls.

    Parameters
    ----------
    inputs : RenderInputs                    canonical Gaussian state + gates
    joint : JointParams                      axis / origin / type_logit / limits
    phi_render_rev : Tensor [F]              signed radians (NEW.1)
    phi_render_pri : Tensor [F]              signed world units (NEW.1)
    camera : LockedCamera
    type_soft : Tensor scalar in [0, 1] or None
    committed_type : str or None
    sh_degree : int   default 0 (DC only)

    Returns
    -------
    rgb_T3HW : Tensor [F, 3, H, W]           in [0, 1] (rasterizer output range)
    """
    F = int(phi_render_rev.shape[0])
    if phi_render_pri.shape[0] != F:
        raise ValueError(
            "phi_render_rev / phi_render_pri must have same length"
        )

    # ★ S3 fix: build rasterizer settings + instance ONCE for the whole
    # render call. camera is locked-off and sh_degree is fixed, so all 42
    # rasterize calls below can share this. Saves O(42) python-side object
    # construction per train iter.
    raster_settings = _build_raster_settings(camera, sh_degree=sh_degree)
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # Gradient-checkpoint each rasterize. In the uncommitted phase this loop
    # issues 42 grad-retaining rasterize calls (rev + pri x 21 frames); the
    # rasterizer's saved geom/binning/img buffers for all of them together
    # exhaust the GPU once both A14B experts (~56 GB) are resident. Checkpoint
    # keeps only one frame's buffers live and recomputes the rest in backward.
    # The rasterize is far cheaper than the dual-expert Wan DiT, so the
    # recompute cost is modest. Closes over the shared rasterizer.
    from torch.utils.checkpoint import checkpoint

    def _ckpt_rasterize(means, op, rot, sc, sh):
        return _rasterize_one_frame(means, op, rot, sc, sh, rasterizer)

    rgb_frames: List[torch.Tensor] = []

    for t_idx in range(F):
        # Build the two SE(3) candidates (skip whichever is unused under commit).
        if committed_type is None or committed_type == "revolute":
            T_rev = SE3_revolute(joint.axis, joint.origin, phi_render_rev[t_idx])
            R_rev = T_rev[:3, :3]
            t_rev = T_rev[:3, 3]
            R_quat_rev = quaternion_from_axis_angle(joint.axis, phi_render_rev[t_idx])
            means_r, op_r, rot_r, sc_r, sh_r = _warp_one_state(
                inputs, R_rev, t_rev, R_quat_rev,
            )
            rgb_rev = checkpoint(
                _ckpt_rasterize, means_r, op_r, rot_r, sc_r, sh_r,
                use_reentrant=False,
            )                                              # [3, H, W]
        else:
            rgb_rev = None

        if committed_type is None or committed_type == "prismatic":
            T_pri = SE3_prismatic(joint.axis, phi_render_pri[t_idx])
            R_pri = T_pri[:3, :3]                          # identity (3, 3)
            t_pri = T_pri[:3, 3]
            means_p, op_p, rot_p, sc_p, sh_p = _warp_one_state(
                inputs, R_pri, t_pri, R_quat_or_none=None,
            )
            rgb_pri = checkpoint(
                _ckpt_rasterize, means_p, op_p, rot_p, sc_p, sh_p,
                use_reentrant=False,
            )                                              # [3, H, W]
        else:
            rgb_pri = None

        # Combine according to mode.
        if committed_type is None:
            if type_soft is None:
                raise ValueError(
                    "render_21_with_warp: type_soft is required when "
                    "committed_type is None (P1 two-branch blend)"
                )
            rgb_t = (1.0 - type_soft) * rgb_rev + type_soft * rgb_pri
        elif committed_type == "revolute":
            rgb_t = rgb_rev
        elif committed_type == "prismatic":
            rgb_t = rgb_pri
        else:
            raise ValueError(f"unknown committed_type: {committed_type!r}")

        rgb_frames.append(rgb_t)

    rgb_T3HW = torch.stack(rgb_frames, dim=0)             # [F, 3, H, W]
    # ★ S1 fix: clamp raw rasterizer output to [0, 1] before downstream loss.
    # Rasterized per-pixel color = sum(T_i * alpha_i * c_i) where c_i comes
    # from SH-decoded RGB; not guaranteed to stay in [0, 1]. LPIPS and Wan
    # VAE both assume in-range input; out-of-range values produce undefined
    # behavior in VGG and out-of-distribution input to the VAE.
    rgb_T3HW = rgb_T3HW.clamp(0.0, 1.0)
    return rgb_T3HW


def render_move_silhouettes(
    inputs: RenderInputs,
    joint: JointParams,
    phi_render_rev: torch.Tensor,
    phi_render_pri: torch.Tensor,
    camera: LockedCamera,
    frame_indices: Tuple[int, ...],
    committed_type: str,
    sh_degree: int = 0,
) -> torch.Tensor:
    """Render move-branch silhouettes for selected endpoint frames.

    Returns [K, 1, H, W]. Base rows remain in the rasterizer call with zero
    opacity so tensor layout and graph structure match the normal renderer.
    """
    if committed_type not in ("revolute", "prismatic"):
        raise ValueError(
            "render_move_silhouettes requires committed_type "
            f"'revolute' or 'prismatic', got {committed_type!r}"
        )
    F = int(phi_render_rev.shape[0])
    if phi_render_pri.shape[0] != F:
        raise ValueError(
            "phi_render_rev / phi_render_pri must have same length"
        )
    if len(frame_indices) == 0:
        raise ValueError("frame_indices must be non-empty")

    raster_settings = _build_raster_settings(camera, sh_degree=sh_degree)
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    from torch.utils.checkpoint import checkpoint

    def _ckpt_rasterize(means, op, rot, sc, sh):
        return _rasterize_one_frame(means, op, rot, sc, sh, rasterizer)

    silhouettes: List[torch.Tensor] = []
    for frame_idx_raw in frame_indices:
        frame_idx = int(frame_idx_raw)
        if frame_idx < 0 or frame_idx >= F:
            raise ValueError(f"frame index {frame_idx} out of range [0, {F})")
        if committed_type == "revolute":
            T = SE3_revolute(joint.axis, joint.origin, phi_render_rev[frame_idx])
            R = T[:3, :3]
            t = T[:3, 3]
            R_quat = quaternion_from_axis_angle(joint.axis, phi_render_rev[frame_idx])
            means, op, rot, sc, sh = _warp_one_state(
                inputs, R, t, R_quat, move_only=True,
            )
        else:
            T = SE3_prismatic(joint.axis, phi_render_pri[frame_idx])
            R = T[:3, :3]
            t = T[:3, 3]
            means, op, rot, sc, sh = _warp_one_state(
                inputs, R, t, R_quat_or_none=None, move_only=True,
            )
        rgb = checkpoint(
            _ckpt_rasterize, means, op, rot, sc, sh,
            use_reentrant=False,
        ).clamp(0.0, 1.0)
        silhouettes.append(rgb.abs().sum(dim=0, keepdim=True).clamp(0.0, 1.0))
    return torch.stack(silhouettes, dim=0)


__all__ = [
    "StageDCameraConfig", "sample_camera_for_iter",
    "LockedCamera", "build_locked_camera",
    "DGSWithParent", "RenderInputs", "render_21_with_warp",
    "render_static_gaussians", "render_move_silhouettes",
]
