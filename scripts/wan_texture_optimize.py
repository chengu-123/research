"""Texture-only optimization for FreeArt3D outputs guided by Wan InP frames.

This stage freezes the FreeArt3D geometry, part split, and trajectory. It only
updates embedded base-color texture atlases in the exported GLB files.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import struct
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import imageio.v2 as imageio
import nvdiffrast.torch as dr
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from pygltflib import BufferView, GLTF2


GLB_TO_FREEART3D_WORLD = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float32,
)

GLTF_COMPONENT_DTYPES = {
    5121: np.uint8,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}

GLTF_TYPE_COUNTS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
}


@dataclass(frozen=True)
class MeshPart:
    positions: torch.Tensor
    uvs: torch.Tensor
    faces: torch.Tensor
    material_id: int


@dataclass(frozen=True)
class StateMesh:
    path: Path
    parts: Tuple[MeshPart, ...]


@dataclass(frozen=True)
class CameraSpec:
    source: torch.Tensor
    target: torch.Tensor
    fov_degrees: float
    near: float
    far: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize FreeArt3D GLB texture atlases using Wan InP keyframes."
    )
    parser.add_argument("--origin_dir", required=True, help="FreeArt3D object output directory.")
    parser.add_argument("--wan_dir", required=True, help="Stage A Fun-InP output directory.")
    parser.add_argument("--out_dir", required=True, help="Output directory for optimized textures.")
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.04)
    parser.add_argument("--lambda_tex", type=float, default=0.02)
    parser.add_argument("--lambda_tv", type=float, default=0.001)
    parser.add_argument("--residual_size", type=int, default=128)
    parser.add_argument("--max_delta", type=float, default=0.16)
    parser.add_argument("--frame_indices", default="0,4,8,12,16,20")
    parser.add_argument("--states", default="0,1,2,3,4,5")
    parser.add_argument("--fov", type=float, default=45.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--log_interval", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def read_glb_chunks(path: Path) -> Tuple[dict, bytes]:
    data = path.read_bytes()
    magic, version, _total = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF" or version != 2:
        raise ValueError(f"not a binary glTF 2.0 file: {path}")
    offset = 12
    gltf = None
    binary = None
    while offset + 8 <= len(data):
        length, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        chunk = data[offset:offset + length]
        offset += length
        if chunk_type == 0x4E4F534A:
            gltf = json.loads(chunk.decode("utf-8"))
        elif chunk_type == 0x004E4942:
            binary = chunk
    if gltf is None or binary is None:
        raise ValueError(f"missing JSON or BIN chunk in {path}")
    return gltf, binary


def accessor_array(gltf: dict, binary: bytes, accessor_index: int) -> np.ndarray:
    accessor = gltf["accessors"][int(accessor_index)]
    view = gltf["bufferViews"][int(accessor["bufferView"])]
    dtype = GLTF_COMPONENT_DTYPES[int(accessor["componentType"])]
    elem_count = GLTF_TYPE_COUNTS[str(accessor["type"])]
    count = int(accessor["count"])
    accessor_offset = int(accessor.get("byteOffset", 0))
    view_offset = int(view.get("byteOffset", 0))
    stride = int(view.get("byteStride", np.dtype(dtype).itemsize * elem_count))
    item_bytes = np.dtype(dtype).itemsize * elem_count
    start = view_offset + accessor_offset
    if stride == item_bytes:
        arr = np.frombuffer(binary, dtype=dtype, count=count * elem_count, offset=start)
        return arr.reshape(count, elem_count).copy()
    arr = np.empty((count, elem_count), dtype=dtype)
    for row in range(count):
        begin = start + row * stride
        arr[row] = np.frombuffer(binary[begin:begin + item_bytes], dtype=dtype, count=elem_count)
    return arr


def quaternion_matrix_xyzw(value: Sequence[float]) -> np.ndarray:
    x, y, z, w = [float(item) for item in value]
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 0.0:
        raise ValueError("zero-length glTF node quaternion")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def node_local_matrix(node: dict) -> np.ndarray:
    if "matrix" in node:
        return np.asarray(node["matrix"], dtype=np.float32).reshape(4, 4, order="F")
    translation = np.asarray(node.get("translation", [0.0, 0.0, 0.0]), dtype=np.float32)
    rotation = quaternion_matrix_xyzw(node.get("rotation", [0.0, 0.0, 0.0, 1.0]))
    scale = np.asarray(node.get("scale", [1.0, 1.0, 1.0]), dtype=np.float32)
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = rotation @ np.diag(scale)
    matrix[:3, 3] = translation
    return matrix


def mesh_world_matrices(gltf: dict) -> Dict[int, List[np.ndarray]]:
    nodes = gltf.get("nodes", [])
    scenes = gltf.get("scenes", [])
    scene_index = int(gltf.get("scene", 0))
    root_nodes = scenes[scene_index]["nodes"]
    matrices: Dict[int, List[np.ndarray]] = {}

    def visit(node_index: int, parent: np.ndarray) -> None:
        node = nodes[int(node_index)]
        world = parent @ node_local_matrix(node)
        if "mesh" in node:
            mesh_index = int(node["mesh"])
            matrices.setdefault(mesh_index, []).append(world)
        for child in node.get("children", []):
            visit(int(child), world)

    for root in root_nodes:
        visit(int(root), np.eye(4, dtype=np.float32))
    return matrices


def transform_positions(positions: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    ones = np.ones((positions.shape[0], 1), dtype=np.float32)
    hom = np.concatenate([positions.astype(np.float32), ones], axis=1)
    glb_world = (hom @ matrix.T)[:, :3]
    return glb_world @ GLB_TO_FREEART3D_WORLD.T


def load_state_mesh(path: Path, device: torch.device) -> StateMesh:
    gltf, binary = read_glb_chunks(path)
    parts: List[MeshPart] = []
    matrices_by_mesh = mesh_world_matrices(gltf)
    for mesh_index, mesh in enumerate(gltf.get("meshes", [])):
        matrices = matrices_by_mesh.get(mesh_index, [])
        if not matrices:
            raise ValueError(f"mesh {mesh_index} has no scene node instance: {path}")
        for matrix in matrices:
            for primitive in mesh.get("primitives", []):
                attributes = primitive["attributes"]
                positions = accessor_array(gltf, binary, attributes["POSITION"]).astype(np.float32)
                positions = transform_positions(positions, matrix)
                uvs = accessor_array(gltf, binary, attributes["TEXCOORD_0"]).astype(np.float32)
                faces = accessor_array(gltf, binary, primitive["indices"]).astype(np.int64)
                if faces.shape[1] != 1:
                    raise ValueError(f"indices accessor must be scalar: {path}")
                faces = faces.reshape(-1, 3)
                parts.append(
                    MeshPart(
                        positions=torch.from_numpy(positions).to(device=device),
                        uvs=torch.from_numpy(uvs).to(device=device),
                        faces=torch.from_numpy(faces).to(device=device, dtype=torch.int32),
                        material_id=int(primitive["material"]),
                    )
                )
    if not parts:
        raise ValueError(f"no mesh primitives found: {path}")
    return StateMesh(path=path, parts=tuple(parts))


def extract_embedded_images(path: Path) -> List[Image.Image]:
    gltf, binary = read_glb_chunks(path)
    images: List[Image.Image] = []
    for image in gltf.get("images", []):
        view = gltf["bufferViews"][int(image["bufferView"])]
        start = int(view.get("byteOffset", 0))
        length = int(view["byteLength"])
        payload = binary[start:start + length]
        images.append(Image.open(BytesIO(payload)).convert("RGB"))
    if not images:
        raise ValueError(f"no embedded images found: {path}")
    return images


def image_to_texture_param(image: Image.Image, device: torch.device) -> torch.Tensor:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).to(device=device)
    tensor = tensor.clamp(1.0e-4, 1.0 - 1.0e-4)
    return torch.logit(tensor).detach().clone().requires_grad_(True)


def texture_from_param(param: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(param)


def texture_from_residual(
    initial: torch.Tensor,
    residual_param: torch.Tensor,
    max_delta: float,
) -> torch.Tensor:
    residual_low = torch.tanh(residual_param) * float(max_delta)
    residual_chw = residual_low.permute(2, 0, 1)[None, ...]
    residual = F.interpolate(
        residual_chw,
        size=(int(initial.shape[0]), int(initial.shape[1])),
        mode="bilinear",
        align_corners=False,
    )[0].permute(1, 2, 0)
    return (initial + residual).clamp(0.0, 1.0)


def parse_int_list(value: str) -> List[int]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    return [int(item) for item in items]


def load_wan_keyframes(wan_dir: Path, frame_indices: Sequence[int], device: torch.device) -> torch.Tensor:
    video_path = wan_dir / "wan_video_target_3FHW_uint8.pt"
    if not video_path.is_file():
        raise FileNotFoundError(f"Wan tensor not found: {video_path}")
    video = torch.load(video_path, map_location="cpu")
    if video.ndim != 4 or int(video.shape[0]) != 3:
        raise ValueError(f"expected Wan tensor [3,F,H,W], got {tuple(video.shape)}")
    frames = video[:, frame_indices].permute(1, 2, 3, 0).to(torch.float32) / 255.0
    return frames.to(device=device)


def load_original_render(path: Path, size_hw: Tuple[int, int], device: torch.device) -> torch.Tensor:
    image = Image.open(path).convert("RGB").resize((int(size_hw[1]), int(size_hw[0])), Image.BICUBIC)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).to(device=device)


def mesh_bounds(mesh: StateMesh) -> Tuple[torch.Tensor, torch.Tensor]:
    vertices = torch.cat([part.positions for part in mesh.parts], dim=0)
    return vertices.amin(dim=0), vertices.amax(dim=0)


def build_camera(reference_mesh: StateMesh, fov_degrees: float, device: torch.device) -> CameraSpec:
    min_corner, max_corner = mesh_bounds(reference_mesh)
    center = (min_corner + max_corner) * 0.5
    extent = max_corner - min_corner
    object_scale = torch.max(extent)
    camera_distance = 2.1 * object_scale
    azi = math.pi / 8.0
    height = math.sin(math.radians(45.0)) * camera_distance
    x_pos = math.sin(azi) * camera_distance
    y_pos = -math.cos(azi) * camera_distance
    source = torch.stack([center[0] + x_pos, center[1] + y_pos, center[2] + height])
    return CameraSpec(
        source=source.to(device=device),
        target=center.to(device=device),
        fov_degrees=float(fov_degrees),
        near=float(object_scale.detach().cpu()) * 0.02,
        far=float(object_scale.detach().cpu()) * 20.0,
    )


def vertices_to_clip(vertices: torch.Tensor, camera: CameraSpec, height: int, width: int) -> torch.Tensor:
    forward = F.normalize(camera.target - camera.source, dim=0)
    up_hint = torch.tensor([0.0, 0.0, 1.0], device=vertices.device)
    right = F.normalize(torch.cross(forward, up_hint, dim=0), dim=0)
    up = F.normalize(torch.cross(right, forward, dim=0), dim=0)
    rel = vertices - camera.source[None, :]
    x = rel @ right
    y = rel @ up
    z = (rel @ forward).clamp_min(camera.near)
    aspect = float(width) / float(height)
    tan_half = math.tan(math.radians(camera.fov_degrees) * 0.5)
    x_clip = x / (tan_half * aspect)
    y_clip = y / tan_half
    z_ndc = 2.0 * (z - camera.near) / (camera.far - camera.near) - 1.0
    z_clip = z_ndc * z
    return torch.stack([x_clip, y_clip, z_clip, z], dim=-1)


def merge_parts(parts: Sequence[MeshPart]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    positions: List[torch.Tensor] = []
    uvs: List[torch.Tensor] = []
    faces: List[torch.Tensor] = []
    face_materials: List[torch.Tensor] = []
    offset = 0
    for part in parts:
        positions.append(part.positions)
        uvs.append(part.uvs)
        faces.append(part.faces + int(offset))
        face_materials.append(
            torch.full((part.faces.shape[0],), int(part.material_id), device=part.faces.device, dtype=torch.long)
        )
        offset += int(part.positions.shape[0])
    return (
        torch.cat(positions, dim=0),
        torch.cat(uvs, dim=0),
        torch.cat(faces, dim=0),
        torch.cat(face_materials, dim=0),
    )


def render_state(
    glctx: dr.RasterizeCudaContext,
    mesh: StateMesh,
    textures: Sequence[torch.Tensor],
    camera: CameraSpec,
    size_hw: Tuple[int, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    height, width = int(size_hw[0]), int(size_hw[1])
    positions, uvs, faces, face_materials = merge_parts(mesh.parts)
    clip = vertices_to_clip(positions, camera, height, width)
    rast, _ = dr.rasterize(glctx, clip[None, ...], faces, resolution=(height, width))
    uv_attr, _ = dr.interpolate(uvs[None, ...], rast, faces)
    uv_attr = uv_attr.contiguous()
    samples = []
    for tex in textures:
        samples.append(dr.texture(tex[None, ...].contiguous(), uv_attr, filter_mode="linear")[0])
    stacked = torch.stack(samples, dim=0)
    prim = rast[0, :, :, 3].to(torch.long) - 1
    valid = prim >= 0
    clamped = prim.clamp_min(0)
    mat = face_materials[clamped].clamp(0, len(textures) - 1)
    color = stacked.permute(1, 2, 0, 3)[torch.arange(height, device=prim.device)[:, None], torch.arange(width, device=prim.device)[None, :], mat]
    color = torch.where(valid[..., None], color, torch.zeros_like(color))
    return torch.flip(color, dims=[0]), torch.flip(valid.to(torch.float32), dims=[0])


def total_variation(texture: torch.Tensor) -> torch.Tensor:
    dx = (texture[:, 1:, :] - texture[:, :-1, :]).abs().mean()
    dy = (texture[1:, :, :] - texture[:-1, :, :]).abs().mean()
    return dx + dy


def psnr_from_mse(mse: float) -> float:
    return -10.0 * math.log10(max(float(mse), 1.0e-10))


def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum().clamp_min(1.0) * 3.0
    return ((pred - target).abs() * mask[..., None]).sum() / denom


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum().clamp_min(1.0) * 3.0
    return (((pred - target) ** 2) * mask[..., None]).sum() / denom


def save_png(path: Path, image: torch.Tensor) -> None:
    arr = (image.detach().cpu().clamp(0, 1).numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(arr).save(path)


def make_compare_strip(
    path: Path,
    labels: Sequence[str],
    rows: Sequence[Sequence[Path]],
    cell_size: Tuple[int, int] = (256, 256),
) -> None:
    width, height = int(cell_size[0]), int(cell_size[1])
    label_h = 28
    canvas = Image.new("RGB", (width * len(labels), (height + label_h) * len(rows)), "white")
    draw = ImageDraw.Draw(canvas)
    for row_idx, row in enumerate(rows):
        y0 = row_idx * (height + label_h)
        for col_idx, image_path in enumerate(row):
            x0 = col_idx * width
            draw.text((x0 + 6, y0 + 6), labels[col_idx], fill=(0, 0, 0))
            image = Image.open(image_path).convert("RGB").resize((width, height), Image.BICUBIC)
            canvas.paste(image, (x0, y0 + label_h))
    canvas.save(path)


def tensor_to_png_bytes(texture: torch.Tensor) -> bytes:
    arr = (texture.detach().cpu().clamp(0, 1).numpy() * 255.0).round().astype(np.uint8)
    image = Image.fromarray(arr)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def replace_glb_images(src_path: Path, dst_path: Path, image_payloads: Sequence[bytes]) -> None:
    gltf = GLTF2().load_binary(str(src_path))
    binary = bytearray(gltf.binary_blob())
    if len(gltf.images) != len(image_payloads):
        raise ValueError(
            f"image count mismatch for {src_path}: glb has {len(gltf.images)}, "
            f"payloads={len(image_payloads)}"
        )
    for idx, payload in enumerate(image_payloads):
        while len(binary) % 4 != 0:
            binary.append(0)
        offset = len(binary)
        binary.extend(payload)
        view = BufferView(buffer=0, byteOffset=offset, byteLength=len(payload))
        gltf.bufferViews.append(view)
        gltf.images[idx].bufferView = len(gltf.bufferViews) - 1
        gltf.images[idx].mimeType = "image/png"
        gltf.images[idx].uri = None
    while len(binary) % 4 != 0:
        binary.append(0)
    gltf.buffers[0].byteLength = len(binary)
    gltf.set_binary_blob(bytes(binary))
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    gltf.save_binary(str(dst_path))


def copy_metadata(origin_sds: Path, out_sds: Path) -> None:
    out_sds.mkdir(parents=True, exist_ok=True)
    for name in ("joint_info.json", "config.yaml", "output_check_list.json"):
        src = origin_sds / name
        if src.is_file():
            shutil.copy2(src, out_sds / name)


def write_optimized_glbs(origin_sds: Path, out_sds: Path, textures: Sequence[torch.Tensor]) -> None:
    payloads = [tensor_to_png_bytes(tex) for tex in textures]
    copy_metadata(origin_sds, out_sds)
    part_in = origin_sds / "part_meshes"
    part_out = out_sds / "part_meshes"
    replace_glb_images(part_in / "fixed.glb", part_out / "fixed.glb", [payloads[0]])
    replace_glb_images(part_in / "articulated_00.glb", part_out / "articulated_00.glb", [payloads[1]])
    states_out = out_sds / "states"
    states_out.mkdir(parents=True, exist_ok=True)
    for state_path in sorted((origin_sds / "states").glob("qpos_*.glb")):
        replace_glb_images(state_path, states_out / state_path.name, payloads)


def write_metrics(path: Path, metrics: Dict[str, object]) -> None:
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    torch.manual_seed(int(args.seed))
    device = torch.device(args.device)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for nvdiffrast texture optimization.")

    origin_dir = Path(args.origin_dir).resolve()
    origin_sds = origin_dir / "sds_output"
    wan_dir = Path(args.wan_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_sds = out_dir / "sds_output"
    viz_dir = out_dir / "viz"
    render_dir = out_dir / "renders"
    atlas_dir = out_dir / "atlas"
    for directory in (viz_dir, render_dir, atlas_dir):
        directory.mkdir(parents=True, exist_ok=True)

    states = parse_int_list(args.states)
    frame_indices = parse_int_list(args.frame_indices)
    if len(states) != len(frame_indices):
        raise ValueError("--states and --frame_indices must have the same length")

    state_meshes = [
        load_state_mesh(origin_sds / "states" / f"qpos_{state:02d}.glb", device=device)
        for state in states
    ]
    reference_mesh = load_state_mesh(origin_sds / "states" / f"qpos_{states[-1]:02d}.glb", device=device)
    camera = build_camera(reference_mesh, float(args.fov), device)
    targets = load_wan_keyframes(wan_dir, frame_indices, device)
    size_hw = (int(targets.shape[1]), int(targets.shape[2]))

    initial_images = extract_embedded_images(origin_sds / "states" / f"qpos_{states[-1]:02d}.glb")
    if len(initial_images) != 2:
        raise ValueError("expected exactly two embedded textures in the state GLB")
    initial_params = [image_to_texture_param(image, device) for image in initial_images]
    initial_textures = [texture_from_param(param).detach().clone() for param in initial_params]
    residual_size = int(args.residual_size)
    if residual_size <= 0:
        raise ValueError("--residual_size must be positive")
    params = [
        torch.zeros(
            (residual_size, residual_size, 3),
            device=device,
            dtype=initial_textures[0].dtype,
            requires_grad=True,
        )
        for _ in initial_textures
    ]
    optimizer = torch.optim.Adam(params, lr=float(args.lr))
    glctx = dr.RasterizeCudaContext(device=device)

    log_path = out_dir / "losses.jsonl"
    with log_path.open("w", encoding="utf-8") as log_f:
        for iteration in range(int(args.iters) + 1):
            textures = [
                texture_from_residual(init, param, float(args.max_delta))
                for init, param in zip(initial_textures, params)
            ]
            losses = []
            for mesh, target in zip(state_meshes, targets):
                pred, alpha = render_state(glctx, mesh, textures, camera, size_hw)
                losses.append(masked_l1(pred, target, alpha))
            target_loss = torch.stack(losses).mean()
            tex_reg = torch.stack(
                [(tex - init).abs().mean() for tex, init in zip(textures, initial_textures)]
            ).mean()
            tv_loss = torch.stack(
                [total_variation(tex - init) for tex, init in zip(textures, initial_textures)]
            ).mean()
            loss = target_loss + float(args.lambda_tex) * tex_reg + float(args.lambda_tv) * tv_loss
            if iteration < int(args.iters):
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            if iteration % int(args.log_interval) == 0 or iteration == int(args.iters):
                record = {
                    "iter": int(iteration),
                    "loss": float(loss.detach().cpu()),
                    "target_l1": float(target_loss.detach().cpu()),
                    "tex_reg": float(tex_reg.detach().cpu()),
                    "tv": float(tv_loss.detach().cpu()),
                }
                log_f.write(json.dumps(record) + "\n")
                log_f.flush()
                print(record, flush=True)

    textures = [
        texture_from_residual(init, param, float(args.max_delta)).detach()
        for init, param in zip(initial_textures, params)
    ]
    for idx, texture in enumerate(textures):
        save_png(atlas_dir / f"optimized_texture_{idx}.png", texture)
        save_png(atlas_dir / f"initial_texture_{idx}.png", initial_textures[idx])

    metrics: Dict[str, object] = {"states": states, "frame_indices": frame_indices, "per_state": []}
    compare_rows: List[List[Path]] = []
    baseline_losses = []
    optimized_losses = []
    baseline_mses = []
    optimized_mses = []
    for row_idx, (state, frame_idx, mesh, target) in enumerate(zip(states, frame_indices, state_meshes, targets)):
        baseline, alpha = render_state(glctx, mesh, initial_textures, camera, size_hw)
        optimized, _ = render_state(glctx, mesh, textures, camera, size_hw)
        base_l1 = masked_l1(baseline, target, alpha)
        opt_l1 = masked_l1(optimized, target, alpha)
        base_mse = masked_mse(baseline, target, alpha)
        opt_mse = masked_mse(optimized, target, alpha)
        baseline_losses.append(float(base_l1.cpu()))
        optimized_losses.append(float(opt_l1.cpu()))
        baseline_mses.append(float(base_mse.cpu()))
        optimized_mses.append(float(opt_mse.cpu()))
        target_path = render_dir / f"state_{state:02d}_wan_frame_{frame_idx:02d}.png"
        base_path = render_dir / f"state_{state:02d}_baseline_nvdiff.png"
        opt_path = render_dir / f"state_{state:02d}_optimized_nvdiff.png"
        freeart_path = origin_dir / "renderings" / f"rendering_joint_00_state_{state:02d}.png"
        save_png(target_path, target)
        save_png(base_path, baseline)
        save_png(opt_path, optimized)
        compare_rows.append([freeart_path, target_path, base_path, opt_path])
        metrics["per_state"].append(
            {
                "state": int(state),
                "wan_frame": int(frame_idx),
                "baseline_l1": float(base_l1.cpu()),
                "optimized_l1": float(opt_l1.cpu()),
                "baseline_psnr": psnr_from_mse(float(base_mse.cpu())),
                "optimized_psnr": psnr_from_mse(float(opt_mse.cpu())),
            }
        )

    metrics["mean_baseline_l1"] = float(np.mean(baseline_losses))
    metrics["mean_optimized_l1"] = float(np.mean(optimized_losses))
    metrics["mean_l1_delta"] = float(np.mean(baseline_losses) - np.mean(optimized_losses))
    metrics["mean_baseline_psnr"] = psnr_from_mse(float(np.mean(baseline_mses)))
    metrics["mean_optimized_psnr"] = psnr_from_mse(float(np.mean(optimized_mses)))
    write_metrics(out_dir / "metrics.json", metrics)

    make_compare_strip(
        viz_dir / "compare_keyframes.png",
        ["FreeArt3D", "Wan InP target", "Initial atlas render", "Optimized atlas render"],
        compare_rows,
    )
    write_optimized_glbs(origin_sds, out_sds, textures)
    print(f"Wrote optimized texture stage to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
