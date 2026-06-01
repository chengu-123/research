"""Camera sanity check for the post-FreeArt3D texture stage."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import nvdiffrast.torch as dr
import numpy as np
import torch
from PIL import Image, ImageDraw

from wan_texture_optimize import (
    build_camera,
    extract_embedded_images,
    image_to_texture_param,
    load_state_mesh,
    render_state,
    save_png,
    texture_from_param,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render fixed FreeArt3D states with the texture-stage camera."
    )
    parser.add_argument("--origin_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--states", default="0,1,2,3,4,5")
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--fov", type=float, default=45.0)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def parse_int_list(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def foreground_bbox(image: Image.Image) -> Tuple[int, int, int, int]:
    rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    mask = rgba[:, :, 3] > 16
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return (0, 0, 0, 0)
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def draw_bbox(image: Image.Image, bbox: Tuple[int, int, int, int], color: Tuple[int, int, int]) -> Image.Image:
    out = image.convert("RGB")
    draw = ImageDraw.Draw(out)
    draw.rectangle(bbox, outline=color, width=3)
    return out


def make_strip(path: Path, rows: List[Tuple[str, Path, Path]], cell: Tuple[int, int]) -> None:
    width, height = cell
    label_h = 34
    canvas = Image.new("RGB", (2 * width, len(rows) * (height + label_h)), "white")
    draw = ImageDraw.Draw(canvas)
    for row_idx, (label, original_path, rendered_path) in enumerate(rows):
        y0 = row_idx * (height + label_h)
        draw.text((6, y0 + 5), f"{label} FreeArt3D render", fill=(0, 0, 0))
        draw.text((width + 6, y0 + 5), f"{label} texture-stage render", fill=(0, 0, 0))
        original = Image.open(original_path).convert("RGB").resize(cell, Image.BICUBIC)
        rendered = Image.open(rendered_path).convert("RGB").resize(cell, Image.BICUBIC)
        canvas.paste(original, (0, y0 + label_h))
        canvas.paste(rendered, (width, y0 + label_h))
    canvas.save(path)


def save_mask(path: Path, mask: np.ndarray) -> None:
    arr = (mask.astype(np.uint8) * 255)
    Image.fromarray(arr, mode="L").save(path)


def make_mask_strip(path: Path, rows: List[Tuple[str, Path, Path]], cell: Tuple[int, int]) -> None:
    width, height = cell
    label_h = 34
    canvas = Image.new("RGB", (2 * width, len(rows) * (height + label_h)), "white")
    draw = ImageDraw.Draw(canvas)
    for row_idx, (label, original_path, rendered_path) in enumerate(rows):
        y0 = row_idx * (height + label_h)
        draw.text((6, y0 + 5), f"{label} FreeArt3D alpha", fill=(0, 0, 0))
        draw.text((width + 6, y0 + 5), f"{label} texture-stage alpha", fill=(0, 0, 0))
        original = Image.open(original_path).convert("L").resize(cell, Image.NEAREST).convert("RGB")
        rendered = Image.open(rendered_path).convert("L").resize(cell, Image.NEAREST).convert("RGB")
        canvas.paste(original, (0, y0 + label_h))
        canvas.paste(rendered, (width, y0 + label_h))
    canvas.save(path)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for nvdiffrast camera checking.")

    origin_dir = Path(args.origin_dir).resolve()
    origin_sds = origin_dir / "sds_output"
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    states = parse_int_list(args.states)
    size_hw = (int(args.height), int(args.width))

    reference_mesh = load_state_mesh(origin_sds / "states" / f"qpos_{states[-1]:02d}.glb", device=device)
    camera = build_camera(reference_mesh, float(args.fov), device)
    initial_images = extract_embedded_images(origin_sds / "states" / f"qpos_{states[-1]:02d}.glb")
    params = [image_to_texture_param(image, device) for image in initial_images]
    textures = [texture_from_param(param).detach() for param in params]
    glctx = dr.RasterizeCudaContext(device=device)

    rows: List[Tuple[str, Path, Path]] = []
    mask_rows: List[Tuple[str, Path, Path]] = []
    records = []
    for state in states:
        mesh = load_state_mesh(origin_sds / "states" / f"qpos_{state:02d}.glb", device=device)
        rendered, alpha = render_state(glctx, mesh, textures, camera, size_hw)
        rendered_path = out_dir / f"nvdiff_state_{state:02d}.png"
        rendered_mask_path = out_dir / f"nvdiff_mask_state_{state:02d}.png"
        save_png(rendered_path, rendered)
        rendered_mask = alpha.detach().cpu().numpy() > 0.5
        save_mask(rendered_mask_path, rendered_mask)
        original_path = origin_dir / "renderings" / f"rendering_joint_00_state_{state:02d}.png"
        original_mask_path = out_dir / f"freeart_alpha_state_{state:02d}.png"
        original_alpha = np.asarray(Image.open(original_path).convert("RGBA"))[:, :, 3] > 16
        save_mask(original_mask_path, original_alpha)
        rows.append((f"state {state:02d}", original_path, rendered_path))
        mask_rows.append((f"state {state:02d}", original_mask_path, rendered_mask_path))

        original = Image.open(original_path)
        original_bbox = foreground_bbox(original)
        ys, xs = np.nonzero(rendered_mask)
        if len(xs) == 0:
            rendered_bbox = (0, 0, 0, 0)
        else:
            rendered_bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        records.append(
            {
                "state": int(state),
                "original_bbox": list(original_bbox),
                "rendered_bbox": list(rendered_bbox),
                "original_size": [int(original.width), int(original.height)],
                "rendered_size": [int(args.width), int(args.height)],
            }
        )

    make_strip(out_dir / "camera_compare_strip.png", rows, cell=(320, 320))
    make_mask_strip(out_dir / "camera_mask_strip.png", mask_rows, cell=(320, 320))
    (out_dir / "camera_metrics.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Wrote camera check to {out_dir}")


if __name__ == "__main__":
    main()
