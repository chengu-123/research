import blenderproc as bproc

"""Print Blender-imported world-space bounds for GLB files."""

import argparse
import json
from pathlib import Path

import bpy
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect GLB bounds after Blender import.")
    parser.add_argument("paths", nargs="+")
    return parser.parse_args()


def imported_bounds(path: Path) -> dict:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(path))
    vertices = []
    for obj in bpy.context.view_layer.objects:
        if obj.type != "MESH":
            continue
        matrix_world = obj.matrix_world.copy()
        for vertex in obj.data.vertices:
            co = matrix_world @ vertex.co
            vertices.append([float(co.x), float(co.y), float(co.z)])
    if not vertices:
        raise ValueError(f"no mesh vertices imported from {path}")
    arr = np.asarray(vertices, dtype=np.float32)
    return {
        "path": str(path),
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
        "extent": (arr.max(axis=0) - arr.min(axis=0)).tolist(),
        "center": ((arr.max(axis=0) + arr.min(axis=0)) * 0.5).tolist(),
    }


def main() -> None:
    args = parse_args()
    records = [imported_bounds(Path(path).resolve()) for path in args.paths]
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
