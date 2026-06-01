import json
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path.cwd()
CASES = ["30857", "7128", "7201", "7201_new"]
KEYS = [
    "stage_b/O_stack.npy",
    "stage_b/O_base_canonical.npy",
    "stage_b/O_move_per_state.npy",
    "stage_b/P_base_canonical.npy",
    "stage_b/P_move_evidence_per_state.npy",
    "stage_b/viz/bmcsa/M_motion_corridor_64.npy",
]

for case in CASES:
    case_dir = ROOT / "outputs" / case
    if not case_dir.exists():
        continue
    print(f"== {case} ==")

    for rel in ["stage_a/wan_video_target.mp4", "stage_a/wan_video_target_3FHW_uint8.pt"]:
        path = case_dir / rel
        if path.exists():
            print(f"exists {rel} size={path.stat().st_size}")

    for img_rel in ["stage_b/input_keyframes.png", "stage_b/inputs/00_seg.png", "stage_b/inputs/05_seg.png"]:
        path = case_dir / img_rel
        if path.exists():
            with Image.open(path) as image:
                print(f"image {img_rel} size={image.size} mode={image.mode}")

    for rel in KEYS:
        path = case_dir / rel
        if not path.exists():
            continue
        arr = np.load(path)
        finite = arr[np.isfinite(arr)]
        if finite.size:
            stats = f"min={finite.min():.4f} max={finite.max():.4f} mean={finite.mean():.4f}"
        else:
            stats = "no_finite_values"
        print(f"array {rel} shape={arr.shape} dtype={arr.dtype} {stats}")

    stage_c = case_dir / "stage_c" / "stage_c_joint_init.json"
    if stage_c.exists():
        data = json.loads(stage_c.read_text(encoding="utf-8"))
        summary = {
            "joint_type": data.get("joint_type"),
            "confidence": data.get("confidence"),
            "axis_fit_source": data.get("diagnostics", {}).get("axis_fit_source"),
            "type_str": data.get("diagnostics", {}).get("type_str"),
            "anchors_count": data.get("anchors_count"),
        }
        print(f"stage_c {json.dumps(summary, sort_keys=True)}")
