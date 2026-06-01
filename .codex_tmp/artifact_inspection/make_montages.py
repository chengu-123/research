from pathlib import Path

from PIL import Image, ImageDraw
import cv2


ROOT = Path.cwd()
OUT = ROOT / ".codex_tmp" / "artifact_inspection"
OUT.mkdir(parents=True, exist_ok=True)

CASES = ["30857", "7128", "7201"]
RELS = [
    "stage_a/input_s_0_with_carpet.png",
    "stage_a/keyframes_6.png",
    "stage_a/wan_video_grid.png",
    "stage_b/input_keyframes.png",
    "stage_b/inputs/00_seg.png",
    "stage_b/inputs/01_seg.png",
    "stage_b/inputs/02_seg.png",
    "stage_b/inputs/03_seg.png",
    "stage_b/inputs/04_seg.png",
    "stage_b/inputs/05_seg.png",
]

for case in CASES:
    paths = []
    for rel in RELS:
        path = ROOT / "outputs" / case / rel
        if path.exists():
            paths.append((rel, path))

    thumbs = []
    for label, path in paths:
        image = Image.open(path).convert("RGB")
        image.thumbnail((360, 220), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (380, 260), "white")
        tile.paste(image, ((380 - image.width) // 2, 28))
        ImageDraw.Draw(tile).text((8, 8), f"{case}/{label}", fill=(0, 0, 0))
        thumbs.append(tile)

    if not thumbs:
        continue

    cols = 2
    rows = (len(thumbs) + cols - 1) // cols
    montage = Image.new("RGB", (cols * 380, rows * 260), (245, 245, 245))
    for idx, tile in enumerate(thumbs):
        montage.paste(tile, ((idx % cols) * 380, (idx // cols) * 260))

    montage.save(OUT / f"{case}_png_montage.png")

for case in ["30857", "7201"]:
    video_path = ROOT / "outputs" / case / "stage_a" / "wan_video_target.mp4"
    if not video_path.exists():
        continue

    cap = cv2.VideoCapture(str(video_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = sorted(set(int(i * max(frame_count - 1, 0) / 5) for i in range(6)))
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        image.thumbnail((360, 220), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (380, 260), "white")
        tile.paste(image, ((380 - image.width) // 2, 28))
        ImageDraw.Draw(tile).text((8, 8), f"{case}/frame_{idx}", fill=(0, 0, 0))
        frames.append(tile)
    cap.release()

    if not frames:
        continue

    cols = 2
    rows = (len(frames) + cols - 1) // cols
    montage = Image.new("RGB", (cols * 380, rows * 260), (245, 245, 245))
    for idx, tile in enumerate(frames):
        montage.paste(tile, ((idx % cols) * 380, (idx // cols) * 260))
    montage.save(OUT / f"{case}_video_frame_montage.png")

print(OUT)
