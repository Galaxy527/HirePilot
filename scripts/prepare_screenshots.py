"""Crop browser chrome only for README screenshots (no blur)."""
from __future__ import annotations

from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "pic"
OUT = ROOT / "docs" / "screenshots"
OUT.mkdir(parents=True, exist_ok=True)

# All captures under pic/ → docs/screenshots/
SELECT = {
    "01-login.jpg": "{5BD4BC64-14CD-4b57-B6D2-3538B7706BDA}.png",
    "02-candidate-upload.jpg": "{E8C137F7-38DE-4d03-8467-84D8D062B33B}.png",
    "03-resume-score.jpg": "{B16D33F6-E435-408c-94BF-F24F410DBCC3}.png",
    "04-mock-interview.jpg": "{B3E595E9-A904-4279-99F3-F01D23B8D656}.png",
    "05-hr-dashboard.jpg": "{8D2B4D15-63DE-43ac-A9A2-FCFD69B10671}.png",
    "06-hr-jobs.jpg": "{48AB694B-88D9-4ef8-B9ED-60056170344B}.png",
    "07-hr-agent-chat.jpg": "{577CA50E-0EC4-4fbc-94BD-340E65689C26}.png",
    "08-hr-resume-detail.jpg": "{631B99C2-6945-45c8-8F6C-81D1B91BA3CB}.png",
    "09-ops-rag-eval.jpg": "{674E172A-7220-44e2-9A42-5397A96AC5AA}.png",
    "10-ops-logs.jpg": "{222F22A0-BFD9-4854-B337-F3861FFDE0A4}.png",
    "11-notifications.jpg": "{678D62EE-5200-46e7-AE9C-336E7BB4D6DE}.png",
}

TOP_CROP = 128
MAX_W = 1600


def process(im: Image.Image, kind: str) -> Image.Image:
    w, h = im.size
    im = im.crop((0, TOP_CROP, w, h)).convert("RGB")
    w, h = im.size

    # Notifications page is mostly empty below the cards — tighten for README.
    if kind == "11-notifications.jpg":
        im = im.crop((0, 0, w, min(h, 720)))
        w, h = im.size

    if w > MAX_W:
        nh = int(h * MAX_W / w)
        im = im.resize((MAX_W, nh), Image.Resampling.LANCZOS)
    return im


def main() -> None:
    for junk in OUT.glob("id_*.jpg"):
        junk.unlink()
    for junk in OUT.glob("_id_*"):
        junk.unlink()
    for junk in OUT.glob("_preview_*"):
        junk.unlink()
    for junk in OUT.glob("*.png"):
        junk.unlink()

    for out_name, src_name in SELECT.items():
        src = SRC / src_name
        if not src.exists():
            raise SystemExit(f"missing source: {src}")
        out = process(Image.open(src), out_name)
        dest = OUT / out_name
        out.save(dest, "JPEG", quality=88, optimize=True)
        print(f"wrote {dest.name} {out.size} {dest.stat().st_size}")
    print(f"done {len(SELECT)}")


if __name__ == "__main__":
    main()
