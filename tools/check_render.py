from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageChops


def analyze_page(path: Path, margin_threshold: int = 12) -> dict:
    image = Image.open(path).convert("L")
    width, height = image.size
    background = Image.new("L", image.size, 255)
    diff = ImageChops.difference(image, background)
    bbox = diff.getbbox()
    if bbox is None:
        return {
            "page": path.name,
            "width": width,
            "height": height,
            "blank": True,
            "margins": None,
            "flags": ["blank"],
        }
    left, top, right, bottom = bbox
    margins = {
        "left": left,
        "top": top,
        "right": width - right,
        "bottom": height - bottom,
    }
    flags = []
    if left < margin_threshold or top < margin_threshold or width - right < margin_threshold or height - bottom < margin_threshold:
        flags.append("near_edge")
    if (bottom - top) < height * 0.08:
        flags.append("short_content")
    return {
        "page": path.name,
        "width": width,
        "height": height,
        "blank": False,
        "margins": margins,
        "flags": flags,
    }


def main() -> None:
    render_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("docs/render_manual")
    pages = sorted(render_dir.glob("page-*.png"))
    if not pages:
        raise FileNotFoundError(f"No page-*.png found in {render_dir}")
    flagged = []
    for page in pages:
        result = analyze_page(page)
        if result["flags"]:
            flagged.append(result)
    print(f"pages={len(pages)} flagged={len(flagged)}")
    for result in flagged:
        print(result)
    if flagged:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
