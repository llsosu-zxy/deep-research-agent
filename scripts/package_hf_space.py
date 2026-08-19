from __future__ import annotations

import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = [
    "agents",
    "app",
    "core",
    "data/corpus",
    "docs",
    "eval",
    "scripts",
    "ui",
    "hf_space_app.py",
    "requirements.txt",
    "README.md",
    ".env.example",
]


def main() -> None:
    dist = ROOT / "dist"
    dist.mkdir(exist_ok=True)
    output = dist / "hf_space.zip"
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for include in INCLUDE:
            path = ROOT / include
            if path.is_file():
                archive.write(path, path.relative_to(ROOT))
                continue
            if not path.is_dir():
                print(f"skip missing: {include}")
                continue
            for file in path.rglob("*"):
                if "__pycache__" in file.parts or file.suffix in {".pyc", ".pyo"}:
                    continue
                if file.is_file():
                    archive.write(file, file.relative_to(ROOT))
    print(f"Wrote {output} ({output.stat().st_size / 1024:.1f} KiB)")


if __name__ == "__main__":
    sys.exit(main())
