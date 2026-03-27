#!/usr/bin/env python3
"""Download a few small openly licensed / Wikimedia URLs for local smoke tests."""

from __future__ import annotations

import urllib.request
from pathlib import Path

# Direct Wikimedia Commons file URLs (PD or appropriate license for testing)
SAMPLES = [
    (
        "apple.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/1/15/Red_Apple.jpg/320px-Red_Apple.jpg",
    ),
    (
        "bottle.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Plastic_bottle_3.jpg/320px-Plastic_bottle_3.jpg",
    ),
    (
        "banana.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8a/Banana-Single.jpg/320px-Banana-Single.jpg",
    ),
]


def main():
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, url in SAMPLES:
        dest = out_dir / name
        if dest.is_file():
            print("skip exists:", dest)
            continue
        print("fetch", url)
        urllib.request.urlretrieve(url, dest)
        print("wrote", dest)

    print("Done. Try: python main.py", out_dir / "bottle.jpg")


if __name__ == "__main__":
    main()
