"""Capture five native Stray views; uses an existing forecast and local service."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent
VIEWS = [
    {"file": "01-stray-earth-cover.png", "day": 180, "location": "24,-24", "zoom": 1.06},
    {"file": "02-stray-europe-africa.png", "day": 180, "location": "29,12", "zoom": 2.0},
    {"file": "03-stray-utah-cohorts.png", "day": 7, "location": "40.25,-111.5", "zoom": 32,
     "stream": True},
    {"file": "04-stray-north-america.png", "day": 90, "location": "40,-101", "zoom": 2.8},
    {"file": "05-stray-bird-migration.png", "day": 180, "location": "43,10", "zoom": 6,
     "stream": True, "species": "Ciconia ciconia"},
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-url", default="http://127.0.0.1:8766")
    parser.add_argument("--forecast", type=Path,
                        default=ROOT / "htb-gui/runs/server-50y-20261108/forecast.json")
    args = parser.parse_args()
    gui = ROOT / "htb-gui"
    records = []
    with tempfile.TemporaryDirectory(prefix="stray-captures-") as temp:
        for view in VIEWS:
            frame = Path(temp) / "frame.ppm"
            env = {k: v for k, v in os.environ.items()
                   if not k.startswith(("ATLAS_TEST_", "ATLAS_SCREENSHOT", "ATLAS_FOCUS_", "ATLAS_STREAM_"))}
            env.update(__NV_PRIME_RENDER_OFFLOAD="1", __GLX_VENDOR_LIBRARY_NAME="nvidia",
                       ATLAS_INFERENCE_URL=args.inference_url,
                       ATLAS_FORECAST=str(args.forecast.resolve()),
                       ATLAS_TEST_FORECAST_DAY=str(view["day"]), ATLAS_TEST_TRAIL_HOURS="168",
                       ATLAS_TEST_LOCATION=view["location"], ATLAS_TEST_ZOOM=str(view["zoom"]),
                       ATLAS_SCREENSHOT_SIZE="1920x1080", ATLAS_SCREENSHOT=str(frame))
            if view.get("stream"):
                env.update(ATLAS_STREAM_TILES="1", ATLAS_STREAM_WARMUP="8")
            if view.get("species"):
                env["ATLAS_FOCUS_SPECIES"] = view["species"]
            result = subprocess.run([str(gui / "build/wildlife_atlas"), "--smoke-test", "forecast"],
                                    cwd=gui, env=env, capture_output=True, text=True, check=True)
            output = OUTPUT / view["file"]
            with Image.open(frame) as image:
                assert image.size == (1920, 1080)
                image.save(output, optimize=True)
            assert output.stat().st_size < 5_000_000, output
            record = {**view, "trail_hours": 168, "dimensions": [1920, 1080],
                      "bytes": output.stat().st_size,
                      "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                      "render_log": result.stdout.strip()}
            records.append(record)
            print(view["file"], output.stat().st_size, "bytes", flush=True)
    (OUTPUT / "captures.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
