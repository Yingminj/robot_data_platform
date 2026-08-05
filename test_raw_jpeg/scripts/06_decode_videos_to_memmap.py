#!/usr/bin/env python3
"""Stage 6: decode the six datasets' videos into uint8 memmaps.

The feature evaluation runs in the ``dino`` environment, which has torch but no
PyAV.  Decoding once here rather than inside that script also avoids decoding
the same eighteen videos three times over (once per model).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import av
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import CAMERAS, CONFIG, ROOT, open_memmap  # noqa: E402

CRF_LEVELS: list[int] = CONFIG["video"]["crf_levels"]
SOURCES = ["raw", "jpeg100", "jpeg80"]


def decode_frames(path: Path):
    with av.open(str(path), "r") as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            yield frame.to_ndarray(format="rgb24")


def main() -> None:
    reference = open_memmap("raw", CAMERAS[0], "r")
    shape = reference.shape
    frames = shape[0]
    print(f"target shape {shape}")

    for source in SOURCES:
        for crf in CRF_LEVELS:
            dataset = ROOT / "lerobot" / f"{source}_crf{crf}"
            kind = f"vid_{source}_crf{crf}"
            for camera in CAMERAS:
                target = ROOT / "intermediate" / f"{kind}__{camera}.npy"
                if target.exists():
                    print(f"skip {target.name}")
                    continue
                matches = sorted((dataset / "videos" / f"observation.images.{camera}").rglob("*.mp4"))
                if len(matches) != 1:
                    raise RuntimeError(f"expected one mp4 for {camera} in {dataset}: {matches}")
                started = time.perf_counter()
                out = open_memmap(kind, camera, "w+", shape=shape)
                count = 0
                for index, frame in enumerate(decode_frames(matches[0])):
                    if index >= frames:
                        raise RuntimeError(f"{matches[0]} has more than {frames} frames")
                    out[index] = frame
                    count += 1
                if count != frames:
                    raise RuntimeError(f"{matches[0]}: decoded {count}, expected {frames}")
                out.flush()
                del out
                print(f"wrote {target.name} in {time.perf_counter() - started:.1f}s")
    print("stage 6 complete")


if __name__ == "__main__":
    main()
