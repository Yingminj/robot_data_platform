#!/usr/bin/env python3
"""Stage 1b: real on-the-wire JPEG sizes from the production recordings.

``express_mcap`` is a different session from ``express_raw``, so its frames
cannot be compared pixel-for-pixel against the raw ground truth.  What it can
do is confirm that the JPEG we synthesise at quality 80 in stage 1 lands in the
same size range as what the RealSense node actually wrote, i.e. that the
simulation is not accidentally compressing a different picture.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/kewei/YING/robot_data_platform/tool")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from robot_data.align.bag_io import open_bag_reader  # noqa: E402
from robot_data.profiles import load_profile  # noqa: E402

from common import CONFIG, ROOT, jpeg_chroma_subsampling, write_json  # noqa: E402

TOPIC = "/quad_tile/compressed"
MAX_BAGS = 12


def main() -> None:
    root = Path(CONFIG["reference_jpeg80_bags"])
    profile = load_profile(CONFIG["profile"])
    bags = sorted(p for p in root.glob("*/data") if p.is_dir())
    if not bags:
        raise SystemExit(f"no bags under {root}")
    selected = bags[:MAX_BAGS]
    print(f"{len(bags)} bags available, sampling {len(selected)}")

    per_bag = []
    all_sizes: list[int] = []
    subsampling: set[str] = set()
    dimensions: set[str] = set()

    for bag in selected:
        reader = open_bag_reader(bag, profile)
        reader.open()
        try:
            connections = [c for c in reader.connections if c.topic == TOPIC]
            if not connections:
                print(f"  {bag.parent.name}: no {TOPIC}")
                continue
            sizes: list[int] = []
            stamps: list[int] = []
            for connection, timestamp, rawdata in reader.messages(connections=connections):
                message = reader.deserialize(rawdata, connection.msgtype)
                buffer = bytes(message.data)
                sizes.append(len(buffer))
                stamps.append(int(timestamp))
                if len(sizes) == 1:
                    subsampling.add(jpeg_chroma_subsampling(buffer))
                    import cv2

                    decoded = cv2.imdecode(np.frombuffer(buffer, np.uint8), cv2.IMREAD_COLOR)
                    dimensions.add(f"{decoded.shape[1]}x{decoded.shape[0]}")
        finally:
            reader.close()
        array = np.asarray(sizes, dtype=np.float64)
        span_s = (stamps[-1] - stamps[0]) / 1e9 if len(stamps) > 1 else 0.0
        per_bag.append(
            {
                "bag": bag.parent.name,
                "frames": int(array.size),
                "span_s": span_s,
                "measured_fps": float(array.size / span_s) if span_s else None,
                "bytes_mean": float(array.mean()),
                "bytes_median": float(np.median(array)),
                "total_bytes": float(array.sum()),
            }
        )
        all_sizes.extend(sizes)
        print(f"  {bag.parent.name}: {array.size} frames, mean {array.mean()/1024:.1f} KiB")

    pooled = np.asarray(all_sizes, dtype=np.float64)
    summary = {
        "topic": TOPIC,
        "bags_available": len(bags),
        "bags_sampled": len(per_bag),
        "frames": int(pooled.size),
        "chroma_subsampling": sorted(subsampling),
        "decoded_size": sorted(dimensions),
        "bytes_mean": float(pooled.mean()),
        "bytes_median": float(np.median(pooled)),
        "bytes_p05": float(np.percentile(pooled, 5)),
        "bytes_p95": float(np.percentile(pooled, 95)),
        "bytes_min": float(pooled.min()),
        "bytes_max": float(pooled.max()),
        "bitrate_mbps_at_30fps": float(pooled.mean() * 8 * 30 / 1e6),
        "per_bag": per_bag,
    }
    write_json(ROOT / "results" / "reference_mcap_sizes.json", summary)
    print(f"pooled mean {pooled.mean()/1024:.1f} KiB over {pooled.size} frames, "
          f"subsampling={sorted(subsampling)} size={sorted(dimensions)}")


if __name__ == "__main__":
    main()
