#!/usr/bin/env python3
"""Stage 1: extract the raw ground-truth frames and measure recording-side JPEG.

Two passes over the same bag:

1. ``align_rosbag`` with the exact configuration the converter uses, so the
   ground-truth tiles are byte-identical to what ``rosbag2_to_lerobotv3.py``
   feeds its video encoder. Saved as one uint8 memmap per camera.
2. A second pass that decodes the *mosaic* messages the alignment selected and
   runs them through the recording-side JPEG encoder at each quality level.
   The production node compresses the whole 1280x1440 canvas before anyone
   crops tiles out of it (``realsense_node.cpp:333``), so the JPEG has to be
   applied at mosaic level, not per tile, or the block grid lands in the wrong
   place.

Pass 2 asserts that cropping the freshly decoded mosaic reproduces the pass-1
tiles exactly, which is what proves the two passes are looking at the same
frames.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/kewei/YING/robot_data_platform/tool")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from robot_data.align import AlignmentConfig, align_rosbag  # noqa: E402
from robot_data.align.bag_io import open_bag_reader, scan_bag  # noqa: E402
from robot_data.align.indexing import previous_indices  # noqa: E402
from robot_data.align.window import compute_window  # noqa: E402
from robot_data.profiles import load_profile  # noqa: E402
from robot_data.ros.media import decode_image  # noqa: E402

from common import (  # noqa: E402
    CAMERAS,
    CONFIG,
    METRIC_KEYS,
    ROOT,
    decode_jpeg,
    encode_jpeg,
    frame_metrics,
    jpeg_chroma_subsampling,
    open_memmap,
    summarize_rows,
    write_json,
)

JPEG_LEVELS: list[int] = CONFIG["jpeg"]["levels"]


def build_config() -> AlignmentConfig:
    alignment = CONFIG["alignment"]
    return AlignmentConfig(
        fps=CONFIG["fps"],
        mode=alignment["mode"],
        grid_anchor=alignment["grid_anchor"],
        state_tolerance_ms=alignment["state_tolerance_ms"],
        action_gap_policy=alignment["action_gap_policy"],
    )


def selected_mosaic_indices(bag: Path, profile, cfg: AlignmentConfig) -> np.ndarray:
    """Reproduce the converter's per-row choice of source mosaic message."""
    raw = scan_bag(bag, profile, cfg)
    window = compute_window(raw, cfg)
    anchor_times = raw.image_t[profile.resolved_anchor_camera]
    grid_ns = anchor_times[
        (anchor_times >= window.start_ns) & (anchor_times <= window.end_ns)
    ].astype(np.int64)
    topic_times = raw.image_t[profile.resolved_anchor_camera]
    indices, _, valid = previous_indices(topic_times, grid_ns)
    if not valid.all():
        raise RuntimeError("anchor camera has rows with no preceding frame")
    return indices


def decode_mosaics(bag: Path, profile, wanted: set[int]):
    """Yield ``(message_index, mosaic_rgb)`` for the selected /quad_tile messages."""
    topic = profile.cameras[CAMERAS[0]]
    reader = open_bag_reader(bag, profile)
    reader.open()
    try:
        connections = [c for c in reader.connections if c.topic == topic]
        counter = 0
        for connection, _, rawdata in reader.messages(connections=connections):
            index = counter
            counter += 1
            if index not in wanted:
                continue
            message = reader.deserialize(rawdata, connection.msgtype)
            yield index, decode_image(message, connection.msgtype)
    finally:
        reader.close()


def main() -> None:
    bag = Path(CONFIG["source_raw_bag"])
    profile = load_profile(CONFIG["profile"])
    cfg = build_config()
    results = ROOT / "results"

    print(f"profile={profile.name} cameras={CAMERAS} anchor={profile.resolved_anchor_camera}")

    # ---- pass 1: ground-truth tiles -------------------------------------
    print("[pass 1] aligning bag (this decodes every selected frame)…", flush=True)
    started = time.perf_counter()
    episode = align_rosbag(bag, profile, cfg)
    frames = episode.frame_count
    print(f"[pass 1] {frames} rows in {time.perf_counter() - started:.1f}s")

    tile_shape = None
    for camera in CAMERAS:
        stack = episode.images[camera]
        assert stack.shape[0] == frames, (camera, stack.shape)
        tile_shape = stack.shape[1:]
        out = open_memmap("raw", camera, "w+", shape=stack.shape)
        out[:] = stack
        out.flush()
        del out
        print(f"[pass 1] wrote raw tiles for {camera}: {stack.shape}")

    write_json(
        results / "episode_audit.json",
        {
            "frame_count": frames,
            "tile_shape": list(tile_shape),
            "fps": episode.fps,
            "audit": episode.audit,
        },
    )
    del episode

    # ---- pass 2: mosaic-level JPEG --------------------------------------
    indices = selected_mosaic_indices(bag, profile, cfg)
    if indices.size != frames:
        raise RuntimeError(f"index selection gave {indices.size} rows, alignment gave {frames}")
    wanted = set(int(i) for i in indices)
    print(f"[pass 2] {len(wanted)} unique mosaic messages back the {frames} rows")

    raw_tiles = {camera: open_memmap("raw", camera, "r") for camera in CAMERAS}
    jpeg_out = {
        quality: {
            camera: open_memmap(f"jpeg{quality}", camera, "w+", shape=raw_tiles[camera].shape)
            for camera in CAMERAS
        }
        for quality in JPEG_LEVELS
    }
    tiles = {camera: profile.camera_tiles[camera] for camera in CAMERAS}
    position_of = {int(index): row for row, index in enumerate(indices)}

    rows: list[dict] = []
    mosaic_rows: list[dict] = []
    byte_sizes = {quality: [] for quality in JPEG_LEVELS}
    subsampling: dict[int, str] = {}
    raw_mosaic_bytes = None
    mosaic_shape = None

    started = time.perf_counter()
    done = 0
    for index, mosaic in decode_mosaics(bag, profile, wanted):
        row = position_of[index]
        if mosaic_shape is None:
            mosaic_shape = mosaic.shape
            raw_mosaic_bytes = int(np.prod(mosaic.shape))
            print(f"[pass 2] mosaic {mosaic.shape}, raw {raw_mosaic_bytes} B/frame")
        # Consistency check: the crop of this mosaic must be the pass-1 tile.
        for camera in CAMERAS:
            expected = raw_tiles[camera][row]
            got = tiles[camera].apply(mosaic)
            if not np.array_equal(expected, got):
                raise RuntimeError(f"row {row} camera {camera}: pass-2 crop differs from pass-1")

        for quality in JPEG_LEVELS:
            buffer = encode_jpeg(mosaic, quality)
            byte_sizes[quality].append(len(buffer))
            if quality not in subsampling:
                subsampling[quality] = jpeg_chroma_subsampling(buffer)
            decoded = decode_jpeg(buffer)
            mosaic_rows.append(
                {
                    "frame_index": row,
                    "quality": quality,
                    "bytes": len(buffer),
                    **frame_metrics(mosaic, decoded),
                }
            )
            for camera in CAMERAS:
                tile = tiles[camera].apply(decoded)
                jpeg_out[quality][camera][row] = tile
                rows.append(
                    {
                        "frame_index": row,
                        "quality": quality,
                        "camera": camera,
                        **frame_metrics(raw_tiles[camera][row], tile),
                    }
                )
        done += 1
        if done % 50 == 0 or done == frames:
            rate = done / (time.perf_counter() - started)
            print(f"[pass 2] {done}/{frames} ({rate:.1f} fps)", flush=True)

    for quality in JPEG_LEVELS:
        for camera in CAMERAS:
            jpeg_out[quality][camera].flush()

    # ---- write results ---------------------------------------------------
    fieldnames = ["frame_index", "quality", "camera", *METRIC_KEYS, "mae_r", "mae_g", "mae_b",
                  "bias_r", "bias_g", "bias_b"]
    with (results / "per_frame_jpeg_tile_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    mosaic_fields = ["frame_index", "quality", "bytes", *METRIC_KEYS]
    with (results / "per_frame_jpeg_mosaic_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=mosaic_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(mosaic_rows)

    duration_s = frames / CONFIG["fps"]
    summary = {
        "frames": frames,
        "fps": CONFIG["fps"],
        "duration_s": duration_s,
        "mosaic_shape": list(mosaic_shape),
        "tile_shape": list(tile_shape),
        "raw_mosaic_bytes_per_frame": raw_mosaic_bytes,
        "raw_mosaic_total_bytes": raw_mosaic_bytes * frames,
        "raw_mosaic_bitrate_mbps": raw_mosaic_bytes * 8 * CONFIG["fps"] / 1e6,
        "raw_tiles_total_bytes": int(sum(int(np.prod(raw_tiles[c].shape)) for c in CAMERAS)),
        "jpeg": {},
        "per_camera": {},
        "mosaic_metrics": {},
    }
    for quality in JPEG_LEVELS:
        sizes = np.asarray(byte_sizes[quality], dtype=np.float64)
        summary["jpeg"][str(quality)] = {
            "chroma_subsampling": subsampling[quality],
            "bytes_mean": float(sizes.mean()),
            "bytes_median": float(np.median(sizes)),
            "bytes_min": float(sizes.min()),
            "bytes_max": float(sizes.max()),
            "total_bytes": float(sizes.sum()),
            "compression_ratio_vs_raw": float(raw_mosaic_bytes / sizes.mean()),
            "bitrate_mbps": float(sizes.mean() * 8 * CONFIG["fps"] / 1e6),
        }
        subset = [r for r in mosaic_rows if r["quality"] == quality]
        summary["mosaic_metrics"][str(quality)] = summarize_rows(subset, METRIC_KEYS)
        for camera in CAMERAS:
            subset = [r for r in rows if r["quality"] == quality and r["camera"] == camera]
            summary["per_camera"].setdefault(str(quality), {})[camera] = summarize_rows(
                subset, METRIC_KEYS
            )

    write_json(results / "jpeg_summary.json", summary)
    print(json.dumps(summary["jpeg"], indent=2))
    print("stage 1 complete")


if __name__ == "__main__":
    main()
