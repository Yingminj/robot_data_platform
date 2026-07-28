#!/usr/bin/env python3
"""Encode the HDF5 top camera using LeRobot's AV1 settings and evaluate pixels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import av
import cv2
import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def sha256(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stream_info(path: Path) -> dict:
    with av.open(str(path), "r") as container:
        stream = container.streams.video[0]
        codec = stream.codec_context
        return {
            "codec": codec.name,
            "pix_fmt": codec.pix_fmt,
            "width": int(stream.width),
            "height": int(stream.height),
            "average_rate": str(stream.average_rate),
            "base_rate": str(stream.base_rate),
            "time_base": str(stream.time_base),
            "frames_metadata": int(stream.frames),
            "duration_seconds": (
                float(stream.duration * stream.time_base) if stream.duration is not None else None
            ),
        }


def encode_hdf5_dataset(
    dataset: h5py.Dataset,
    output: Path,
    *,
    fps: int,
    crf: int,
    preset: int,
    gop: int,
    pix_fmt: str,
) -> float:
    """Mirror LeRobot's PyAV RGB encoding path without a PNG round-trip."""
    output.parent.mkdir(parents=True, exist_ok=True)
    height, width = map(int, dataset.shape[1:3])
    options = {"g": str(gop), "preset": str(preset)}
    # FFmpeg's libsvtav1 wrapper treats crf=0 as its "not set" sentinel and
    # silently falls back to SVT's CRF 35. LeRobot passes CRF through this
    # wrapper. Preserve the requested CRF 0 by sending it directly to SVT;
    # positive CRFs use the same wrapper option as LeRobot.
    if crf == 0:
        options["svtav1-params"] = "fast-decode=0:crf=0"
    else:
        options["crf"] = str(crf)
        options["svtav1-params"] = "fast-decode=0"
    start = time.perf_counter()
    with av.open(str(output), "w", options={"movflags": "faststart"}) as container:
        stream = container.add_stream("libsvtav1", fps, options=options)
        stream.width = width
        stream.height = height
        stream.pix_fmt = pix_fmt
        for frame_rgb in dataset:
            frame = av.VideoFrame.from_ndarray(np.asarray(frame_rgb), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return time.perf_counter() - start


def decode_rgb_frames(path: Path):
    with av.open(str(path), "r") as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            yield frame.to_ndarray(format="rgb24")


def ssim_rgb(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Wang et al. SSIM with the common 11x11 Gaussian window, averaged over RGB."""
    x = reference.astype(np.float64)
    y = candidate.astype(np.float64)
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    scores = []
    for channel in range(3):
        xc = x[..., channel]
        yc = y[..., channel]
        mu_x = cv2.GaussianBlur(xc, (11, 11), 1.5)
        mu_y = cv2.GaussianBlur(yc, (11, 11), 1.5)
        mu_x2 = mu_x * mu_x
        mu_y2 = mu_y * mu_y
        mu_xy = mu_x * mu_y
        sigma_x2 = cv2.GaussianBlur(xc * xc, (11, 11), 1.5) - mu_x2
        sigma_y2 = cv2.GaussianBlur(yc * yc, (11, 11), 1.5) - mu_y2
        sigma_xy = cv2.GaussianBlur(xc * yc, (11, 11), 1.5) - mu_xy
        numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
        denominator = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
        scores.append(float(np.mean(numerator / denominator)))
    return float(np.mean(scores))


def frame_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict:
    delta = candidate.astype(np.int16) - reference.astype(np.int16)
    abs_delta = np.abs(delta)
    squared = delta.astype(np.float64) ** 2
    mse_255 = float(squared.mean())
    psnr_db = math.inf if mse_255 == 0 else 10.0 * math.log10((255.0**2) / mse_255)
    return {
        "mse_255": mse_255,
        "mse_normalized": mse_255 / (255.0**2),
        "rmse_255": math.sqrt(mse_255),
        "mae_255": float(abs_delta.mean()),
        "psnr_db": psnr_db,
        "ssim": ssim_rgb(reference, candidate),
        "max_abs_error": int(abs_delta.max()),
        "p95_abs_error": float(np.percentile(abs_delta, 95)),
        "p99_abs_error": float(np.percentile(abs_delta, 99)),
        "unchanged_channel_fraction": float(np.mean(abs_delta == 0)),
        "unchanged_pixel_fraction": float(np.mean(np.all(abs_delta == 0, axis=2))),
        "mae_r": float(abs_delta[..., 0].mean()),
        "mae_g": float(abs_delta[..., 1].mean()),
        "mae_b": float(abs_delta[..., 2].mean()),
        "bias_r": float(delta[..., 0].mean()),
        "bias_g": float(delta[..., 1].mean()),
        "bias_b": float(delta[..., 2].mean()),
    }


def summarize_rows(rows: list[dict], keys: list[str]) -> dict:
    summary = {}
    for key in keys:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        summary[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "median": float(np.median(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "p05": float(np.percentile(values, 5)),
            "p95": float(np.percentile(values, 95)),
        }
    return summary


def save_worst_frame_figure(
    reference: np.ndarray,
    candidate: np.ndarray,
    output: Path,
    title: str,
) -> None:
    abs_diff = np.abs(candidate.astype(np.int16) - reference.astype(np.int16)).astype(np.uint8)
    amplified = np.clip(abs_diff.astype(np.uint16) * 4, 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(abs_diff.max(axis=2), cv2.COLORMAP_TURBO)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    panel = np.concatenate([reference, candidate, amplified, heat], axis=1)
    panel_bgr = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
    labels = ["original", "decoded", "|diff| x4", "max-channel heatmap"]
    for i, label in enumerate(labels):
        cv2.rectangle(panel_bgr, (i * reference.shape[1], 0), ((i + 1) * reference.shape[1], 42), (0, 0, 0), -1)
        cv2.putText(
            panel_bgr,
            label,
            (i * reference.shape[1] + 12, 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    cv2.putText(
        panel_bgr,
        title,
        (12, panel_bgr.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(output), panel_bgr)


def evaluate_one(
    dataset: h5py.Dataset,
    video_path: Path,
    label: str,
    output_dir: Path,
) -> tuple[list[dict], float]:
    rows = []
    candidates = decode_rgb_frames(video_path)
    decode_start = time.perf_counter()
    worst = None
    for index in range(dataset.shape[0]):
        reference = np.asarray(dataset[index])
        try:
            candidate = next(candidates)
        except StopIteration as exc:
            raise RuntimeError(f"{video_path} ended before frame {index}") from exc
        if reference.shape != candidate.shape:
            raise RuntimeError(f"Shape mismatch at frame {index}: {reference.shape} vs {candidate.shape}")
        row = {"frame_index": index, "variant": label, **frame_metrics(reference, candidate)}
        rows.append(row)
        if worst is None or row["ssim"] < worst[0]:
            worst = (row["ssim"], index, reference.copy(), candidate.copy())
    try:
        extra = next(candidates)
        raise RuntimeError(f"{video_path} contains extra frame with shape {extra.shape}")
    except StopIteration:
        pass
    decode_seconds = time.perf_counter() - decode_start
    assert worst is not None
    save_worst_frame_figure(
        worst[2],
        worst[3],
        output_dir / f"worst_pixel_frame_{label}.png",
        f"{label}: frame {worst[1]}, SSIM={worst[0]:.6f}",
    )
    return rows, decode_seconds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--force", action="store_true", help="Re-encode videos even if they exist.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    source = Path(cfg["source_hdf5"])
    videos_dir = ROOT / "videos"
    results_dir = ROOT / "results"
    figures_dir = ROOT / "figures"
    for directory in (videos_dir, results_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)

    variants = cfg["encoder"]["quality_levels"]
    fps = int(cfg["fps"])
    encoder = cfg["encoder"]
    all_rows = []
    summary = {
        "source": {
            "path": str(source),
            "file_size_bytes": source.stat().st_size,
            "sha256": sha256(source),
            "camera_key": cfg["camera_key"],
            "source_color_order": cfg["source_color_order"],
        },
        "configuration": cfg,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "h5py": h5py.__version__,
            "pyav": av.__version__,
            "ffmpeg_libraries": {k: list(v) for k, v in av.library_versions.items()},
        },
        "variants": {},
    }
    metric_keys = [
        "mse_255",
        "mse_normalized",
        "rmse_255",
        "mae_255",
        "psnr_db",
        "ssim",
        "max_abs_error",
        "p95_abs_error",
        "p99_abs_error",
        "unchanged_channel_fraction",
        "unchanged_pixel_fraction",
        "mae_r",
        "mae_g",
        "mae_b",
        "bias_r",
        "bias_g",
        "bias_b",
    ]

    with h5py.File(source, "r") as h5:
        dataset = h5[cfg["camera_key"]]
        if dataset.dtype != np.uint8 or dataset.ndim != 4 or dataset.shape[-1] != 3:
            raise ValueError(f"Expected uint8 NHWC RGB dataset, got {dataset.dtype} {dataset.shape}")
        raw_logical = int(dataset.size * dataset.dtype.itemsize)
        raw_stored = int(dataset.id.get_storage_size())
        summary["source"].update(
            {
                "frame_count": int(dataset.shape[0]),
                "height": int(dataset.shape[1]),
                "width": int(dataset.shape[2]),
                "channels": int(dataset.shape[3]),
                "dtype": str(dataset.dtype),
                "hdf5_compression": dataset.compression,
                "hdf5_chunks": list(dataset.chunks) if dataset.chunks else None,
                "camera_logical_bytes": raw_logical,
                "camera_hdf5_storage_bytes": raw_stored,
            }
        )
        for label, crf in variants.items():
            video_path = videos_dir / f"top_{label}.mp4"
            if args.force or not video_path.exists():
                encode_seconds = encode_hdf5_dataset(
                    dataset,
                    video_path,
                    fps=fps,
                    crf=int(crf),
                    preset=int(encoder["preset"]),
                    gop=int(encoder["g"]),
                    pix_fmt=encoder["pix_fmt"],
                )
            else:
                encode_seconds = None
            rows, decode_seconds = evaluate_one(dataset, video_path, label, figures_dir)
            all_rows.extend(rows)
            video_bytes = video_path.stat().st_size
            summary["variants"][label] = {
                "crf": int(crf),
                "encoder_options": {
                    "vcodec": "libsvtav1",
                    "pix_fmt": encoder["pix_fmt"],
                    "g": int(encoder["g"]),
                    "requested_preset": int(encoder["preset"]),
                    "effective_preset_observed_in_svt_log": 10,
                    "fast_decode": 0,
                    "crf_transport": (
                        "svtav1-params=crf=0 (FFmpeg wrapper workaround)"
                        if int(crf) == 0
                        else "FFmpeg/PyAV crf option (same as LeRobot)"
                    ),
                },
                "path": str(video_path),
                "sha256": sha256(video_path),
                "video_size_bytes": video_bytes,
                "ratio_vs_raw_logical": video_bytes / raw_logical,
                "space_saving_vs_raw_logical": 1.0 - video_bytes / raw_logical,
                "ratio_vs_hdf5_camera_storage": video_bytes / raw_stored,
                "encode_seconds": encode_seconds,
                "encode_fps": (dataset.shape[0] / encode_seconds if encode_seconds else None),
                "sequential_decode_seconds": decode_seconds,
                "sequential_decode_fps": dataset.shape[0] / decode_seconds,
                "stream": stream_info(video_path),
                "pixel_metrics": summarize_rows(rows, metric_keys),
                "worst_ssim_frame": int(min(rows, key=lambda row: row["ssim"])["frame_index"]),
                "worst_psnr_frame": int(min(rows, key=lambda row: row["psnr_db"])["frame_index"]),
            }

    csv_path = results_dir / "per_frame_pixel_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    with (results_dir / "pixel_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps(summary["variants"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
