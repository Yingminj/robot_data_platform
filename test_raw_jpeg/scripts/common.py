"""Shared metrics and paths for the raw-vs-JPEG image quality test.

The pixel metric implementations are copied from
``test_lerobot/scripts/encode_and_pixel_eval.py`` on purpose: the two reports
quote the same numbers, so they must come from the same estimator.
"""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CONFIG = json.loads((ROOT / "config.json").read_text())
CAMERAS: list[str] = CONFIG["cameras"]


def memmap_path(kind: str, camera: str) -> Path:
    return ROOT / "intermediate" / f"{kind}__{camera}.npy"


def open_memmap(kind: str, camera: str, mode: str, shape: tuple[int, ...] | None = None):
    path = memmap_path(kind, camera)
    if mode == "w+":
        assert shape is not None
        return np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8, shape=shape)
    return np.load(path, mmap_mode="r")


# --------------------------------------------------------------------------
# pixel metrics
# --------------------------------------------------------------------------


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
        "rmse_255": math.sqrt(mse_255),
        "mae_255": float(abs_delta.mean()),
        "psnr_db": psnr_db,
        "ssim": ssim_rgb(reference, candidate),
        "max_abs_error": int(abs_delta.max()),
        "p95_abs_error": float(np.percentile(abs_delta, 95)),
        "p99_abs_error": float(np.percentile(abs_delta, 99)),
        "unchanged_pixel_fraction": float(np.mean(np.all(abs_delta == 0, axis=2))),
        "mae_r": float(abs_delta[..., 0].mean()),
        "mae_g": float(abs_delta[..., 1].mean()),
        "mae_b": float(abs_delta[..., 2].mean()),
        "bias_r": float(delta[..., 0].mean()),
        "bias_g": float(delta[..., 1].mean()),
        "bias_b": float(delta[..., 2].mean()),
    }


METRIC_KEYS = [
    "mse_255",
    "rmse_255",
    "mae_255",
    "psnr_db",
    "ssim",
    "max_abs_error",
    "p95_abs_error",
    "p99_abs_error",
    "unchanged_pixel_fraction",
]


def summarize_rows(rows: list[dict], keys: list[str]) -> dict:
    summary = {}
    for key in keys:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            summary[key] = {"mean": math.inf, "median": math.inf, "min": math.inf, "max": math.inf}
            continue
        summary[key] = {
            "mean": float(np.mean(finite)),
            "std": float(np.std(finite)),
            "median": float(np.median(finite)),
            "min": float(np.min(finite)),
            "max": float(np.max(finite)),
            "p05": float(np.percentile(finite, 5)),
            "p95": float(np.percentile(finite, 95)),
            "infinite_frames": int(values.size - finite.size),
        }
    return summary


# --------------------------------------------------------------------------
# JPEG introspection
# --------------------------------------------------------------------------


def jpeg_chroma_subsampling(buffer: bytes) -> str:
    """Read the SOF0 marker and report the luma sampling factors as 4:x:x."""
    index = 2
    size = len(buffer)
    while index + 4 <= size:
        if buffer[index] != 0xFF:
            index += 1
            continue
        marker = buffer[index + 1]
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            index += 2
            continue
        (length,) = struct.unpack(">H", buffer[index + 2 : index + 4])
        if marker in (0xC0, 0xC1, 0xC2):
            payload = buffer[index + 4 : index + 2 + length]
            components = payload[5]
            factors = []
            for c in range(components):
                sampling = payload[6 + c * 3 + 1]
                factors.append((sampling >> 4, sampling & 0x0F))
            h, v = factors[0]
            if (h, v) == (2, 2):
                return "4:2:0"
            if (h, v) == (2, 1):
                return "4:2:2"
            if (h, v) == (1, 1):
                return "4:4:4"
            return f"h{h}v{v}"
        index += 2 + length
    return "unknown"


def encode_jpeg(mosaic_rgb: np.ndarray, quality: int) -> bytes:
    """Reproduce realsense_node.cpp: cv::imencode on the BGR mosaic canvas."""
    bgr = cv2.cvtColor(mosaic_rgb, cv2.COLOR_RGB2BGR)
    ok, buffer = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError(f"cv2.imencode failed at quality {quality}")
    return buffer.tobytes()


def decode_jpeg(buffer: bytes) -> np.ndarray:
    """Reproduce robot_data.ros.media._decode_compressed_image."""
    array = np.frombuffer(buffer, dtype=np.uint8)
    decoded = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if decoded is None:
        raise RuntimeError("cv2.imdecode failed")
    return np.ascontiguousarray(cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB), dtype=np.uint8)


def human_bytes(value: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(value) < 1024.0 or unit == "GiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} GiB"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
