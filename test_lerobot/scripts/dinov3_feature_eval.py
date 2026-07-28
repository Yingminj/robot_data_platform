#!/usr/bin/env python3
"""Measure DINOv3 feature drift caused by the two AV1 compression levels."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
import time
from pathlib import Path

# Reuse the known-compatible h5py/PyAV wheels without modifying the DINO environment.
TEST_SITE_PACKAGES = Path("/home/kewei/anaconda3/envs/test/lib/python3.10/site-packages")
if TEST_SITE_PACKAGES.exists():
    sys.path.insert(0, str(TEST_SITE_PACKAGES))

import av
import cv2
import h5py
import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def decoded_batches(path: Path, batch_size: int):
    batch = []
    with av.open(str(path), "r") as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            batch.append(frame.to_ndarray(format="rgb24"))
            if len(batch) == batch_size:
                yield batch
                batch = []
    if batch:
        yield batch


def hdf5_batches(dataset: h5py.Dataset, batch_size: int):
    for start in range(0, dataset.shape[0], batch_size):
        yield [np.asarray(frame) for frame in dataset[start : start + batch_size]]


def cosine(a: torch.Tensor, b: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return F.cosine_similarity(a.float(), b.float(), dim=dim, eps=1e-8)


def relative_l2(reference: torch.Tensor, candidate: torch.Tensor, dim: int = -1) -> torch.Tensor:
    numerator = torch.linalg.vector_norm((candidate - reference).float(), dim=dim)
    denominator = torch.linalg.vector_norm(reference.float(), dim=dim).clamp_min(1e-8)
    return numerator / denominator


def summarize(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
        "p05": float(np.percentile(array, 5)),
        "p95": float(np.percentile(array, 95)),
    }


def extract(model, transform, frames: list[np.ndarray], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = torch.stack([transform(frame) for frame in frames]).to(device, non_blocking=True)
    with torch.inference_mode():
        output = model.forward_features(inputs)
    return output["x_norm_clstoken"].cpu(), output["x_norm_patchtokens"].cpu()


def decode_frame(path: Path, target_index: int) -> np.ndarray:
    with av.open(str(path), "r") as container:
        for index, frame in enumerate(container.decode(container.streams.video[0])):
            if index == target_index:
                return frame.to_ndarray(format="rgb24")
    raise IndexError(f"Frame {target_index} not found in {path}")


def save_feature_worst_panel(
    reference: np.ndarray,
    candidate: np.ndarray,
    output: Path,
    *,
    label: str,
    frame_index: int,
    cls_cosine: float,
) -> None:
    abs_diff = np.abs(candidate.astype(np.int16) - reference.astype(np.int16)).astype(np.uint8)
    amplified = np.clip(abs_diff.astype(np.uint16) * 4, 0, 255).astype(np.uint8)
    panel = np.concatenate([reference, candidate, amplified], axis=1)
    panel = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
    for i, text in enumerate(["original", "decoded", "|diff| x4"]):
        x0 = i * reference.shape[1]
        cv2.rectangle(panel, (x0, 0), (x0 + reference.shape[1], 42), (0, 0, 0), -1)
        cv2.putText(panel, text, (x0 + 12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
    cv2.putText(
        panel,
        f"{label}: worst CLS frame {frame_index}, cosine={cls_cosine:.6f}",
        (12, panel.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
    )
    cv2.imwrite(str(output), panel)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()
    cfg = load_config(args.config)
    dino_cfg = cfg["dinov3"]
    repo = Path(dino_cfg["repo"])
    checkpoint = Path(dino_cfg["checkpoint"])
    source = Path(cfg["source_hdf5"])
    videos = {
        label: ROOT / "videos" / f"top_{label}.mp4"
        for label in cfg["encoder"]["quality_levels"]
    }
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    sys.path.insert(0, str(repo))
    from dinov3.data.transforms import make_eval_transform

    transform = make_eval_transform(
        resize_size=int(dino_cfg["resize_size"]),
        crop_size=int(dino_cfg["crop_size"]),
    )
    load_start = time.perf_counter()
    model = torch.hub.load(
        str(repo),
        dino_cfg["model"],
        source="local",
        weights=str(checkpoint),
    )
    model.eval().to(device)
    load_seconds = time.perf_counter() - load_start
    batch_size = int(dino_cfg["batch_size"])

    rows = []
    embeddings = {}
    inference_start = time.perf_counter()
    with h5py.File(source, "r") as h5:
        dataset = h5[cfg["camera_key"]]
        for label, video_path in videos.items():
            frame_index = 0
            original_cls_all = []
            decoded_cls_all = []
            original_patch_mean_all = []
            decoded_patch_mean_all = []
            original_iter = hdf5_batches(dataset, batch_size)
            decoded_iter = decoded_batches(video_path, batch_size)
            while True:
                try:
                    original_batch = next(original_iter)
                except StopIteration:
                    break
                try:
                    decoded_batch = next(decoded_iter)
                except StopIteration as exc:
                    raise RuntimeError(f"{label} video ended early") from exc
                if len(original_batch) != len(decoded_batch):
                    raise RuntimeError(f"{label} batch length mismatch")
                original_cls, original_patch = extract(model, transform, original_batch, device)
                decoded_cls, decoded_patch = extract(model, transform, decoded_batch, device)
                cls_cos = cosine(original_cls, decoded_cls)
                cls_rel_l2 = relative_l2(original_cls, decoded_cls)
                original_patch_mean = original_patch.mean(dim=1)
                decoded_patch_mean = decoded_patch.mean(dim=1)
                pooled_cos = cosine(original_patch_mean, decoded_patch_mean)
                pooled_rel_l2 = relative_l2(original_patch_mean, decoded_patch_mean)
                token_cos = cosine(original_patch, decoded_patch)
                token_rel_l2 = relative_l2(original_patch, decoded_patch)

                original_cls_all.append(original_cls.numpy())
                decoded_cls_all.append(decoded_cls.numpy())
                original_patch_mean_all.append(original_patch_mean.numpy())
                decoded_patch_mean_all.append(decoded_patch_mean.numpy())
                for i in range(len(original_batch)):
                    token_cos_i = token_cos[i].numpy()
                    token_l2_i = token_rel_l2[i].numpy()
                    bounded_cos = float(np.clip(cls_cos[i].item(), -1.0, 1.0))
                    rows.append(
                        {
                            "frame_index": frame_index,
                            "variant": label,
                            "cls_cosine": bounded_cos,
                            "cls_angle_degrees": math.degrees(math.acos(bounded_cos)),
                            "cls_relative_l2": float(cls_rel_l2[i]),
                            "mean_patch_cosine": float(pooled_cos[i]),
                            "mean_patch_relative_l2": float(pooled_rel_l2[i]),
                            "patch_token_cosine_mean": float(token_cos_i.mean()),
                            "patch_token_cosine_p05": float(np.percentile(token_cos_i, 5)),
                            "patch_token_cosine_min": float(token_cos_i.min()),
                            "patch_token_relative_l2_mean": float(token_l2_i.mean()),
                            "patch_token_relative_l2_p95": float(np.percentile(token_l2_i, 95)),
                            "patch_token_relative_l2_max": float(token_l2_i.max()),
                        }
                    )
                    frame_index += 1
            try:
                next(decoded_iter)
                raise RuntimeError(f"{label} video contains extra frames")
            except StopIteration:
                pass
            if frame_index != dataset.shape[0]:
                raise RuntimeError(f"{label}: evaluated {frame_index}, expected {dataset.shape[0]}")
            embeddings[f"{label}__original_cls"] = np.concatenate(original_cls_all)
            embeddings[f"{label}__decoded_cls"] = np.concatenate(decoded_cls_all)
            embeddings[f"{label}__original_mean_patch"] = np.concatenate(original_patch_mean_all)
            embeddings[f"{label}__decoded_mean_patch"] = np.concatenate(decoded_patch_mean_all)
    inference_seconds = time.perf_counter() - inference_start

    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    with (results_dir / "per_frame_dinov3_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(results_dir / "dinov3_global_embeddings.npz", **embeddings)

    metric_keys = [key for key in rows[0] if key not in {"frame_index", "variant"}]
    summary = {
        "model": {
            "name": dino_cfg["model"],
            "repo": str(repo),
            "checkpoint": str(checkpoint),
            "checkpoint_size_bytes": checkpoint.stat().st_size,
            "embedding_dimension": int(next(iter(embeddings.values())).shape[1]),
            "patch_grid": [
                int(dino_cfg["crop_size"]) // 16,
                int(dino_cfg["crop_size"]) // 16,
            ],
        },
        "preprocessing": {
            "resize_shorter_side": int(dino_cfg["resize_size"]),
            "center_crop": int(dino_cfg["crop_size"]),
            "interpolation": "torchvision InterpolationMode.BICUBIC",
            "rgb_scale": "[0, 1]",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "runtime": {
            "device": str(device),
            "torch": torch.__version__,
            "python": sys.version,
            "platform": platform.platform(),
            "model_load_seconds": load_seconds,
            "inference_seconds": inference_seconds,
            "frames_per_variant": int(len(rows) / len(videos)),
        },
        "variants": {},
    }
    for label in videos:
        selected = [row for row in rows if row["variant"] == label]
        summary["variants"][label] = {
            key: summarize([row[key] for row in selected]) for key in metric_keys
        }
        summary["variants"][label]["worst_cls_cosine_frame"] = int(
            min(selected, key=lambda row: row["cls_cosine"])["frame_index"]
        )
        summary["variants"][label]["worst_patch_cosine_frame"] = int(
            min(selected, key=lambda row: row["patch_token_cosine_mean"])["frame_index"]
        )
    figures_dir = ROOT / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(source, "r") as h5:
        dataset = h5[cfg["camera_key"]]
        for label, video_path in videos.items():
            target_row = min(
                (row for row in rows if row["variant"] == label),
                key=lambda row: row["cls_cosine"],
            )
            target = int(target_row["frame_index"])
            save_feature_worst_panel(
                np.asarray(dataset[target]),
                decode_frame(video_path, target),
                figures_dir / f"worst_dinov3_cls_frame_{label}.png",
                label=label,
                frame_index=target,
                cls_cosine=float(target_row["cls_cosine"]),
            )
    with (results_dir / "dinov3_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
