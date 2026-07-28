#!/usr/bin/env python3
"""Compare compression-induced feature drift across DINOv3 and DINOv2 models."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import platform
import sys
import time
from pathlib import Path

# Reuse the known-compatible h5py/PyAV wheels without modifying environments.
TEST_SITE_PACKAGES = Path("/home/kewei/anaconda3/envs/test/lib/python3.10/site-packages")
if TEST_SITE_PACKAGES.exists():
    sys.path.insert(0, str(TEST_SITE_PACKAGES))

import av
import cv2
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import v2


ROOT = Path(__file__).resolve().parents[1]
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


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


def make_transform(resize_size: int, crop_size: int) -> v2.Compose:
    return v2.Compose(
        [
            v2.ToImage(),
            v2.Resize(resize_size, interpolation=v2.InterpolationMode.BICUBIC),
            v2.CenterCrop(crop_size),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=MEAN, std=STD),
        ]
    )


def load_model(model_cfg: dict, device: torch.device) -> tuple[torch.nn.Module, float]:
    """Build architecture locally and load the supplied checkpoint without cache writes."""
    start = time.perf_counter()
    model = torch.hub.load(
        model_cfg["repo"],
        model_cfg["entrypoint"],
        source="local",
        pretrained=False,
    )
    state_dict = torch.load(model_cfg["checkpoint"], map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval().to(device)
    return model, time.perf_counter() - start


def extract(
    model: torch.nn.Module,
    transform: v2.Compose,
    frames: list[np.ndarray],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = torch.stack([transform(frame) for frame in frames]).to(device, non_blocking=True)
    with torch.inference_mode():
        output = model.forward_features(inputs)
    return output["x_norm_clstoken"].cpu(), output["x_norm_patchtokens"].cpu()


def extract_references(
    model: torch.nn.Module,
    transform: v2.Compose,
    dataset: h5py.Dataset,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    cls_batches = []
    patch_batches = []
    for frames in hdf5_batches(dataset, batch_size):
        cls, patch = extract(model, transform, frames, device)
        cls_batches.append(cls)
        patch_batches.append(patch)
    return torch.cat(cls_batches), torch.cat(patch_batches)


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
    model_name: str,
    variant: str,
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
        f"{model_name}, {variant}: frame {frame_index}, CLS cosine={cls_cosine:.6f}",
        (12, panel.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (255, 255, 255),
        2,
    )
    cv2.imwrite(str(output), panel)


def center_crop_visual(frame_rgb: np.ndarray, resize_size: int, crop_size: int) -> np.ndarray:
    height, width = frame_rgb.shape[:2]
    scale = resize_size / min(height, width)
    resized_width = int(round(width * scale))
    resized_height = int(round(height * scale))
    resized = cv2.resize(frame_rgb, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)
    x0 = (resized_width - crop_size) // 2
    y0 = (resized_height - crop_size) // 2
    return resized[y0 : y0 + crop_size, x0 : x0 + crop_size]


def save_pca_feature_map_panel(
    *,
    frame_rgb: np.ndarray,
    patch_arrays: dict[str, np.ndarray],
    frame_index: int,
    patch_grid: tuple[int, int],
    output: Path,
    model_name: str,
    resize_size: int,
    crop_size: int,
) -> None:
    """Project original and compressed patch tokens through one shared 3D PCA basis."""
    ordered_keys = ["original", "crf0_min_loss", "crf20_balanced", "crf50_max_loss"]
    per_source = [patch_arrays[key][frame_index].astype(np.float32, copy=False) for key in ordered_keys]
    token_count = per_source[0].shape[0]
    merged = torch.from_numpy(np.concatenate(per_source, axis=0))
    torch.manual_seed(frame_index)
    _, _, basis = torch.pca_lowrank(merged, q=3, center=True, niter=4)
    projected = (merged - merged.mean(dim=0, keepdim=True)) @ basis
    projected = projected.numpy()
    low = np.percentile(projected, 1, axis=0)
    high = np.percentile(projected, 99, axis=0)
    normalized = np.clip((projected - low) / np.maximum(high - low, 1e-8), 0.0, 1.0)
    maps = []
    for source_index in range(len(ordered_keys)):
        start = source_index * token_count
        patch_rgb = normalized[start : start + token_count].reshape(*patch_grid, 3)
        patch_rgb = (patch_rgb * 255.0).round().astype(np.uint8)
        maps.append(cv2.resize(patch_rgb, (crop_size, crop_size), interpolation=cv2.INTER_NEAREST))

    panels = [center_crop_visual(frame_rgb, resize_size, crop_size), *maps]
    labels = ["input crop", "original feature", "CRF 0", "CRF 20", "CRF 50"]
    canvas = np.concatenate(panels, axis=1)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    for index, label in enumerate(labels):
        x0 = index * crop_size
        cv2.rectangle(canvas, (x0, 0), (x0 + crop_size, 34), (0, 0, 0), -1)
        cv2.putText(
            canvas,
            label,
            (x0 + 8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        canvas,
        f"{model_name}, frame {frame_index}; shared PCA basis and color scaling",
        (8, canvas.shape[0] - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), canvas)


def compare_variant(
    *,
    model: torch.nn.Module,
    transform: v2.Compose,
    reference_cls: torch.Tensor,
    reference_patch: torch.Tensor,
    video_path: Path,
    batch_size: int,
    device: torch.device,
    model_cfg: dict,
    variant: str,
) -> tuple[list[dict], np.ndarray, np.ndarray, np.ndarray]:
    rows = []
    decoded_cls_batches = []
    decoded_patch_mean_batches = []
    decoded_patch_batches = []
    frame_index = 0
    for decoded_batch in decoded_batches(video_path, batch_size):
        decoded_cls, decoded_patch = extract(model, transform, decoded_batch, device)
        batch_len = decoded_cls.shape[0]
        ref_cls = reference_cls[frame_index : frame_index + batch_len]
        ref_patch = reference_patch[frame_index : frame_index + batch_len]
        if ref_cls.shape[0] != batch_len:
            raise RuntimeError(f"{variant} contains more frames than the HDF5 source")

        cls_cos = cosine(ref_cls, decoded_cls)
        cls_rel_l2 = relative_l2(ref_cls, decoded_cls)
        ref_patch_mean = ref_patch.mean(dim=1)
        decoded_patch_mean = decoded_patch.mean(dim=1)
        pooled_cos = cosine(ref_patch_mean, decoded_patch_mean)
        pooled_rel_l2 = relative_l2(ref_patch_mean, decoded_patch_mean)
        token_cos = cosine(ref_patch, decoded_patch)
        token_rel_l2 = relative_l2(ref_patch, decoded_patch)
        decoded_cls_batches.append(decoded_cls.numpy())
        decoded_patch_mean_batches.append(decoded_patch_mean.numpy())
        decoded_patch_batches.append(decoded_patch.numpy())

        for i in range(batch_len):
            token_cos_i = token_cos[i].numpy()
            token_l2_i = token_rel_l2[i].numpy()
            bounded_cos = float(np.clip(cls_cos[i].item(), -1.0, 1.0))
            rows.append(
                {
                    "model_id": model_cfg["id"],
                    "model_name": model_cfg["display_name"],
                    "frame_index": frame_index + i,
                    "variant": variant,
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
        frame_index += batch_len
    if frame_index != reference_cls.shape[0]:
        raise RuntimeError(f"{variant}: evaluated {frame_index}, expected {reference_cls.shape[0]}")
    return (
        rows,
        np.concatenate(decoded_cls_batches),
        np.concatenate(decoded_patch_mean_batches),
        np.concatenate(decoded_patch_batches),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--models", nargs="*", help="Optional model IDs to evaluate.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    feature_cfg = cfg["feature_evaluation"]
    models_cfg = feature_cfg["models"]
    if args.models:
        requested = set(args.models)
        models_cfg = [model for model in models_cfg if model["id"] in requested]
        missing = requested - {model["id"] for model in models_cfg}
        if missing:
            raise ValueError(f"Unknown model IDs: {sorted(missing)}")
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
    transform = make_transform(
        resize_size=int(feature_cfg["resize_size"]),
        crop_size=int(feature_cfg["crop_size"]),
    )

    all_rows = []
    embeddings = {}
    summary = {
        "preprocessing": {
            "resize_shorter_side": int(feature_cfg["resize_size"]),
            "center_crop": int(feature_cfg["crop_size"]),
            "interpolation": "torchvision InterpolationMode.BICUBIC",
            "rgb_scale": "[0, 1]",
            "mean": list(MEAN),
            "std": list(STD),
        },
        "runtime": {
            "device": str(device),
            "torch": torch.__version__,
            "python": sys.version,
            "platform": platform.platform(),
        },
        "models": {},
    }
    metric_keys = [
        "cls_cosine",
        "cls_angle_degrees",
        "cls_relative_l2",
        "mean_patch_cosine",
        "mean_patch_relative_l2",
        "patch_token_cosine_mean",
        "patch_token_cosine_p05",
        "patch_token_cosine_min",
        "patch_token_relative_l2_mean",
        "patch_token_relative_l2_p95",
        "patch_token_relative_l2_max",
    ]
    figures_dir = ROOT / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    feature_maps_dir = ROOT / "results" / "feature_maps"
    feature_maps_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(source, "r") as h5:
        dataset = h5[cfg["camera_key"]]
        for model_cfg in models_cfg:
            checkpoint = Path(model_cfg["checkpoint"])
            model, load_seconds = load_model(model_cfg, device)
            batch_size = int(model_cfg["batch_size"])
            inference_start = time.perf_counter()
            reference_cls, reference_patch = extract_references(
                model, transform, dataset, batch_size, device
            )
            model_rows = []
            embeddings[f"{model_cfg['id']}__original_cls"] = reference_cls.numpy()
            embeddings[f"{model_cfg['id']}__original_mean_patch"] = reference_patch.mean(dim=1).numpy()
            patch_arrays = {"original": reference_patch.numpy()}
            variants_summary = {}
            for variant, video_path in videos.items():
                rows, decoded_cls, decoded_patch_mean, decoded_patch = compare_variant(
                    model=model,
                    transform=transform,
                    reference_cls=reference_cls,
                    reference_patch=reference_patch,
                    video_path=video_path,
                    batch_size=batch_size,
                    device=device,
                    model_cfg=model_cfg,
                    variant=variant,
                )
                all_rows.extend(rows)
                model_rows.extend(rows)
                embeddings[f"{model_cfg['id']}__{variant}__decoded_cls"] = decoded_cls
                embeddings[f"{model_cfg['id']}__{variant}__decoded_mean_patch"] = decoded_patch_mean
                patch_arrays[variant] = decoded_patch
                variant_summary = {
                    key: summarize([row[key] for row in rows]) for key in metric_keys
                }
                worst_cls = min(rows, key=lambda row: row["cls_cosine"])
                variant_summary["worst_cls_cosine_frame"] = int(worst_cls["frame_index"])
                variant_summary["worst_patch_cosine_frame"] = int(
                    min(rows, key=lambda row: row["patch_token_cosine_mean"])["frame_index"]
                )
                variants_summary[variant] = variant_summary
                target = int(worst_cls["frame_index"])
                save_feature_worst_panel(
                    np.asarray(dataset[target]),
                    decode_frame(video_path, target),
                    figures_dir / f"worst_cls_{model_cfg['id']}_{variant}.png",
                    model_name=model_cfg["display_name"],
                    variant=variant,
                    frame_index=target,
                    cls_cosine=float(worst_cls["cls_cosine"]),
                )

            inference_seconds = time.perf_counter() - inference_start
            patch_grid = (
                int(feature_cfg["crop_size"]) // int(model_cfg["patch_size"]),
                int(feature_cfg["crop_size"]) // int(model_cfg["patch_size"]),
            )
            selected_frames = sorted(
                {
                    0,
                    dataset.shape[0] // 4,
                    dataset.shape[0] // 2,
                    3 * dataset.shape[0] // 4,
                    dataset.shape[0] - 1,
                    *[
                        int(variants_summary[variant]["worst_cls_cosine_frame"])
                        for variant in videos
                    ],
                }
            )
            pca_dir = figures_dir / "feature_maps" / model_cfg["id"]
            for frame_index in selected_frames:
                save_pca_feature_map_panel(
                    frame_rgb=np.asarray(dataset[frame_index]),
                    patch_arrays=patch_arrays,
                    frame_index=frame_index,
                    patch_grid=patch_grid,
                    output=pca_dir / f"frame_{frame_index:04d}_pca.png",
                    model_name=model_cfg["display_name"],
                    resize_size=int(feature_cfg["resize_size"]),
                    crop_size=int(feature_cfg["crop_size"]),
                )
            patch_map_path = feature_maps_dir / f"{model_cfg['id']}_patch_tokens_float16.npz"
            np.savez_compressed(
                patch_map_path,
                **{key: value.astype(np.float16) for key, value in patch_arrays.items()},
            )
            summary["models"][model_cfg["id"]] = {
                "display_name": model_cfg["display_name"],
                "family": model_cfg["family"],
                "entrypoint": model_cfg["entrypoint"],
                "repo": model_cfg["repo"],
                "checkpoint": str(checkpoint),
                "checkpoint_size_bytes": checkpoint.stat().st_size,
                "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
                "embedding_dimension": int(reference_cls.shape[1]),
                "patch_size": int(model_cfg["patch_size"]),
                "patch_grid": list(patch_grid),
                "patch_token_count": int(reference_patch.shape[1]),
                "batch_size": batch_size,
                "model_load_seconds": load_seconds,
                "inference_seconds": inference_seconds,
                "frames_per_variant": int(dataset.shape[0]),
                "patch_feature_maps": {
                    "path": str(patch_map_path),
                    "dtype": "float16",
                    "keys": ["original", *videos.keys()],
                    "shape_per_key": list(patch_arrays["original"].shape),
                    "pca_visualization_dir": str(pca_dir),
                    "pca_visualization_frames": selected_frames,
                },
                "variants": variants_summary,
            }
            del model, reference_cls, reference_patch, patch_arrays
            gc.collect()

    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "per_frame_model_feature_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    np.savez_compressed(results_dir / "model_global_embeddings.npz", **embeddings)
    with (results_dir / "model_feature_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
