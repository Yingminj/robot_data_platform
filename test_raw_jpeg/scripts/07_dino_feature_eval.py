#!/usr/bin/env python3
"""Stage 7: DINO feature drift, aligned with test_lerobot/scripts/dinov3_feature_eval.py.

Same three models, same deterministic preprocessing (RGB, short side 256,
center crop 224, bicubic, ImageNet mean/std), same metric definitions, so the
cosines here can be read next to the ones in ``test_lerobot/REPORT.md``.

What differs is the experiment around it, not the estimator:

* eight variants instead of three -- two recording-side JPEG levels plus the
  2x3 {source, CRF} grid -- all measured against the same uncompressed frames;
* three cameras instead of one, which matters because the mosaic gives the head
  and the wrists very different pixel-level quality;
* 1103 frames instead of 220.

Frames come from the memmaps written by stages 1 and 6 rather than from videos,
so this script needs no PyAV and every model sees byte-identical inputs.
"""

from __future__ import annotations

import csv
import gc
import json
import math
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import v2

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import CAMERAS, CONFIG, ROOT, write_json  # noqa: E402

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
FEATURE_CFG = CONFIG["feature_evaluation"]
VARIANTS: dict[str, str] = FEATURE_CFG["variants"]

METRIC_KEYS = [
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


def load_memmap(kind: str, camera: str):
    return np.load(ROOT / "intermediate" / f"{kind}__{camera}.npy", mmap_mode="r")


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


def load_model(model_cfg: dict, device: torch.device):
    model = torch.hub.load(
        model_cfg["repo"], model_cfg["entrypoint"], source="local", pretrained=False
    )
    state_dict = torch.load(model_cfg["checkpoint"], map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval().to(device)
    return model


def extract(model, transform, frames, device) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = torch.stack([transform(frame) for frame in frames]).to(device, non_blocking=True)
    with torch.inference_mode():
        output = model.forward_features(inputs)
    return output["x_norm_clstoken"].cpu(), output["x_norm_patchtokens"].cpu()


def batches(array, batch_size: int):
    for start in range(0, array.shape[0], batch_size):
        yield [np.asarray(frame) for frame in array[start : start + batch_size]]


def extract_all(model, transform, array, batch_size, device):
    cls_parts, patch_parts = [], []
    for frames in batches(array, batch_size):
        cls, patch = extract(model, transform, frames, device)
        cls_parts.append(cls)
        patch_parts.append(patch)
    return torch.cat(cls_parts), torch.cat(patch_parts)


def center_crop_visual(frame_rgb: np.ndarray, resize_size: int, crop_size: int) -> np.ndarray:
    height, width = frame_rgb.shape[:2]
    scale = resize_size / min(height, width)
    resized = cv2.resize(
        frame_rgb,
        (int(round(width * scale)), int(round(height * scale))),
        interpolation=cv2.INTER_CUBIC,
    )
    x0 = (resized.shape[1] - crop_size) // 2
    y0 = (resized.shape[0] - crop_size) // 2
    return resized[y0 : y0 + crop_size, x0 : x0 + crop_size]


def save_pca_panel(
    *,
    frame_rgb: np.ndarray,
    patch_by_key: dict[str, np.ndarray],
    patch_grid: tuple[int, int],
    output: Path,
    title: str,
    resize_size: int,
    crop_size: int,
) -> None:
    """One shared 3D PCA basis across every key, so colours are comparable in-panel."""
    keys = list(patch_by_key)
    stacked = [patch_by_key[key].astype(np.float32, copy=False) for key in keys]
    token_count = stacked[0].shape[0]
    merged = torch.from_numpy(np.concatenate(stacked, axis=0))
    torch.manual_seed(0)
    _, _, basis = torch.pca_lowrank(merged, q=3, center=True, niter=4)
    projected = ((merged - merged.mean(dim=0, keepdim=True)) @ basis).numpy()
    low = np.percentile(projected, 1, axis=0)
    high = np.percentile(projected, 99, axis=0)
    normalized = np.clip((projected - low) / np.maximum(high - low, 1e-8), 0.0, 1.0)

    maps = []
    for index in range(len(keys)):
        start = index * token_count
        patch_rgb = normalized[start : start + token_count].reshape(*patch_grid, 3)
        patch_rgb = (patch_rgb * 255.0).round().astype(np.uint8)
        maps.append(cv2.resize(patch_rgb, (crop_size, crop_size), interpolation=cv2.INTER_NEAREST))

    panels = [center_crop_visual(frame_rgb, resize_size, crop_size), *maps]
    labels = ["input crop", *keys]
    canvas = cv2.cvtColor(np.concatenate(panels, axis=1), cv2.COLOR_RGB2BGR)
    for index, label in enumerate(labels):
        x0 = index * crop_size
        cv2.rectangle(canvas, (x0, 0), (x0 + crop_size, 30), (0, 0, 0), -1)
        cv2.putText(canvas, label, (x0 + 6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (0, canvas.shape[0] - 26), (canvas.shape[1], canvas.shape[0]), (0, 0, 0), -1)
    cv2.putText(canvas, title, (8, canvas.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), canvas)


def save_worst_panel(reference, candidate, output: Path, title: str) -> None:
    abs_diff = np.abs(candidate.astype(np.int16) - reference.astype(np.int16)).astype(np.uint8)
    amplified = np.clip(abs_diff.astype(np.uint16) * 4, 0, 255).astype(np.uint8)
    canvas = cv2.cvtColor(np.concatenate([reference, candidate, amplified], axis=1), cv2.COLOR_RGB2BGR)
    width = reference.shape[1]
    for index, label in enumerate(["original", "decoded", "|diff| x4"]):
        cv2.rectangle(canvas, (index * width, 0), ((index + 1) * width, 38), (0, 0, 0), -1)
        cv2.putText(canvas, label, (index * width + 10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.66,
                    (255, 255, 255), 2, cv2.LINE_AA)
    cv2.rectangle(canvas, (0, canvas.shape[0] - 32), (canvas.shape[1], canvas.shape[0]), (0, 0, 0), -1)
    cv2.putText(canvas, title, (10, canvas.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                (255, 255, 255), 1, cv2.LINE_AA)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), canvas)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resize_size = int(FEATURE_CFG["resize_size"])
    crop_size = int(FEATURE_CFG["crop_size"])
    pca_index = int(FEATURE_CFG["pca_frame_index"])
    transform = make_transform(resize_size, crop_size)
    results = ROOT / "results"
    figures = ROOT / "figures"

    rows: list[dict] = []
    embeddings: dict[str, np.ndarray] = {}
    summary = {
        "aligned_with": "test_lerobot/scripts/dinov3_feature_eval.py",
        "preprocessing": {
            "resize_shorter_side": resize_size,
            "center_crop": crop_size,
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
        "cameras": CAMERAS,
        "variants": list(VARIANTS),
        "models": {},
    }

    references = {camera: load_memmap("raw", camera) for camera in CAMERAS}
    frames_total = references[CAMERAS[0]].shape[0]
    print(f"device={device} frames={frames_total} cameras={CAMERAS} variants={list(VARIANTS)}")

    for model_cfg in FEATURE_CFG["models"]:
        model_id = model_cfg["id"]
        started = time.perf_counter()
        model = load_model(model_cfg, device)
        batch_size = int(model_cfg["batch_size"])
        patch_size = int(model_cfg["patch_size"])
        grid = crop_size // patch_size
        model_summary: dict = {
            "display_name": model_cfg["display_name"],
            "family": model_cfg["family"],
            "checkpoint": model_cfg["checkpoint"],
            "patch_size": patch_size,
            "patch_grid": [grid, grid],
            "tokens": grid * grid,
            "batch_size": batch_size,
            "per_camera": {},
        }
        print(f"\n=== {model_cfg['display_name']} ({grid}x{grid} tokens)")

        for camera in CAMERAS:
            reference_cls, reference_patch = extract_all(
                model, transform, references[camera], batch_size, device
            )
            embeddings[f"{model_id}__{camera}__original_cls"] = reference_cls.numpy().astype(np.float16)
            embeddings[f"{model_id}__{camera}__original_mean_patch"] = (
                reference_patch.mean(dim=1).numpy().astype(np.float16)
            )
            pca_patches = {"original": reference_patch[pca_index].numpy()}
            variant_summary: dict = {}

            for variant, kind in VARIANTS.items():
                candidate = load_memmap(kind, camera)
                if candidate.shape != references[camera].shape:
                    raise RuntimeError(f"{kind}/{camera}: shape {candidate.shape} != reference")
                variant_rows: list[dict] = []
                cls_parts, mean_patch_parts = [], []
                index = 0
                for frames in batches(candidate, batch_size):
                    cls, patch = extract(model, transform, frames, device)
                    count = cls.shape[0]
                    ref_cls = reference_cls[index : index + count]
                    ref_patch = reference_patch[index : index + count]

                    cls_cos = cosine(ref_cls, cls)
                    cls_rel = relative_l2(ref_cls, cls)
                    ref_mean = ref_patch.mean(dim=1)
                    cand_mean = patch.mean(dim=1)
                    pooled_cos = cosine(ref_mean, cand_mean)
                    pooled_rel = relative_l2(ref_mean, cand_mean)
                    token_cos = cosine(ref_patch, patch)
                    token_rel = relative_l2(ref_patch, patch)
                    cls_parts.append(cls.numpy())
                    mean_patch_parts.append(cand_mean.numpy())
                    if index <= pca_index < index + count:
                        pca_patches[variant] = patch[pca_index - index].numpy()

                    for i in range(count):
                        token_cos_i = token_cos[i].numpy()
                        token_rel_i = token_rel[i].numpy()
                        bounded = float(np.clip(cls_cos[i].item(), -1.0, 1.0))
                        variant_rows.append(
                            {
                                "model_id": model_id,
                                "model_name": model_cfg["display_name"],
                                "camera": camera,
                                "variant": variant,
                                "frame_index": index + i,
                                "cls_cosine": bounded,
                                "cls_angle_degrees": math.degrees(math.acos(bounded)),
                                "cls_relative_l2": float(cls_rel[i]),
                                "mean_patch_cosine": float(pooled_cos[i]),
                                "mean_patch_relative_l2": float(pooled_rel[i]),
                                "patch_token_cosine_mean": float(token_cos_i.mean()),
                                "patch_token_cosine_p05": float(np.percentile(token_cos_i, 5)),
                                "patch_token_cosine_min": float(token_cos_i.min()),
                                "patch_token_relative_l2_mean": float(token_rel_i.mean()),
                                "patch_token_relative_l2_p95": float(np.percentile(token_rel_i, 95)),
                                "patch_token_relative_l2_max": float(token_rel_i.max()),
                            }
                        )
                    index += count
                if index != frames_total:
                    raise RuntimeError(f"{variant}/{camera}: evaluated {index} of {frames_total}")

                rows.extend(variant_rows)
                embeddings[f"{model_id}__{camera}__{variant}_cls"] = np.concatenate(
                    cls_parts
                ).astype(np.float16)
                embeddings[f"{model_id}__{camera}__{variant}_mean_patch"] = np.concatenate(
                    mean_patch_parts
                ).astype(np.float16)
                variant_summary[variant] = {
                    key: summarize([row[key] for row in variant_rows]) for key in METRIC_KEYS
                }
                worst = min(variant_rows, key=lambda row: row["cls_cosine"])
                variant_summary[variant]["worst_cls_frame"] = worst["frame_index"]
                if camera == "top":
                    save_worst_panel(
                        np.asarray(references[camera][worst["frame_index"]]),
                        np.asarray(candidate[worst["frame_index"]]),
                        figures / "feature_worst" / f"{model_id}_{variant}_top.png",
                        f"{model_cfg['display_name']}, {variant}, camera top: "
                        f"frame {worst['frame_index']}, CLS cosine={worst['cls_cosine']:.6f}",
                    )
                print(
                    f"  {camera:8s} {variant:12s} CLS {variant_summary[variant]['cls_cosine']['mean']:.6f} "
                    f"patch {variant_summary[variant]['patch_token_cosine_mean']['mean']:.6f}",
                    flush=True,
                )

            model_summary["per_camera"][camera] = variant_summary
            save_pca_panel(
                frame_rgb=np.asarray(references[camera][pca_index]),
                patch_by_key=pca_patches,
                patch_grid=(grid, grid),
                output=figures / "feature_maps" / model_id / f"{camera}_frame_{pca_index:04d}_pca.png",
                title=f"{model_cfg['display_name']}, camera {camera}, frame {pca_index}; "
                      "shared PCA basis and colour scaling",
                resize_size=resize_size,
                crop_size=crop_size,
            )
            del reference_cls, reference_patch
            gc.collect()

        model_summary["elapsed_s"] = time.perf_counter() - started
        summary["models"][model_id] = model_summary
        del model
        gc.collect()
        torch.cuda.empty_cache()
        print(f"=== {model_id} done in {model_summary['elapsed_s']:.1f}s")

    fieldnames = ["model_id", "model_name", "camera", "variant", "frame_index", *METRIC_KEYS]
    with (results / "per_frame_model_feature_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    np.savez_compressed(results / "model_global_embeddings.npz", **embeddings)
    write_json(results / "model_feature_summary.json", summary)
    print(f"\nwrote {len(rows)} rows")


if __name__ == "__main__":
    main()
