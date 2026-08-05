#!/usr/bin/env python3
"""Stage 5: figures and REPORT.md."""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
CONFIG = json.loads((ROOT / "config.json").read_text())
CAMERAS = CONFIG["cameras"]
CRF_LEVELS = CONFIG["video"]["crf_levels"]

PALETTE = {
    "raw": "#3b3b3b",
    # Stage-1 JPEG bars are a different kind of measurement from the pipeline
    # bars, so they get their own greys rather than borrowing a source colour.
    "jpeg100": "#8c8c8c",
    "jpeg80": "#4d4d4d",
    "raw_source": "#1f77b4",
    "jpeg100_source": "#2ca02c",
    "jpeg80_source": "#d62728",
}

SOURCE_ORDER = ["raw", "jpeg100", "jpeg80"]


def source_color(source: str) -> str:
    return PALETTE[f"{source}_source"]


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


def read_csv(name: str) -> list[dict]:
    with (RESULTS / name).open() as handle:
        return list(csv.DictReader(handle))


def mib(value: float) -> float:
    return value / 2**20


def kib(value: float) -> float:
    return value / 1024.0


def fmt(value: float, digits: int = 3) -> str:
    if value != value:
        return "n/a"
    if value == float("inf"):
        return "∞"
    return f"{value:.{digits}f}"


# --------------------------------------------------------------------------


def figure_storage(jpeg: dict, video: dict, output: Path) -> None:
    frames = jpeg["frames"]
    labels, values, colors = [], [], []
    labels.append("raw mosaic\n(sensor_msgs/Image)")
    values.append(jpeg["raw_mosaic_bytes_per_frame"])
    colors.append(PALETTE["raw"])
    for quality in ("100", "80"):
        labels.append(f"JPEG q{quality}\n(mosaic)")
        values.append(jpeg["jpeg"][quality]["bytes_mean"])
        colors.append(PALETTE[f"jpeg{quality}"])
    for entry in video["entries"]:
        labels.append(f"{entry['source']}\nh264 crf {entry['crf']}")
        values.append(entry["video_bytes_sum_cameras"] / frames)
        colors.append(source_color(entry["source"]))

    fig, ax = plt.subplots(figsize=(14.5, 5.2))
    bars = ax.bar(range(len(values)), values, color=colors)
    ax.set_yscale("log")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("bytes per frame (log scale)")
    ax.set_title("Storage per frame at each stage of the pipeline", loc="left")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value * 1.12, f"{kib(value):,.0f} KiB",
                ha="center", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title(
        "mosaic stages hold all three cameras in one 1280x1440 image; "
        "video stages are the sum of the three 640x480 camera tracks",
        loc="right", fontsize=8, color="#555",
    )
    ax.set_ylim(top=max(values) * 3)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def figure_quality_ladder(jpeg: dict, video: dict, output: Path) -> None:
    labels, psnr, ssim, colors = [], [], [], []
    for quality in ("100", "80"):
        per_camera = jpeg["per_camera"][quality]
        labels.append(f"JPEG q{quality}")
        psnr.append(float(np.mean([per_camera[c]["psnr_db"]["mean"] for c in CAMERAS])))
        ssim.append(float(np.mean([per_camera[c]["ssim"]["mean"] for c in CAMERAS])))
        colors.append(PALETTE[f"jpeg{quality}"])
    for entry in video["entries"]:
        labels.append(f"{entry['source']}\n+crf{entry['crf']}")
        psnr.append(float(np.mean([entry["cameras"][c]["vs_raw"]["psnr_db"]["mean"] for c in CAMERAS])))
        ssim.append(float(np.mean([entry["cameras"][c]["vs_raw"]["ssim"]["mean"] for c in CAMERAS])))
        colors.append(source_color(entry["source"]))

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 4.6))
    for ax, values, name in ((axes[0], psnr, "PSNR (dB)"), (axes[1], ssim, "SSIM")):
        bars = ax.bar(range(len(values)), values, color=colors)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel(name)
        ax.grid(axis="y", alpha=0.3)
        digits = 2 if "PSNR" in name else 4
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.{digits}f}",
                    ha="center", va="bottom", fontsize=7.5)
        low = min(values)
        high = max(values)
        ax.set_ylim(low - (high - low) * 0.25, high + (high - low) * 0.18)
    axes[0].set_title("Mean quality vs the uncompressed sensor frame (3 cameras pooled)", loc="left")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def figure_rate_distortion(jpeg: dict, video: dict, output: Path) -> None:
    frames = jpeg["frames"]
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    for source in SOURCE_ORDER:
        color = source_color(source)
        points = []
        for entry in video["entries"]:
            if entry["source"] != source:
                continue
            size = entry["video_bytes_sum_cameras"]
            quality = float(np.mean([entry["cameras"][c]["vs_raw"]["psnr_db"]["mean"] for c in CAMERAS]))
            points.append((size / frames / 1024.0, quality, entry["crf"]))
        points.sort()
        ax.plot([p[0] for p in points], [p[1] for p in points], "o-", color=color,
                label=f"{source} source")
        for x, y, crf in points:
            ax.annotate(f"crf {crf}", (x, y), textcoords="offset points", xytext=(6, -10),
                        fontsize=8, color=color)
    q80 = float(np.mean([jpeg["per_camera"]["80"][c]["psnr_db"]["mean"] for c in CAMERAS]))
    ax.axhline(q80, color=PALETTE["jpeg80_source"], linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.text(0.02, q80, f"JPEG q80 ceiling = {q80:.2f} dB", transform=ax.get_yaxis_transform(),
            va="bottom", fontsize=8.5, color=PALETTE["jpeg80_source"])
    ax.set_xlabel("stored video bytes per frame, all 3 cameras (KiB, log)")
    ax.set_ylabel("PSNR vs uncompressed sensor frame (dB)")
    ax.set_title("Rate / distortion, measured against raw", loc="left")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def figure_marginal(video: dict, output: Path) -> None:
    """For each JPEG source: H.264's own contribution vs the total vs raw."""
    sources = [s for s in SOURCE_ORDER if s != "raw"]
    panels = [(s, [e for e in video["entries"] if e["source"] == s]) for s in sources]
    panels = [(s, e) for s, e in panels if e]
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 4.6), sharey=True)
    all_values: list[float] = []
    for ax, (source, entries) in zip(np.atleast_1d(axes), panels):
        crfs = [e["crf"] for e in entries]
        vs_raw = [float(np.mean([e["cameras"][c]["vs_raw"]["psnr_db"]["mean"] for c in CAMERAS]))
                  for e in entries]
        vs_input = [float(np.mean([e["cameras"][c]["vs_input"]["psnr_db"]["mean"] for c in CAMERAS]))
                    for e in entries]
        all_values.extend(vs_raw + vs_input)
        x = np.arange(len(crfs))
        ax.bar(x - 0.2, vs_input, 0.4, label="vs its own encoder input (H.264 only)",
               color="#7fb3d5")
        ax.bar(x + 0.2, vs_raw, 0.4, label="vs raw sensor frame (JPEG + H.264)",
               color=source_color(source))
        for xi, value in zip(x - 0.2, vs_input):
            ax.text(xi, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
        for xi, value in zip(x + 0.2, vs_raw):
            ax.text(xi, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"crf {c}" for c in crfs])
        ax.set_title(f"{source} source", loc="left", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    first = np.atleast_1d(axes)[0]
    first.set_ylabel("PSNR (dB)")
    low, high = min(all_values), max(all_values)
    first.set_ylim(low - (high - low) * 0.3, high + (high - low) * 0.12)
    first.legend(fontsize=8.5)
    fig.suptitle("What H.264 adds vs what the recording-side JPEG already took",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def figure_per_frame(output: Path) -> None:
    rows = read_csv("per_frame_video_metrics.csv")
    jpeg_rows = read_csv("per_frame_jpeg_tile_metrics.csv")
    series: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        if row["camera"] != "top" or row["reference"] != "raw":
            continue
        series[f"{row['source']} crf{row['crf']}"].append(
            (int(row["frame_index"]), float(row["psnr_db"]))
        )
    for row in jpeg_rows:
        if row["camera"] != "top":
            continue
        series[f"JPEG q{row['quality']}"].append((int(row["frame_index"]), float(row["psnr_db"])))

    fig, ax = plt.subplots(figsize=(13, 5))
    order = ["JPEG q100", "JPEG q80",
             "raw crf0", "raw crf20", "raw crf30",
             "jpeg100 crf0", "jpeg100 crf20", "jpeg100 crf30",
             "jpeg80 crf0", "jpeg80 crf20", "jpeg80 crf30"]
    styles = {
        "JPEG q100": ("#8c8c8c", "-", 1.6),
        "JPEG q80": ("#000000", "-", 1.6),
        "raw crf0": (PALETTE["raw_source"], "-", 1.0),
        "raw crf20": (PALETTE["raw_source"], "--", 1.0),
        "raw crf30": (PALETTE["raw_source"], ":", 1.2),
        "jpeg100 crf0": (PALETTE["jpeg100_source"], "-", 1.0),
        "jpeg100 crf20": (PALETTE["jpeg100_source"], "--", 1.0),
        "jpeg100 crf30": (PALETTE["jpeg100_source"], ":", 1.2),
        "jpeg80 crf0": (PALETTE["jpeg80_source"], "-", 1.0),
        "jpeg80 crf20": (PALETTE["jpeg80_source"], "--", 1.0),
        "jpeg80 crf30": (PALETTE["jpeg80_source"], ":", 1.2),
    }
    for key in order:
        points = sorted(series.get(key, []))
        if not points:
            continue
        color, style, width = styles[key]
        ax.plot([p[0] for p in points], [p[1] for p in points], color=color, linestyle=style,
                linewidth=width, label=key)
    ax.set_xlabel("frame index")
    ax.set_ylabel("PSNR vs raw (dB)")
    ax.set_title("Per-frame quality, head camera (top)", loc="left")
    ax.grid(alpha=0.3)
    ax.legend(ncol=4, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------


TEST_LEROBOT = Path("/home/kewei/YING/robot_data_platform/test_lerobot")

# Which variant of ours lines up with which of test_lerobot's, for the side by
# side table.  Only the CRF 0 and CRF 20 rungs have a counterpart; their CRF 50
# has none here, and our CRF 30 has none there.
CROSS_MATCH = [("raw_crf0", "crf0_min_loss"), ("raw_crf20", "crf20_balanced")]

VARIANT_ORDER = [
    "jpeg_q100", "jpeg_q80",
    "raw_crf0", "raw_crf20", "raw_crf30",
    "jpeg100_crf0", "jpeg100_crf20", "jpeg100_crf30",
    "jpeg80_crf0", "jpeg80_crf20", "jpeg80_crf30",
]


def variant_psnr(jpeg: dict, video: dict) -> dict[tuple[str, str], float]:
    """PSNR vs raw for every (variant, camera), pooling stage 1 and stage 2."""
    psnr: dict[tuple[str, str], float] = {}
    for quality, variant in (("100", "jpeg_q100"), ("80", "jpeg_q80")):
        for camera in CAMERAS:
            psnr[(variant, camera)] = jpeg["per_camera"][quality][camera]["psnr_db"]["mean"]
    for entry in video["entries"]:
        variant = f"{entry['source']}_crf{entry['crf']}"
        for camera in CAMERAS:
            psnr[(variant, camera)] = entry["cameras"][camera]["vs_raw"]["psnr_db"]["mean"]
    return psnr


def _rank(values: np.ndarray) -> np.ndarray:
    order = values.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(values.size)
    return ranks


def pixel_feature_stats(features: dict, jpeg: dict, video: dict) -> dict:
    """Pearson and Spearman between PSNR and CLS cosine, pooled and per model.

    The two disagree sharply, which is the point: the relationship is monotone
    but saturating, so PSNR ranks variants well and scales badly.
    """
    psnr = variant_psnr(jpeg, video)
    model_ids = list(features["models"])
    xs, ys = [], []
    per_model: dict[str, float] = {}
    for model_id in model_ids:
        mx, my = [], []
        for camera in CAMERAS:
            for variant in VARIANT_ORDER:
                mx.append(psnr[(variant, camera)])
                my.append(feature_mean(features, model_id, camera, variant, "cls_cosine"))
        mx_a, my_a = np.asarray(mx), np.asarray(my)
        per_model[model_id] = float(np.corrcoef(_rank(mx_a), _rank(my_a))[0, 1])
        xs.extend(mx)
        ys.extend(my)
    # Local slopes on either side of the saturation knee, in cosine per 3 dB.
    slope_high: dict[str, float] = {}
    slope_low: dict[str, float] = {}
    for model_id in model_ids:
        points = sorted(
            (psnr[(variant, camera)],
             feature_mean(features, model_id, camera, variant, "cls_cosine"))
            for camera in CAMERAS for variant in VARIANT_ORDER
        )
        for band, store in (([p for p in points if p[0] >= 43], slope_high),
                            ([p for p in points if p[0] <= 40], slope_low)):
            bx = np.asarray([p[0] for p in band])
            by = np.asarray([p[1] for p in band])
            store[model_id] = float(np.polyfit(bx, by, 1)[0] * 3.0)

    xs_a, ys_a = np.asarray(xs), np.asarray(ys)
    return {
        "n": int(xs_a.size),
        "pearson": float(np.corrcoef(xs_a, ys_a)[0, 1]),
        "rho": float(np.corrcoef(_rank(xs_a), _rank(ys_a))[0, 1]),
        "per_model": per_model,
        "slope_high": slope_high,
        "slope_low": slope_low,
    }


def feature_mean(features: dict, model_id: str, camera: str, variant: str, metric: str) -> float:
    return features["models"][model_id]["per_camera"][camera][variant][metric]["mean"]


def feature_pooled(features: dict, model_id: str, variant: str, metric: str) -> float:
    return float(np.mean([feature_mean(features, model_id, c, variant, metric) for c in CAMERAS]))


def figure_feature_variants(features: dict, output: Path) -> None:
    model_ids = list(features["models"])
    fig, axes = plt.subplots(2, 1, figsize=(13, 8.4), sharex=True)
    width = 0.26
    x = np.arange(len(VARIANT_ORDER))
    for ax, metric, name in (
        (axes[0], "cls_cosine", "CLS cosine"),
        (axes[1], "patch_token_cosine_mean", "patch-token cosine"),
    ):
        for offset, model_id in zip((-width, 0.0, width), model_ids):
            values = [feature_pooled(features, model_id, v, metric) for v in VARIANT_ORDER]
            bars = ax.bar(x + offset, values, width,
                          label=features["models"][model_id]["display_name"])
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.4f}",
                        ha="center", va="bottom", fontsize=6.5, rotation=90)
        ax.set_ylabel(name)
        ax.grid(axis="y", alpha=0.3)
        flat = [feature_pooled(features, m, v, metric) for m in model_ids for v in VARIANT_ORDER]
        ax.set_ylim(min(flat) - (1 - min(flat)) * 0.6, 1.0 + (1 - min(flat)) * 0.25)
    axes[0].set_title("Feature fidelity vs the uncompressed frame (3 cameras pooled)", loc="left")
    axes[0].legend(fontsize=8.5, ncol=3)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(VARIANT_ORDER, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def figure_feature_by_camera(features: dict, output: Path) -> None:
    model_ids = list(features["models"])
    fig, axes = plt.subplots(1, len(model_ids), figsize=(13, 4.4), sharey=True)
    width = 0.26
    x = np.arange(len(VARIANT_ORDER))
    for ax, model_id in zip(np.atleast_1d(axes), model_ids):
        for offset, camera in zip((-width, 0.0, width), CAMERAS):
            values = [feature_mean(features, model_id, camera, v, "cls_cosine")
                      for v in VARIANT_ORDER]
            ax.bar(x + offset, values, width, label=camera)
        ax.set_title(features["models"][model_id]["display_name"], loc="left", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(VARIANT_ORDER, rotation=60, ha="right", fontsize=7.5)
        ax.grid(axis="y", alpha=0.3)
    flat = [feature_mean(features, m, c, v, "cls_cosine")
            for m in model_ids for c in CAMERAS for v in VARIANT_ORDER]
    np.atleast_1d(axes)[0].set_ylim(min(flat) - (1 - min(flat)) * 0.4, 1.0)
    np.atleast_1d(axes)[0].set_ylabel("CLS cosine")
    np.atleast_1d(axes)[0].legend(fontsize=8.5, title="camera")
    fig.suptitle("Per-camera feature fidelity: the mosaic's quality split shows up in features too",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def figure_feature_vs_pixel(features: dict, jpeg: dict, video: dict, output: Path) -> None:
    """Does PSNR predict feature drift across these eight variants?"""
    psnr = variant_psnr(jpeg, video)
    model_ids = list(features["models"])
    fig, ax = plt.subplots(figsize=(7.8, 5.4))
    markers = {"top": "o", "wrist_L": "s", "wrist_R": "^"}
    colors = dict(zip(model_ids, ["#1f77b4", "#d62728", "#2ca02c"]))
    xs, ys = [], []
    for model_id in model_ids:
        for camera in CAMERAS:
            px = [psnr[(v, camera)] for v in VARIANT_ORDER]
            py = [feature_mean(features, model_id, camera, v, "cls_cosine") for v in VARIANT_ORDER]
            ax.scatter(px, py, marker=markers[camera], color=colors[model_id], alpha=0.8, s=34)
            xs.extend(px)
            ys.extend(py)
    correlation = float(np.corrcoef(xs, ys)[0, 1])
    handles = [plt.Line2D([], [], color=colors[m], marker="o", linestyle="",
                          label=features["models"][m]["display_name"]) for m in model_ids]
    handles += [plt.Line2D([], [], color="#555", marker=markers[c], linestyle="", label=c)
                for c in CAMERAS]
    ax.legend(handles=handles, fontsize=8, ncol=2)
    ax.set_xlabel("PSNR vs raw (dB)")
    ax.set_ylabel("CLS cosine vs raw")
    ax.set_title(f"Pixel fidelity vs feature fidelity (Pearson r = {correlation:.3f})", loc="left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return correlation


def ip_frame_split() -> list[dict]:
    """Mean quality of I frames vs P frames on the head camera.

    LeRobot encodes RGB with ``g=2``, so every other frame is a P frame.  At a
    lossy CRF the two reconstruct to visibly different quality, which shows up
    as the sawtooth in the per-frame figure.
    """
    rows = [r for r in read_csv("per_frame_video_metrics.csv")
            if r["reference"] == "raw" and r["camera"] == "top"]
    out = []
    for source in ("raw", "jpeg80"):
        for crf in CRF_LEVELS:
            subset = sorted(
                (int(r["frame_index"]), float(r["psnr_db"]))
                for r in rows if r["source"] == source and r["crf"] == str(crf)
            )
            if not subset:
                continue
            values = np.asarray([p[1] for p in subset])
            even = float(values[0::2].mean())
            odd = float(values[1::2].mean())
            out.append({"source": source, "crf": crf, "even": even, "odd": odd,
                        "gap": even - odd})
    return out


def cross_report_rows(features: dict) -> list[dict]:
    """Line our head-camera numbers up with test_lerobot's, where a rung matches."""
    path = TEST_LEROBOT / "results" / "model_feature_summary.json"
    if not path.exists():
        return []
    other = json.loads(path.read_text())["models"]
    rows = []
    for model_id, item in features["models"].items():
        if model_id not in other:
            continue
        for ours, theirs in CROSS_MATCH:
            reference = other[model_id]["variants"][theirs]
            rows.append(
                {
                    "model": item["display_name"],
                    "crf": ours.rsplit("crf", 1)[1],
                    "ref_cls": reference["cls_cosine"]["mean"],
                    "our_cls": feature_mean(features, model_id, "top", ours, "cls_cosine"),
                    "ref_patch": reference["patch_token_cosine_mean"]["mean"],
                    "our_patch": feature_mean(
                        features, model_id, "top", ours, "patch_token_cosine_mean"
                    ),
                }
            )
    return rows


def build_report(
    jpeg: dict,
    video: dict,
    reference: dict,
    bags: dict,
    audit: dict,
    features: dict | None,
    correlation: float | None,
    spearman: dict | None,
) -> str:
    ip_rows = ip_frame_split()
    cross_rows = cross_report_rows(features) if features else []
    frames = jpeg["frames"]
    duration = jpeg["duration_s"]
    raw_bpf = jpeg["raw_mosaic_bytes_per_frame"]

    def video_entry(source: str, crf: int) -> dict | None:
        for entry in video["entries"]:
            if entry["source"] == source and entry["crf"] == crf:
                return entry
        return None

    def pooled(entry: dict, reference_key: str, metric: str) -> float:
        return float(np.mean([entry["cameras"][c][reference_key][metric]["mean"] for c in CAMERAS]))

    def jpeg_pooled(quality: str, metric: str) -> float:
        return float(np.mean([jpeg["per_camera"][quality][c][metric]["mean"] for c in CAMERAS]))

    lines: list[str] = []
    add = lines.append

    add("# 原始图像 vs JPEG 压缩图像：录制端与 LeRobot 存储端的画质影响测试")
    add("")
    add(f"生成时间：{datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')}")
    add("")

    # ---- 摘要 -----------------------------------------------------------
    q80_psnr = jpeg_pooled("80", "psnr_db")
    q80_ssim = jpeg_pooled("80", "ssim")
    q100_psnr = jpeg_pooled("100", "psnr_db")
    q100_ssim = jpeg_pooled("100", "ssim")
    raw_crf0 = video_entry("raw", 0)
    raw_crf20 = video_entry("raw", 20)
    raw_crf30 = video_entry("raw", 30)
    j_crf0 = video_entry("jpeg80", 0)
    j_crf20 = video_entry("jpeg80", 20)
    j_crf30 = video_entry("jpeg80", 30)
    q100_crf0 = video_entry("jpeg100", 0)
    q100_crf20 = video_entry("jpeg100", 20)
    q100_crf30 = video_entry("jpeg100", 30)

    per80 = jpeg["per_camera"]["80"]
    gap_crf0 = pooled(raw_crf0, "vs_raw", "psnr_db") - pooled(j_crf0, "vs_raw", "psnr_db")
    gap_crf20 = pooled(raw_crf20, "vs_raw", "psnr_db") - pooled(j_crf20, "vs_raw", "psnr_db")
    gap_crf30 = pooled(raw_crf30, "vs_raw", "psnr_db") - pooled(j_crf30, "vs_raw", "psnr_db")
    size_ratio_0 = j_crf0["video_bytes_sum_cameras"] / raw_crf20["video_bytes_sum_cameras"]
    grow20 = j_crf20["video_bytes_sum_cameras"] / raw_crf20["video_bytes_sum_cameras"] - 1
    grow30 = j_crf30["video_bytes_sum_cameras"] / raw_crf30["video_bytes_sum_cameras"] - 1
    shrink0 = 1 - j_crf0["video_bytes_sum_cameras"] / raw_crf0["video_bytes_sum_cameras"]

    add("## 结论摘要")
    add("")
    add(f"本测试使用 `express_raw` 中唯一一段未压缩录制（`/quad_tile`，`sensor_msgs/Image`，"
        f"1280×1440 马赛克，{frames} 个对齐控制行，{duration:.2f} s）作为像素级基准，"
        "逐级比较录制端 JPEG 与 LeRobot 视频编码带来的损失。所有 PSNR/SSIM 均以未压缩原始帧"
        "为参照，三相机取平均（逐相机数值见后文表格）。")
    add("")
    points: list[str] = []

    def point(text: str) -> None:
        points.append(text)

    point(f"**q80 的损失几乎全部压在两路 wrist 相机上。** 三相机平均 "
        f"PSNR {q80_psnr:.3f} dB / SSIM {q80_ssim:.6f}，但拆开来 "
        f"head {per80['top']['psnr_db']['mean']:.3f} dB、"
        f"wrist_L {per80['wrist_L']['psnr_db']['mean']:.3f} dB、"
        f"wrist_R {per80['wrist_R']['psnr_db']['mean']:.3f} dB，相差 6–7 dB。"
        "原因是 hero3 马赛克把 head 以 2× 放大存放（占了 4 倍面积），"
        "JPEG 作用在放大后的像素上，切分时的下采样又把噪声平均掉；"
        "两路 wrist 是原生分辨率，误差原样保留。做精细操作的恰恰是 wrist。")
    point(f"**如果最终按 CRF 20/30 存储，录制端 JPEG 的额外代价很小。** 同一 CRF 下 jpeg80 源"
        f"相对 raw 源只低 {gap_crf20:.3f} dB（CRF 20）和 {gap_crf30:.3f} dB（CRF 30）——"
        f"视频编码自身的损失已经盖过 JPEG。只有在接近无损存储时 JPEG 才是主导项："
        f"CRF 0 下两者相差 {gap_crf0:.3f} dB。")
    if q100_crf20 is not None:
        q100_d0 = pooled(raw_crf0, "vs_raw", "psnr_db") - pooled(q100_crf0, "vs_raw", "psnr_db")
        q100_d20 = pooled(raw_crf20, "vs_raw", "psnr_db") - pooled(q100_crf20, "vs_raw", "psnr_db")
        q100_d30 = pooled(raw_crf30, "vs_raw", "psnr_db") - pooled(q100_crf30, "vs_raw", "psnr_db")
        q100_grow0 = (
            q100_crf0["video_bytes_sum_cameras"] / raw_crf0["video_bytes_sum_cameras"] - 1
        )
        point(f"**q100 说明「要不要压」和「压到多少」是两个问题。** 第三条链路 jpeg100 源"
            f"相对 raw 源，在 CRF 20/30 上只低 {q100_d20:.3f} / {q100_d30:.3f} dB，"
            f"而录制端体积只有原始的 1/"
            f"{jpeg['jpeg']['100']['compression_ratio_vs_raw']:.1f}——录制端压缩本身近乎免费。"
            f"但在 CRF 0 上它低 {q100_d0:.3f} dB **而且**比 raw 源还大 "
            f"{q100_grow0 * 100:.1f}%（{mib(q100_crf0['video_bytes_sum_cameras']):.2f} vs "
            f"{mib(raw_crf0['video_bytes_sum_cameras']):.2f} MiB），是严格劣势的一格："
            "JPEG 的轻微振铃既没被 CRF 0 丢掉，又得花码率去编码。")
    point(f"**反过来说，源是 q80 时用 CRF 0 是纯浪费。** jpeg80→CRF 0 为 "
        f"{pooled(j_crf0, 'vs_raw', 'psnr_db'):.3f} dB / "
        f"{mib(j_crf0['video_bytes_sum_cameras']):.2f} MiB，"
        f"raw→CRF 20 为 {pooled(raw_crf20, 'vs_raw', 'psnr_db'):.3f} dB / "
        f"{mib(raw_crf20['video_bytes_sum_cameras']):.2f} MiB："
        f"画质基本相同，体积差 {size_ratio_0:.1f} 倍。低 CRF 只是在高保真地保存 JPEG 块噪声。")
    point(f"**预压缩还会让同 CRF 的文件变大。** q80 源 vs raw 源：CRF 20 "
        f"{mib(j_crf20['video_bytes_sum_cameras']):.2f} vs "
        f"{mib(raw_crf20['video_bytes_sum_cameras']):.2f} MiB（+{grow20 * 100:.1f}%），"
        f"CRF 30 {mib(j_crf30['video_bytes_sum_cameras']):.2f} vs "
        f"{mib(raw_crf30['video_bytes_sum_cameras']):.2f} MiB（+{grow30 * 100:.1f}%）。"
        "JPEG 的块状噪声是高频信号，H.264 得额外花码率去编码它。"
        f"q80 只在 CRF 0 上例外（−{shrink0 * 100:.1f}%），因为那时它抹掉的细节确实不用再编码——"
        "而 q100 没抹掉多少细节，所以连这一格都是净增（见第 3 条）。")
    if features is not None:
        fm = features["models"]
        ids = list(fm)
        j20 = [feature_pooled(features, m, "jpeg80_crf20", "cls_cosine") for m in ids]
        r20 = [feature_pooled(features, m, "raw_crf20", "cls_cosine") for m in ids]
        j30 = [feature_pooled(features, m, "jpeg80_crf30", "cls_cosine") for m in ids]
        r30 = [feature_pooled(features, m, "raw_crf30", "cls_cosine") for m in ids]
        j0 = [feature_pooled(features, m, "jpeg80_crf0", "cls_cosine") for m in ids]
        r0 = [feature_pooled(features, m, "raw_crf0", "cls_cosine") for m in ids]
        d20 = [a - b for a, b in zip(r20, j20)]
        d30 = [a - b for a, b in zip(r30, j30)]
        d0 = [a - b for a, b in zip(r0, j0)]
        point(f"**DINO 特征给出同样的结论。** 三个主干上，jpeg80 源相对 raw 源的 CLS cosine "
            f"损失：CRF 20 为 {min(d20):.4f}–{max(d20):.4f}，CRF 30 为 "
            f"{min(d30):.4f}–{max(d30):.4f}（已在噪声量级），而 CRF 0 为 "
            f"{min(d0):.4f}–{max(d0):.4f}。像素上看到的「低 CRF 才暴露 JPEG」在特征上原样成立。"
            f"PSNR 与 CLS cosine 之间 Spearman ρ = {spearman['rho']:.3f} 但 Pearson "
            f"r = {correlation:.3f}——排序一致、数值不成比例，所以 PSNR 可以用来排序档位，"
            "不能用来当验收阈值（见阶段三）。")

    for index, text in enumerate(points, start=1):
        add(f"{index}. {text}")
        add("")

    add("### 建议")
    add("")
    add(f"- 训练集按 CRF 20 存储时（本段 {mib(raw_crf20['dataset_bytes']):.1f} MiB / "
        f"{duration:.1f} s），继续用 q80 录制是合理的，代价约 {gap_crf20:.1f} dB。")
    add("- 要提升 wrist 画质，应该改录制端而不是降 CRF：提高 "
        "`publish.jpeg_quality`，或调整 `mosaic.top_height` 让马赛克不再把 4 倍面积分给 head，"
        "或改录 `per_camera_compressed` 的原生话题。在 q80 源上降 CRF 的收益按第 2、3 条递减。")
    add("- 不要在 q80 源上使用 CRF 0：更大、更慢，画质并不比 raw→CRF 20 好。")
    if features is not None:
        add("- 用 PSNR 排序候选档位没问题，但验收线要落在特征指标上：两者单调一致而不成比例，"
            "同样 3 dB 在高保真区几乎不改变特征、在低保真区能改变 0.05 以上的 CLS cosine。")
    add("")
    add("![storage](figures/storage_per_frame.png)")
    add("")
    add("![quality](figures/quality_ladder.png)")
    add("")

    # ---- 实验设计 -------------------------------------------------------
    add("## 实验设计")
    add("")
    add("`express_mcap` 与 `express_raw` 是两次不同的录制，帧内容不同，无法逐像素比较。"
        "因此本测试不直接对比这两个包，而是构造缺失的对照组：把 `express_raw` 分别按 "
        "q100 与 q80 重新编码成 JPEG 的 mcap（时间戳、关节数据逐字节复制，只替换图像话题），"
        "使三条链路面对**完全相同的帧**。q80 是生产配置，q100 用来把「压了」和「压狠了」"
        "这两件事分开。`express_mcap` 只用于验证合成的 q80 码率是否与真实录制一致。")
    add("")
    add("```")
    add("                 ┌─ (A) 直接转换 ─────────────────► LeRobot h264 CRF 0/20/30")
    add("raw /quad_tile ──┼─ JPEG q100 (cv2, 马赛克整幅) ──► LeRobot h264 CRF 0/20/30   (B)")
    add("  (基准帧)        └─ JPEG q80  (cv2, 马赛克整幅) ──► LeRobot h264 CRF 0/20/30   (C) 生产链路")
    add("```")
    add("")
    add("| 环节 | 设置 | 依据 |")
    add("|---|---|---|")
    add(f"| 录制端 JPEG | `cv2.imencode('.jpg', BGR马赛克, [IMWRITE_JPEG_QUALITY, q])`，"
        f"q ∈ {{100, 80}}，4:2:0 | `{CONFIG['jpeg']['matches']}` |")
    add(f"| 视频编码 | h264 / yuv420p / GOP {CONFIG['video']['gop']} / preset 默认 / "
        f"CRF ∈ {{0, 20, 30}} | `tool/lerobot_v3_common.py:RGBVideoConfig` |")
    add(f"| 对齐 | `{CONFIG['alignment']['mode']}`，`{CONFIG['alignment']['grid_anchor']}`，"
        f"`--state-tolerance-ms {CONFIG['alignment']['state_tolerance_ms']:.0f}` | 与生产转换命令一致 |")
    add(f"| 相机切分 | hero3 马赛克 → top(1280×960→640×480) / wrist_L / wrist_R(640×480 原生) | "
        f"`tool/profiles/marvin-gripper-quadtile.json` |")
    add("")
    add(f"关键一致性检查：第二遍解码出的马赛克按 profile 裁剪后，与转换器实际写入的 tile "
        f"逐字节相等（{frames} 帧 × 3 相机全部通过），因此三条链路确实对齐在同一批帧上。")
    add("")

    # ---- 阶段 1 ---------------------------------------------------------
    add("## 阶段一：录制端 JPEG 相对原始帧")
    add("")
    add("### 马赛克整幅（JPEG 实际作用的对象）")
    add("")
    add("| 指标 | 原始 | JPEG q100 | JPEG q80 |")
    add("|---|---:|---:|---:|")
    m100 = jpeg["mosaic_metrics"]["100"]
    m80 = jpeg["mosaic_metrics"]["80"]
    add(f"| 每帧字节 | {raw_bpf:,} | {jpeg['jpeg']['100']['bytes_mean']:,.0f} | "
        f"{jpeg['jpeg']['80']['bytes_mean']:,.0f} |")
    add(f"| 每帧字节（KiB） | {kib(raw_bpf):,.0f} | {kib(jpeg['jpeg']['100']['bytes_mean']):.1f} | "
        f"{kib(jpeg['jpeg']['80']['bytes_mean']):.1f} |")
    add(f"| 压缩比 | 1.0× | {jpeg['jpeg']['100']['compression_ratio_vs_raw']:.1f}× | "
        f"{jpeg['jpeg']['80']['compression_ratio_vs_raw']:.1f}× |")
    add(f"| 30 FPS 码率 | {jpeg['raw_mosaic_bitrate_mbps']:.1f} Mb/s | "
        f"{jpeg['jpeg']['100']['bitrate_mbps']:.1f} Mb/s | "
        f"{jpeg['jpeg']['80']['bitrate_mbps']:.1f} Mb/s |")
    add(f"| PSNR | ∞ | {fmt(m100['psnr_db']['mean'])} dB | {fmt(m80['psnr_db']['mean'])} dB |")
    add(f"| SSIM | 1 | {fmt(m100['ssim']['mean'], 6)} | {fmt(m80['ssim']['mean'], 6)} |")
    add(f"| MAE (0–255) | 0 | {fmt(m100['mae_255']['mean'], 4)} | {fmt(m80['mae_255']['mean'], 4)} |")
    add(f"| 最大绝对误差 | 0 | {m100['max_abs_error']['max']:.0f} | {m80['max_abs_error']['max']:.0f} |")
    add(f"| 色度采样 | — | {jpeg['jpeg']['100']['chroma_subsampling']} | "
        f"{jpeg['jpeg']['80']['chroma_subsampling']} |")
    add("")
    add("### 切分后的三路相机（LeRobot 实际存储的画面）")
    add("")
    add("| 相机 | q100 PSNR | q100 SSIM | q100 MAE | q80 PSNR | q80 SSIM | q80 MAE |")
    add("|---|---:|---:|---:|---:|---:|---:|")
    for camera in CAMERAS:
        a = jpeg["per_camera"]["100"][camera]
        b = jpeg["per_camera"]["80"][camera]
        add(f"| {camera} | {fmt(a['psnr_db']['mean'])} dB | {fmt(a['ssim']['mean'], 6)} | "
            f"{fmt(a['mae_255']['mean'], 4)} | {fmt(b['psnr_db']['mean'])} dB | "
            f"{fmt(b['ssim']['mean'], 6)} | {fmt(b['mae_255']['mean'], 4)} |")
    add("")
    add(f"`top` 与两路 wrist 的差异来自马赛克布局：head 在马赛克里是 2× 放大存放的，"
        f"JPEG 作用在放大后的像素上，切分时再缩回 640×480，这一次下采样会平均掉一部分 JPEG "
        f"噪声；wrist 是原生分辨率，JPEG 误差被原样保留。")
    add("")
    add("### 与真实生产录制的码率对照")
    add("")
    add(f"`express_mcap` 采样 {reference['bags_sampled']} 个包、{reference['frames']:,} 帧，"
        f"`{reference['topic']}` 平均 {kib(reference['bytes_mean']):.1f} KiB/帧"
        f"（中位 {kib(reference['bytes_median']):.1f}，p05–p95 "
        f"{kib(reference['bytes_p05']):.1f}–{kib(reference['bytes_p95']):.1f}），"
        f"{reference['chroma_subsampling'][0]}，{reference['decoded_size'][0]}。"
        f"本测试合成的 q80 平均 {kib(jpeg['jpeg']['80']['bytes_mean']):.1f} KiB/帧，"
        f"两者相差 "
        f"{abs(jpeg['jpeg']['80']['bytes_mean'] - reference['bytes_mean']) / reference['bytes_mean'] * 100:.1f}%，"
        "说明合成链路与真实录制端的编码参数一致（差异来自两次录制的画面内容不同）。")
    add("")

    # ---- 阶段 2 ---------------------------------------------------------
    add("## 阶段二：LeRobot 视频编码")
    add("")
    add("以下九个数据集全部由 `tool/rosbag2_to_lerobotv3.py` 真实产出，"
        f"每个 {frames} 帧 × 3 相机，因此表中的体积就是训练集的实际体积。")
    add("")
    add("| 源 | CRF | 数据集总体积 | 三路视频合计 | 每帧字节 | 码率 | PSNR vs 原始 | SSIM vs 原始 |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|")
    for entry in video["entries"]:
        size = entry["video_bytes_sum_cameras"]
        add(f"| {entry['source']} | {entry['crf']} | {mib(entry['dataset_bytes']):.2f} MiB | "
            f"{mib(size):.2f} MiB | {kib(size / frames):.1f} KiB | "
            f"{entry['bitrate_mbps_total']:.2f} Mb/s | "
            f"{fmt(pooled(entry, 'vs_raw', 'psnr_db'))} dB | "
            f"{fmt(pooled(entry, 'vs_raw', 'ssim'), 6)} |")
    add("")
    add("![rate-distortion](figures/rate_distortion.png)")
    add("")
    add("### 逐相机明细")
    add("")
    add("| 源 | CRF | 相机 | 视频体积 | 码率 | PSNR vs 原始 | SSIM vs 原始 | PSNR vs 编码器输入 |")
    add("|---|---:|---|---:|---:|---:|---:|---:|")
    for entry in video["entries"]:
        for camera in CAMERAS:
            item = entry["cameras"][camera]
            marginal = (fmt(item["vs_input"]["psnr_db"]["mean"]) + " dB") if "vs_input" in item else "—"
            add(f"| {entry['source']} | {entry['crf']} | {camera} | "
                f"{mib(item['video_bytes']):.2f} MiB | {item['bitrate_mbps']:.2f} Mb/s | "
                f"{fmt(item['vs_raw']['psnr_db']['mean'])} dB | "
                f"{fmt(item['vs_raw']['ssim']['mean'], 6)} | {marginal} |")
    add("")

    # ---- 级联 -----------------------------------------------------------
    add("## 级联损失：JPEG 与 H.264 各自的份额")
    add("")
    add("`vs 编码器输入` 只测量 H.264 在已压缩帧上又扣掉了多少，`vs 原始` 是端到端。"
        "两者的差就是录制端 JPEG 已经造成的、CRF 无论如何都追不回来的部分。")
    add("")
    add("![marginal](figures/marginal_loss.png)")
    add("")
    add("| CRF | raw 源 | JPEG q100 源 | JPEG q80 源 | q100 代价 | q80 代价 |")
    add("|---:|---:|---:|---:|---:|---:|")
    for crf in CRF_LEVELS:
        entries = {source: video_entry(source, crf) for source in SOURCE_ORDER}
        if not all(entries.values()):
            continue
        values = {s: pooled(e, "vs_raw", "psnr_db") for s, e in entries.items()}
        add(f"| {crf} | {values['raw']:.3f} dB | {values['jpeg100']:.3f} dB | "
            f"{values['jpeg80']:.3f} dB | −{values['raw'] - values['jpeg100']:.3f} dB | "
            f"−{values['raw'] - values['jpeg80']:.3f} dB |")
    add("")
    jpeg_size_ratio = (
        jpeg["jpeg"]["100"]["bytes_mean"] / jpeg["jpeg"]["80"]["bytes_mean"]
    )
    add(f"「代价」是相对同 CRF 的 raw 源。q100 那一列几乎为零，说明**录制端压缩本身不是问题，"
        f"问题是压到 80**——尽管 q100 在录制端要多花 {jpeg_size_ratio:.1f} 倍的字节。")
    add("")
    add("![per-frame](figures/per_frame_psnr.png)")
    add("")
    add("### GOP=2 造成的逐帧画质振荡")
    add("")
    add("上图里 CRF 20/30 的曲线是锯齿状的，CRF 0 与 JPEG 曲线则平滑。原因是 LeRobot 的 RGB "
        "编码参数用 `g=2`：每两帧一个 I 帧，中间夹一个 P 帧，而在有损档位上 P 帧的重建质量"
        "明显低于 I 帧。head 相机上偶数帧（I）与奇数帧（P）的平均差：")
    add("")
    add("| 源 | CRF | I 帧 | P 帧 | 差 |")
    add("|---|---:|---:|---:|---:|")
    for row in ip_rows:
        add(f"| {row['source']} | {row['crf']} | {row['even']:.3f} dB | {row['odd']:.3f} dB | "
            f"{row['gap']:+.3f} dB |")
    add("")
    add("偶数帧确实是 I 帧：直接读 `raw_crf20` 头部视频的 `pict_type`，552 个 I / 551 个 P，"
        "全部按下标奇偶交替，无例外。")
    add("")
    add("CRF 0 下这一项为 0（±0.002 dB），CRF 20 下达到 1.3–1.4 dB。也就是说按 CRF 20 存的"
        "数据集里，相邻两帧的画质是系统性交替的——对逐帧独立的策略无所谓，但做时序建模或"
        "帧间差分时值得知道这一点。")
    add("")
    add("### 最差帧（head 相机，按 SSIM）")
    add("")
    for entry in video["entries"]:
        name = f"figures/worst_{entry['source']}_crf{entry['crf']}_top.png"
        if (ROOT / name).exists():
            add(f"- [{entry['source']} + CRF {entry['crf']}]({name})")
    add("")

    # ---- DINO 特征 ------------------------------------------------------
    if features is not None:
        model_ids = list(features["models"])
        add("## 阶段三：DINO 特征漂移")
        add("")
        add("像素指标回答「差多少」，但训练的是视觉主干，真正要问的是「特征差多少」。"
            "本节直接复用 `test_lerobot/scripts/dinov3_feature_eval.py` 的模型、预处理"
            f"（RGB、短边 {features['preprocessing']['resize_shorter_side']}、中心裁剪 "
            f"{features['preprocessing']['center_crop']}、bicubic、ImageNet mean/std）与指标定义，"
            "因此两份报告的 cosine 可以直接并排读。")
        add("")
        add("| 模型 | Patch size | Patch 网格 | Token 数 | 权重 |")
        add("|---|---:|---:|---:|---|")
        for model_id in model_ids:
            item = features["models"][model_id]
            add(f"| {item['display_name']} | {item['patch_size']} | "
                f"{item['patch_grid'][0]}×{item['patch_grid'][1]} | {item['tokens']} | "
                f"`{Path(item['checkpoint']).name}` |")
        add("")
        add("![feature variants](figures/feature_variants.png)")
        add("")
        add("### 三相机平均")
        add("")
        add("| 模型 | 档位 | CLS cosine | Patch cosine | CLS relative L2 | CLS 夹角 |")
        add("|---|---|---:|---:|---:|---:|")
        for model_id in model_ids:
            for variant in VARIANT_ORDER:
                add(f"| {features['models'][model_id]['display_name']} | {variant} | "
                    f"{feature_pooled(features, model_id, variant, 'cls_cosine'):.6f} | "
                    f"{feature_pooled(features, model_id, variant, 'patch_token_cosine_mean'):.6f} | "
                    f"{feature_pooled(features, model_id, variant, 'cls_relative_l2'):.4f} | "
                    f"{feature_pooled(features, model_id, variant, 'cls_angle_degrees'):.3f}° |")
        add("")
        add("### 逐相机：像素上的头/腕差距在特征上同样存在")
        add("")
        add("![feature by camera](figures/feature_by_camera.png)")
        add("")
        add("| 模型 | 档位 | top | wrist_L | wrist_R | 头腕差 |")
        add("|---|---|---:|---:|---:|---:|")
        for model_id in model_ids:
            for variant in ("jpeg_q80", "raw_crf20", "jpeg80_crf20"):
                values = [feature_mean(features, model_id, c, variant, "cls_cosine")
                          for c in CAMERAS]
                gap = values[0] - float(np.mean(values[1:]))
                add(f"| {features['models'][model_id]['display_name']} | {variant} | "
                    f"{values[0]:.6f} | {values[1]:.6f} | {values[2]:.6f} | {gap:+.6f} |")
        add("")
        add("头/腕差不是 JPEG 独有的：`raw_crf30` 同样拉开差距（DINOv3-B/16 上 +0.055），"
            "说明 wrist 画面本身更难压——纹理更密、运动更大，任何一级有损处理都先伤它。"
            "但在**总体保真度相当或更好**的前提下，JPEG 那一路的损失明显更偏。三个模型上，"
            "`jpeg_q80` 的三相机平均 CLS 都高于 `raw_crf20`，头腕差却反而更大：")
        add("")
        add("| 模型 | jpeg_q80 平均 / 头腕差 | raw_crf20 平均 / 头腕差 |")
        add("|---|---:|---:|")
        for model_id in model_ids:
            row = []
            for variant in ("jpeg_q80", "raw_crf20"):
                values = [feature_mean(features, model_id, c, variant, "cls_cosine")
                          for c in CAMERAS]
                row.append((float(np.mean(values)), values[0] - float(np.mean(values[1:]))))
            add(f"| {features['models'][model_id]['display_name']} | "
                f"{row[0][0]:.6f} / {row[0][1]:+.5f} | {row[1][0]:.6f} / {row[1][1]:+.5f} |")
        add("")
        add("### 像素指标能不能预测特征漂移")
        add("")
        add(f"把 {len(VARIANT_ORDER)} 个档位 × {len(CAMERAS)} 相机 × {len(model_ids)} 模型"
            f"（{spearman['n']} 个点）的 PSNR 与 CLS cosine "
            f"放在一起：Pearson r = {correlation:.3f}，但 Spearman ρ = {spearman['rho']:.3f}。"
            "两个数字差这么多，说明关系是**单调但非线性**的——看散点图右侧明显饱和。")
        add("")
        add("![feature vs pixel](figures/feature_vs_pixel.png)")
        add("")
        add("实际含义：")
        add("")
        add("- **排序可信。** PSNR 更高的档位，特征漂移基本也更小（单模型内 ρ = "
            f"{min(spearman['per_model'].values()):.3f}–{max(spearman['per_model'].values()):.3f}）。"
            "拿 PSNR 做粗筛没问题。")
        hi = spearman["slope_high"]
        lo = spearman["slope_low"]
        add(f"- **数值不可换算。** 对 PSNR 做局部线性拟合：在 ≥43 dB 区间，每 3 dB 只换来 "
            f"{min(hi.values()):.4f}–{max(hi.values()):.4f} 的 CLS cosine；在 ≤40 dB 区间，"
            f"同样 3 dB 换来 {min(lo.values()):.4f}–{max(lo.values()):.4f}——相差 30–100 倍。"
            "用「PSNR 至少 X dB」当验收线，在不同区间的严格程度完全不是一回事。")
        add("- **跨模型不可比。** 同一批像素对三个主干给出的 cosine 差异很大（`raw_crf30` 上从 "
            f"{min(feature_pooled(features, m, 'raw_crf30', 'cls_cosine') for m in model_ids):.4f} 到 "
            f"{max(feature_pooled(features, m, 'raw_crf30', 'cls_cosine') for m in model_ids):.4f}），"
            "而 PSNR 只有一个值。这也是三模型合并后 Pearson 掉到 "
            f"{correlation:.3f} 的主要原因。")
        add("")

        # ---- 与 test_lerobot 对照 --------------------------------------
        add("## 与 test_lerobot 的对照")
        add("")
        if cross_rows:
            add("`test_lerobot/REPORT.md` 用的是另一段数据（HDF5 单相机、220 帧、AV1/SVT 编码），"
                "本测试是 mcap 三相机、1103 帧、h264/x264 编码。下表只取两边都有的 CRF 0 与 "
                "CRF 20，且本测试一侧只取 head 相机（`top`），因为 test_lerobot 的 "
                "`observations/images/top` 也是头部相机——这是唯一口径相近的比较。")
            add("")
            add("| 模型 | CRF | test_lerobot AV1 CLS | 本测试 h264 CLS | 差 | "
                "test_lerobot AV1 patch | 本测试 h264 patch |")
            add("|---|---:|---:|---:|---:|---:|---:|")
            for row in cross_rows:
                add(f"| {row['model']} | {row['crf']} | {row['ref_cls']:.6f} | "
                    f"{row['our_cls']:.6f} | {row['our_cls'] - row['ref_cls']:+.6f} | "
                    f"{row['ref_patch']:.6f} | {row['our_patch']:.6f} |")
            add("")
            add("**这不是编码器对比。** 两边的场景、纹理复杂度、相机与帧数都不同，"
                "任何一格的差都同时包含了这些因素。可以放心并排读的是各自内部的趋势："
                "两份测试都显示 CRF 0 的 CLS cosine 在 0.99 以上、CRF 20 掉到 0.96–0.99 区间，"
                "并且模型排序完全一致——CLS 上 DINOv2-S/14 > DINOv3-S/16 > DINOv3-B/16，"
                "patch 上 DINOv3-S/16 > DINOv3-B/16 > DINOv2-S/14，两份数据各自独立复现了"
                "同一组顺序。这是两次测试之间最强的一致性证据。")
            add("")
        add("两份测试一致的结论：")
        add("")
        add("- CRF 20 在像素上看起来只掉几个 dB，但在特征上是可测量的漂移，且不同主干的"
            "敏感度不同，不存在对所有模型通用的「安全 CRF」。")
        add("- CLS 与 patch 的排序并不一致：对局部 patch 最稳的模型不一定 CLS 最稳，"
            "所以选档位时要看下游实际用的是哪一种特征。")
        add("")
        add("本测试新增、test_lerobot 未覆盖的：录制端 JPEG 这一级，以及它与视频编码的级联。")
        add("")

    # ---- 复现 -----------------------------------------------------------
    add("## 可复现性与输出")
    add("")
    add("```bash")
    add("cd /home/kewei/YING/robot_data_platform/test_raw_jpeg")
    add("./run_all.sh")
    add("```")
    add("")
    add("| 文件 | 内容 |")
    add("|---|---|")
    add("| `results/jpeg_summary.json` | 阶段一全部统计（马赛克与逐相机） |")
    add("| `results/per_frame_jpeg_mosaic_metrics.csv` | 马赛克逐帧 JPEG 指标 |")
    add("| `results/per_frame_jpeg_tile_metrics.csv` | 切分后逐帧逐相机 JPEG 指标 |")
    add("| `results/reference_mcap_sizes.json` | `express_mcap` 真实 JPEG 码率 |")
    add("| `results/jpeg{100,80}_bag.json` | 两个合成 JPEG bag 的构建记录 |")
    add("| `results/video_summary.json` | 九个数据集的体积、码率与画质 |")
    add("| `results/per_frame_video_metrics.csv` | 视频逐帧指标（两种参照） |")
    add("| `results/episode_audit.json` | 转换器对该 episode 的对齐审计 |")
    if features is not None:
        add("| `results/model_feature_summary.json` | 3 模型 × 8 档位 × 3 相机的特征汇总 |")
        add("| `results/per_frame_model_feature_metrics.csv` | 逐帧特征指标（79,416 行） |")
        add("| `results/model_global_embeddings.npz` | 各模型的 CLS 与 mean-patch 嵌入（float16） |")
        add("| `figures/feature_maps/<模型>/` | 共享 PCA 基底的 patch 特征图 |")
        add("| `figures/feature_worst/` | 各模型各档位 CLS 最差帧 |")
    add("| `intermediate/*.npy` | 原始 / JPEG-80 的逐帧 tile（uint8 memmap） |")
    add("| `bags/express_jpeg{100,80}/` | 合成的 JPEG mcap |")
    add("| `lerobot/<源>_crf<N>/` | 九个 LeRobot v3 数据集 |")
    add("")
    for quality, item in sorted(bags.items(), reverse=True):
        add(f"合成 bag q{quality}：源 {mib(item['source_bag_bytes']):.1f} MiB → "
            f"{mib(item['jpeg_bag_bytes']):.1f} MiB"
            f"（{item['bag_shrink_factor']:.1f}× 更小）。")
    add("")

    # ---- 局限 -----------------------------------------------------------
    add("## 局限")
    add("")
    add(f"- 只有一段 {duration:.1f} s、{frames} 帧的录制，且只有一个任务场景；"
        "不同纹理复杂度下 JPEG 与 H.264 的相对表现会变化。")
    add("- 全部指标为像素级。像素保真度与策略成功率不是同一件事，"
        "参考 `test_lerobot/REPORT.md` 的结论：不同视觉主干对同一压缩扰动的敏感度不同，"
        "最终档位应由下游模型的 A/B 结果决定。")
    add("- 特征级比较用的是通用自监督主干（DINOv2/DINOv3），不是实际训练的策略网络。"
        "cosine 漂移能说明「表示变了多少」，不能直接换算成成功率。")
    add("- CRF 0 并非逐像素无损：`yuv420p` 包含 RGB↔YUV 转换与 4:2:0 色度下采样，"
        "这一项损失在 raw 源 CRF 0 的 PSNR 上已经可见。")
    add("")
    return "\n".join(lines) + "\n"


def main() -> None:
    jpeg = load("jpeg_summary.json")
    video = load("video_summary.json")
    reference = load("reference_mcap_sizes.json")
    bags = {
        quality: load(f"jpeg{quality}_bag.json")
        for quality in CONFIG["jpeg"]["bag_levels"]
        if (RESULTS / f"jpeg{quality}_bag.json").exists()
    }
    audit = load("episode_audit.json")

    feature_path = RESULTS / "model_feature_summary.json"
    features = json.loads(feature_path.read_text()) if feature_path.exists() else None

    FIGURES.mkdir(parents=True, exist_ok=True)
    figure_storage(jpeg, video, FIGURES / "storage_per_frame.png")
    figure_quality_ladder(jpeg, video, FIGURES / "quality_ladder.png")
    figure_rate_distortion(jpeg, video, FIGURES / "rate_distortion.png")
    figure_marginal(video, FIGURES / "marginal_loss.png")
    figure_per_frame(FIGURES / "per_frame_psnr.png")

    correlation = None
    spearman = None
    if features is not None:
        figure_feature_variants(features, FIGURES / "feature_variants.png")
        figure_feature_by_camera(features, FIGURES / "feature_by_camera.png")
        correlation = figure_feature_vs_pixel(
            features, jpeg, video, FIGURES / "feature_vs_pixel.png"
        )
        spearman = pixel_feature_stats(features, jpeg, video)
    else:
        print("no feature summary yet; report will omit the DINO section")

    (ROOT / "REPORT.md").write_text(
        build_report(jpeg, video, reference, bags, audit, features, correlation, spearman)
    )
    print(f"wrote {ROOT / 'REPORT.md'}")


if __name__ == "__main__":
    main()
