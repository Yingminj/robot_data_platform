#!/usr/bin/env python3
"""Create plots and a Chinese Markdown report from the measured results."""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
LABELS = ["crf0_min_loss", "crf20_balanced", "crf50_max_loss"]
DISPLAY = {
    "crf0_min_loss": "CRF 0",
    "crf20_balanced": "CRF 20",
    "crf50_max_loss": "CRF 50",
}
COLORS = {
    "crf0_min_loss": "#3478bf",
    "crf20_balanced": "#41a66b",
    "crf50_max_loss": "#d95f4b",
}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["frame_index"] = int(row["frame_index"])
        for key in row:
            if key not in {"frame_index", "variant"}:
                row[key] = float(row[key])
    return rows


def variant_series(rows: list[dict], label: str, key: str) -> np.ndarray:
    selected = sorted((row for row in rows if row["variant"] == label), key=lambda row: row["frame_index"])
    return np.asarray([row[key] for row in selected], dtype=np.float64)


def corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1])


def mib(value: float) -> float:
    return value / (1024.0**2)


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def plot_storage(pixel: dict) -> None:
    raw = pixel["source"]["camera_hdf5_storage_bytes"]
    values = [mib(raw)] + [mib(pixel["variants"][label]["video_size_bytes"]) for label in LABELS]
    names = ["HDF5 raw RGB", *[f"AV1 {DISPLAY[label]}" for label in LABELS]]
    colors = ["#777777", *[COLORS[label] for label in LABELS]]
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    bars = ax.bar(names, values, color=colors)
    ax.set_ylabel("Storage (MiB)")
    ax.set_title("Top-camera storage")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(FIGURES / "storage_comparison.png", dpi=180)
    plt.close(fig)


def plot_pixel_timeseries(rows: list[dict]) -> None:
    frames = np.arange(len(variant_series(rows, LABELS[0], "psnr_db")))
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.4), sharex=True)
    for label in LABELS:
        axes[0].plot(frames, variant_series(rows, label, "psnr_db"), label=DISPLAY[label], color=COLORS[label], lw=1.3)
        axes[1].plot(frames, variant_series(rows, label, "ssim"), label=DISPLAY[label], color=COLORS[label], lw=1.3)
    axes[0].set_ylabel("PSNR (dB)")
    axes[1].set_ylabel("SSIM")
    axes[1].set_xlabel("Frame index")
    axes[0].set_title("Per-frame pixel fidelity")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "pixel_metrics_by_frame.png", dpi=180)
    plt.close(fig)


def plot_dino_timeseries(rows: list[dict]) -> None:
    frames = np.arange(len(variant_series(rows, LABELS[0], "cls_cosine")))
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.4), sharex=True)
    for label in LABELS:
        axes[0].plot(frames, variant_series(rows, label, "cls_cosine"), label=DISPLAY[label], color=COLORS[label], lw=1.3)
        axes[1].plot(
            frames,
            variant_series(rows, label, "patch_token_cosine_mean"),
            label=DISPLAY[label],
            color=COLORS[label],
            lw=1.3,
        )
    axes[0].set_ylabel("CLS cosine")
    axes[1].set_ylabel("Mean patch-token cosine")
    axes[1].set_xlabel("Frame index")
    axes[0].set_title("Per-frame DINOv3 feature fidelity")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend()
        ax.set_ylim(min(ax.get_ylim()[0], 0.87), 1.002)
    fig.tight_layout()
    fig.savefig(FIGURES / "dinov3_metrics_by_frame.png", dpi=180)
    plt.close(fig)


def plot_tradeoff(pixel: dict, dino: dict) -> None:
    raw = pixel["source"]["camera_hdf5_storage_bytes"]
    x = [100.0 * pixel["variants"][label]["video_size_bytes"] / raw for label in LABELS]
    y = [dino["variants"][label]["cls_cosine"]["mean"] for label in LABELS]
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    for i, label in enumerate(LABELS):
        ax.scatter(x[i], y[i], s=110, color=COLORS[label], label=DISPLAY[label], zorder=3)
        if label == "crf0_min_loss":
            offset = (-48, -2)
        elif label == "crf20_balanced":
            offset = (8, -14)
        else:
            offset = (8, 5)
        ax.annotate(DISPLAY[label], (x[i], y[i]), xytext=offset, textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("Video / raw HDF5 top-camera storage (%) [log scale]")
    ax.set_ylabel("Mean DINOv3 CLS cosine")
    ax.set_title("Storage vs semantic-feature fidelity")
    ax.grid(alpha=0.3, which="both")
    ax.set_ylim(min(y) - 0.02, 1.003)
    fig.tight_layout()
    fig.savefig(FIGURES / "storage_feature_tradeoff.png", dpi=180)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    pixel = load_json(RESULTS / "pixel_summary.json")
    dino = load_json(RESULTS / "dinov3_summary.json")
    pixel_rows = load_csv(RESULTS / "per_frame_pixel_metrics.csv")
    dino_rows = load_csv(RESULTS / "per_frame_dinov3_metrics.csv")
    plot_storage(pixel)
    plot_pixel_timeseries(pixel_rows)
    plot_dino_timeseries(dino_rows)
    plot_tradeoff(pixel, dino)

    correlations = {}
    for label in LABELS:
        correlations[label] = {
            "ssim_vs_cls_cosine": corr(
                variant_series(pixel_rows, label, "ssim"),
                variant_series(dino_rows, label, "cls_cosine"),
            ),
            "psnr_vs_cls_cosine": corr(
                variant_series(pixel_rows, label, "psnr_db"),
                variant_series(dino_rows, label, "cls_cosine"),
            ),
            "ssim_vs_patch_cosine": corr(
                variant_series(pixel_rows, label, "ssim"),
                variant_series(dino_rows, label, "patch_token_cosine_mean"),
            ),
        }
    with (RESULTS / "cross_metric_correlations.json").open("w", encoding="utf-8") as f:
        json.dump(correlations, f, indent=2, ensure_ascii=False)

    src = pixel["source"]
    p0 = pixel["variants"][LABELS[0]]
    p20 = pixel["variants"][LABELS[1]]
    p50 = pixel["variants"][LABELS[2]]
    d0 = dino["variants"][LABELS[0]]
    d20 = dino["variants"][LABELS[1]]
    d50 = dino["variants"][LABELS[2]]
    size_factor = p0["video_size_bytes"] / p50["video_size_bytes"]
    size_factor_20 = src["camera_hdf5_storage_bytes"] / p20["video_size_bytes"]
    cls_error_factor = (1.0 - d50["cls_cosine"]["mean"]) / (1.0 - d0["cls_cosine"]["mean"])
    patch_error_factor = (1.0 - d50["patch_token_cosine_mean"]["mean"]) / (
        1.0 - d0["patch_token_cosine_mean"]["mean"]
    )
    report = f"""# HDF5 单帧与 LeRobot AV1 视频存储对比报告

生成时间：{datetime.now().astimezone().isoformat(timespec="seconds")}

## 结论摘要

本测试只比较头部相机 `observations/images/top` 的 {src["frame_count"]} 帧 RGB 图像（{src["width"]}×{src["height"]}，30 FPS），不是整个 HDF5 文件。原始头部相机数组实际占用 {mib(src["camera_hdf5_storage_bytes"]):.2f} MiB。

| 指标 | 原始 HDF5 单帧 | AV1 CRF 0 | AV1 CRF 20 | AV1 CRF 50 |
|---|---:|---:|---:|---:|
| 存储大小 | {mib(src["camera_hdf5_storage_bytes"]):.2f} MiB | {mib(p0["video_size_bytes"]):.2f} MiB | {mib(p20["video_size_bytes"]):.2f} MiB | {mib(p50["video_size_bytes"]):.2f} MiB |
| 占原始头部相机比例 | 100% | {pct(p0["ratio_vs_hdf5_camera_storage"])} | {pct(p20["ratio_vs_hdf5_camera_storage"])} | {pct(p50["ratio_vs_hdf5_camera_storage"])} |
| 空间节省 | 0% | {pct(p0["space_saving_vs_raw_logical"])} | {pct(p20["space_saving_vs_raw_logical"])} | {pct(p50["space_saving_vs_raw_logical"])} |
| 平均 MSE（0–255） | 0 | {p0["pixel_metrics"]["mse_255"]["mean"]:.4f} | {p20["pixel_metrics"]["mse_255"]["mean"]:.4f} | {p50["pixel_metrics"]["mse_255"]["mean"]:.4f} |
| 平均 MAE（0–255） | 0 | {p0["pixel_metrics"]["mae_255"]["mean"]:.4f} | {p20["pixel_metrics"]["mae_255"]["mean"]:.4f} | {p50["pixel_metrics"]["mae_255"]["mean"]:.4f} |
| 平均 PSNR | ∞ | {p0["pixel_metrics"]["psnr_db"]["mean"]:.3f} dB | {p20["pixel_metrics"]["psnr_db"]["mean"]:.3f} dB | {p50["pixel_metrics"]["psnr_db"]["mean"]:.3f} dB |
| 平均 SSIM | 1 | {p0["pixel_metrics"]["ssim"]["mean"]:.6f} | {p20["pixel_metrics"]["ssim"]["mean"]:.6f} | {p50["pixel_metrics"]["ssim"]["mean"]:.6f} |
| DINOv3 CLS 余弦相似度 | 1 | {d0["cls_cosine"]["mean"]:.6f} | {d20["cls_cosine"]["mean"]:.6f} | {d50["cls_cosine"]["mean"]:.6f} |
| DINOv3 平均 patch-token 余弦相似度 | 1 | {d0["patch_token_cosine_mean"]["mean"]:.6f} | {d20["patch_token_cosine_mean"]["mean"]:.6f} | {d50["patch_token_cosine_mean"]["mean"]:.6f} |

CRF 0 将头部相机存储降到原始的 {pct(p0["ratio_vs_hdf5_camera_storage"])}，同时保持较高的像素和 DINOv3 特征一致性。CRF 20 将原始数据压缩约 {size_factor_20:.1f} 倍，平均 CLS cosine 仍为 {d20["cls_cosine"]["mean"]:.6f}，是本测试中更实用的质量/容量折中。CRF 50 的文件又比 CRF 0 小 {size_factor:.1f} 倍，但 DINOv3 CLS 余弦距离 `1-cos` 放大到约 {cls_error_factor:.1f} 倍，逐 patch 的余弦距离放大到约 {patch_error_factor:.1f} 倍。

因此：

- 若数据用于 DINO/视觉表征训练或离线特征提取，CRF 0 明显更稳妥。
- CRF 20 在本片段上保留了较高的 DINOv3 特征一致性，同时显著降低存储，是优先建议进一步做策略成功率 A/B 测试的档位。
- CRF 50 适合极端节省空间、人工浏览或低保真预览；不建议未经下游任务 A/B 验证就作为训练主数据。
- “视觉上仍清晰”不能替代特征评估：CRF 50 的平均 SSIM 仍为 {p50["pixel_metrics"]["ssim"]["mean"]:.4f}，但 CLS 相似度已降至 {d50["cls_cosine"]["mean"]:.4f}。

![存储对比](figures/storage_comparison.png)

![存储与特征权衡](figures/storage_feature_tradeoff.png)

## 测试设置

- HDF5：`{src["path"]}`
- 相机键：`{src["camera_key"]}`；HDF5 中为 RGB 顺序、`uint8`、无压缩、chunk 为 `{src["hdf5_chunks"]}`
- 视频：MP4 容器，`libsvtav1`，`yuv420p`，GOP=2，30 FPS，fast-decode=0
- 质量档：CRF 0、CRF 20 与 CRF 50
- LeRobot 当前 RGB 默认值是 AV1 / yuv420p / GOP 2 / CRF 30 / preset 12；本实验仅将 CRF 改成 0、20、50 三档。[LeRobot 编码参数文档](https://huggingface.co/docs/lerobot/main/video_encoding_parameters)
- 当前 SVT-AV1 3.1.2 把请求的 preset 12 映射为实际 preset 10；三档均发生相同映射，因此质量对比仍受控。
- 当前 FFmpeg `libsvtav1` wrapper 只有 `crf > 0` 才向 SVT 写入 CRF，直接传 `crf=0` 会静默回落到默认 35。本测试对 CRF 0 使用 `svtav1-params=crf=0` 直传，并由 SVT 日志确认实际 `CRF / 0`。[FFmpeg wrapper 源码](https://www.ffmpeg.org/doxygen/trunk/libsvtav1_8c_source.html)

CRF 0 不代表最终 RGB 像素完全无损：`yuv420p` 会进行 RGB↔YUV 转换和 4:2:0 色度下采样，所以仍存在像素差异。这里的“最低损失”指指定 LeRobot AV1/yuv420p 方案下的最低 CRF。

## 逐帧像素结果

| 档位 | PSNR 均值 / 最低 | SSIM 均值 / 最低 | MAE 均值 | 95% 像素通道绝对误差 | 最大绝对误差 |
|---|---:|---:|---:|---:|---:|
| CRF 0 | {p0["pixel_metrics"]["psnr_db"]["mean"]:.3f} / {p0["pixel_metrics"]["psnr_db"]["min"]:.3f} dB | {p0["pixel_metrics"]["ssim"]["mean"]:.6f} / {p0["pixel_metrics"]["ssim"]["min"]:.6f} | {p0["pixel_metrics"]["mae_255"]["mean"]:.3f} | {p0["pixel_metrics"]["p95_abs_error"]["mean"]:.2f} | {p0["pixel_metrics"]["max_abs_error"]["max"]:.0f} |
| CRF 20 | {p20["pixel_metrics"]["psnr_db"]["mean"]:.3f} / {p20["pixel_metrics"]["psnr_db"]["min"]:.3f} dB | {p20["pixel_metrics"]["ssim"]["mean"]:.6f} / {p20["pixel_metrics"]["ssim"]["min"]:.6f} | {p20["pixel_metrics"]["mae_255"]["mean"]:.3f} | {p20["pixel_metrics"]["p95_abs_error"]["mean"]:.2f} | {p20["pixel_metrics"]["max_abs_error"]["max"]:.0f} |
| CRF 50 | {p50["pixel_metrics"]["psnr_db"]["mean"]:.3f} / {p50["pixel_metrics"]["psnr_db"]["min"]:.3f} dB | {p50["pixel_metrics"]["ssim"]["mean"]:.6f} / {p50["pixel_metrics"]["ssim"]["min"]:.6f} | {p50["pixel_metrics"]["mae_255"]["mean"]:.3f} | {p50["pixel_metrics"]["p95_abs_error"]["mean"]:.2f} | {p50["pixel_metrics"]["max_abs_error"]["max"]:.0f} |

![逐帧像素指标](figures/pixel_metrics_by_frame.png)

CRF 0 的最差 SSIM 帧为 {p0["worst_ssim_frame"]}，CRF 20 为 {p20["worst_ssim_frame"]}，CRF 50 为 {p50["worst_ssim_frame"]}。可视化文件：

- [CRF 0 最差像素帧](figures/worst_pixel_frame_crf0_min_loss.png)
- [CRF 20 最差像素帧](figures/worst_pixel_frame_crf20_balanced.png)
- [CRF 50 最差像素帧](figures/worst_pixel_frame_crf50_max_loss.png)

## DINOv3 特征影响

模型为 `dinov3_vits16`，权重维度 {dino["model"]["embedding_dimension"]}。预处理严格采用仓库的分类评估流程：短边缩放至 256、中心裁剪 224×224、bicubic、ImageNet mean/std；因此 DINO 指标对应中心裁剪区域，而像素指标对应完整 640×480 图像。

| 指标 | CRF 0 | CRF 20 | CRF 50 |
|---|---:|---:|---:|
| CLS cosine（均值 / 最低） | {d0["cls_cosine"]["mean"]:.6f} / {d0["cls_cosine"]["min"]:.6f} | {d20["cls_cosine"]["mean"]:.6f} / {d20["cls_cosine"]["min"]:.6f} | {d50["cls_cosine"]["mean"]:.6f} / {d50["cls_cosine"]["min"]:.6f} |
| CLS 夹角均值 | {d0["cls_angle_degrees"]["mean"]:.3f}° | {d20["cls_angle_degrees"]["mean"]:.3f}° | {d50["cls_angle_degrees"]["mean"]:.3f}° |
| CLS relative L2 均值 | {d0["cls_relative_l2"]["mean"]:.4f} | {d20["cls_relative_l2"]["mean"]:.4f} | {d50["cls_relative_l2"]["mean"]:.4f} |
| mean-patch cosine 均值 | {d0["mean_patch_cosine"]["mean"]:.6f} | {d20["mean_patch_cosine"]["mean"]:.6f} | {d50["mean_patch_cosine"]["mean"]:.6f} |
| patch-token cosine 均值 | {d0["patch_token_cosine_mean"]["mean"]:.6f} | {d20["patch_token_cosine_mean"]["mean"]:.6f} | {d50["patch_token_cosine_mean"]["mean"]:.6f} |
| 单 patch 最低 cosine（全帧最差） | {d0["patch_token_cosine_min"]["min"]:.6f} | {d20["patch_token_cosine_min"]["min"]:.6f} | {d50["patch_token_cosine_min"]["min"]:.6f} |

![逐帧 DINOv3 指标](figures/dinov3_metrics_by_frame.png)

最差 CLS 帧：CRF 0 为 {d0["worst_cls_cosine_frame"]}，CRF 20 为 {d20["worst_cls_cosine_frame"]}，CRF 50 为 {d50["worst_cls_cosine_frame"]}。

- [CRF 0 最差 DINOv3 CLS 帧](figures/worst_dinov3_cls_frame_crf0_min_loss.png)
- [CRF 20 最差 DINOv3 CLS 帧](figures/worst_dinov3_cls_frame_crf20_balanced.png)
- [CRF 50 最差 DINOv3 CLS 帧](figures/worst_dinov3_cls_frame_crf50_max_loss.png)

## 像素指标与 DINO 指标相关性

| 档位 | SSIM vs CLS cosine | PSNR vs CLS cosine | SSIM vs patch cosine |
|---|---:|---:|---:|
| CRF 0 | {correlations[LABELS[0]]["ssim_vs_cls_cosine"]:.3f} | {correlations[LABELS[0]]["psnr_vs_cls_cosine"]:.3f} | {correlations[LABELS[0]]["ssim_vs_patch_cosine"]:.3f} |
| CRF 20 | {correlations[LABELS[1]]["ssim_vs_cls_cosine"]:.3f} | {correlations[LABELS[1]]["psnr_vs_cls_cosine"]:.3f} | {correlations[LABELS[1]]["ssim_vs_patch_cosine"]:.3f} |
| CRF 50 | {correlations[LABELS[2]]["ssim_vs_cls_cosine"]:.3f} | {correlations[LABELS[2]]["psnr_vs_cls_cosine"]:.3f} | {correlations[LABELS[2]]["ssim_vs_patch_cosine"]:.3f} |

相关性只描述本段 220 帧内的共变关系，不能外推为通用因果关系。完整数值保存在 `results/cross_metric_correlations.json`。

## 方法与可复现性

1. 从 HDF5 按顺序读取原始 RGB 帧。
2. 使用与 LeRobot 相同的 PyAV 编码路径写入 AV1 MP4。
3. 顺序解码并严格验证三档视频均为 220 帧、640×480、30 FPS，逐帧一一对齐。
4. 在原始/解码帧上计算 MSE、RMSE、MAE、PSNR、SSIM、误差分位数、RGB 通道误差与偏置。
5. 对原始帧和三档解码帧使用同一 DINOv3 模型与预处理，比较 CLS、mean-patch 及 196 个位置对应 patch token。

LeRobot 的原始视频基准同样使用 MSE、PSNR、SSIM，并明确指出视频感知质量未必等价于神经网络输入质量。[LeRobot 视频编码研究](https://huggingface.co/blog/video-encoding)

运行：

```bash
cd {ROOT}
./run_all.sh --force
```

主要机器可读结果：

- `results/per_frame_pixel_metrics.csv`
- `results/pixel_summary.json`
- `results/per_frame_dinov3_metrics.csv`
- `results/dinov3_summary.json`
- `results/dinov3_global_embeddings.npz`
- `results/cross_metric_correlations.json`

## 局限

- 只有一个 7.33 秒片段和一个固定头部相机；对其他场景、运动速度、纹理和光照不能直接外推。
- 本测试衡量 DINOv3 表征漂移，没有训练并评估具体机器人策略；最终 CRF 仍应通过策略成功率 A/B 测试确定。
- 没有测试 LeRobot 默认 CRF 30；CRF 20 是新增的中间档，但最终最优值仍需更多片段和策略成功率验证。
- 编码/解码速度来自单次运行，且像素评估时间包含 Python 指标计算，不应视为纯解码吞吐基准。
"""
    with (ROOT / "REPORT.md").open("w", encoding="utf-8") as f:
        f.write(report)
    print(f"Wrote {ROOT / 'REPORT.md'}")


if __name__ == "__main__":
    main()
