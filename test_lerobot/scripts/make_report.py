#!/usr/bin/env python3
"""Create plots and a Chinese report for pixel and multi-model feature results."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
VARIANTS = ["crf0_min_loss", "crf20_balanced", "crf50_max_loss"]
VARIANT_NAMES = {
    "crf0_min_loss": "CRF 0",
    "crf20_balanced": "CRF 20",
    "crf50_max_loss": "CRF 50",
}
VARIANT_COLORS = {
    "crf0_min_loss": "#3478bf",
    "crf20_balanced": "#41a66b",
    "crf50_max_loss": "#d95f4b",
}
MODELS = ["dinov3_vits16", "dinov3_vitb16", "dinov2_vits14"]
MODEL_NAMES = {
    "dinov3_vits16": "DINOv3-S/16",
    "dinov3_vitb16": "DINOv3-B/16",
    "dinov2_vits14": "DINOv2-S/14",
}
MODEL_COLORS = {
    "dinov3_vits16": "#3478bf",
    "dinov3_vitb16": "#8c57a5",
    "dinov2_vits14": "#e38b2c",
}
MODEL_MARKERS = {
    "dinov3_vits16": "o",
    "dinov3_vitb16": "s",
    "dinov2_vits14": "^",
}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    text_fields = {"variant", "model_id", "model_name"}
    for row in rows:
        row["frame_index"] = int(row["frame_index"])
        for key in row:
            if key not in text_fields | {"frame_index"}:
                row[key] = float(row[key])
    return rows


def select_series(
    rows: list[dict],
    *,
    variant: str,
    key: str,
    model_id: str | None = None,
) -> np.ndarray:
    selected = [
        row
        for row in rows
        if row["variant"] == variant and (model_id is None or row.get("model_id") == model_id)
    ]
    selected.sort(key=lambda row: row["frame_index"])
    return np.asarray([row[key] for row in selected], dtype=np.float64)


def corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1])


def mib(value: float) -> float:
    return value / (1024.0**2)


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def plot_storage(pixel: dict) -> None:
    raw = pixel["source"]["camera_hdf5_storage_bytes"]
    values = [mib(raw)] + [mib(pixel["variants"][variant]["video_size_bytes"]) for variant in VARIANTS]
    names = ["HDF5 raw RGB", *[f"AV1 {VARIANT_NAMES[variant]}" for variant in VARIANTS]]
    colors = ["#777777", *[VARIANT_COLORS[variant] for variant in VARIANTS]]
    fig, ax = plt.subplots(figsize=(7.8, 4.5))
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
    frames = np.arange(len(select_series(rows, variant=VARIANTS[0], key="psnr_db")))
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.4), sharex=True)
    for variant in VARIANTS:
        axes[0].plot(
            frames,
            select_series(rows, variant=variant, key="psnr_db"),
            label=VARIANT_NAMES[variant],
            color=VARIANT_COLORS[variant],
            lw=1.3,
        )
        axes[1].plot(
            frames,
            select_series(rows, variant=variant, key="ssim"),
            label=VARIANT_NAMES[variant],
            color=VARIANT_COLORS[variant],
            lw=1.3,
        )
    axes[0].set_ylabel("PSNR (dB)")
    axes[1].set_ylabel("SSIM")
    axes[1].set_xlabel("Frame index")
    axes[0].set_title("Per-frame pixel fidelity")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(ncol=3)
    fig.tight_layout()
    fig.savefig(FIGURES / "pixel_metrics_by_frame.png", dpi=180)
    plt.close(fig)


def plot_model_timeseries(rows: list[dict], key: str, filename: str, ylabel: str, title: str) -> None:
    fig, axes = plt.subplots(len(MODELS), 1, figsize=(10, 8.5), sharex=True, sharey=True)
    for ax, model_id in zip(axes, MODELS):
        for variant in VARIANTS:
            values = select_series(rows, model_id=model_id, variant=variant, key=key)
            ax.plot(
                np.arange(len(values)),
                values,
                label=VARIANT_NAMES[variant],
                color=VARIANT_COLORS[variant],
                lw=1.25,
            )
        ax.set_ylabel(ylabel)
        ax.set_title(MODEL_NAMES[model_id], loc="left", fontsize=11)
        ax.grid(alpha=0.25)
        ax.set_ylim(min(ax.get_ylim()[0], 0.80), 1.003)
    axes[0].legend(ncol=3, loc="lower left")
    axes[-1].set_xlabel("Frame index")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(FIGURES / filename, dpi=180)
    plt.close(fig)


def plot_model_bars(feature: dict) -> None:
    x = np.arange(len(VARIANTS))
    width = 0.25
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for model_index, model_id in enumerate(MODELS):
        model = feature["models"][model_id]
        cls_values = [model["variants"][variant]["cls_cosine"]["mean"] for variant in VARIANTS]
        patch_values = [
            model["variants"][variant]["patch_token_cosine_mean"]["mean"] for variant in VARIANTS
        ]
        offset = (model_index - 1) * width
        axes[0].bar(x + offset, cls_values, width, label=MODEL_NAMES[model_id], color=MODEL_COLORS[model_id])
        axes[1].bar(x + offset, patch_values, width, label=MODEL_NAMES[model_id], color=MODEL_COLORS[model_id])
    for ax, title in zip(axes, ["CLS feature", "Position-aligned patch tokens"]):
        ax.set_xticks(x, [VARIANT_NAMES[variant] for variant in VARIANTS])
        ax.set_ylim(0.86, 1.003)
        ax.set_ylabel("Mean cosine similarity")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=9)
    fig.suptitle("Compression robustness across feature models")
    fig.tight_layout()
    fig.savefig(FIGURES / "model_feature_comparison.png", dpi=180)
    plt.close(fig)


def plot_storage_feature_tradeoff(pixel: dict, feature: dict) -> None:
    raw = pixel["source"]["camera_hdf5_storage_bytes"]
    x = np.asarray(
        [100.0 * pixel["variants"][variant]["video_size_bytes"] / raw for variant in VARIANTS]
    )
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    for model_id in MODELS:
        y = np.asarray(
            [
                feature["models"][model_id]["variants"][variant]["cls_cosine"]["mean"]
                for variant in VARIANTS
            ]
        )
        ax.plot(
            x,
            y,
            color=MODEL_COLORS[model_id],
            marker=MODEL_MARKERS[model_id],
            ms=8,
            lw=1.8,
            label=MODEL_NAMES[model_id],
        )
    for index, variant in enumerate(VARIANTS):
        top = max(feature["models"][model]["variants"][variant]["cls_cosine"]["mean"] for model in MODELS)
        offset = (-44, -2) if variant == "crf0_min_loss" else (5, 5)
        ax.annotate(VARIANT_NAMES[variant], (x[index], top), xytext=offset, textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("Video / raw HDF5 top-camera storage (%) [log scale]")
    ax.set_ylabel("Mean CLS cosine similarity")
    ax.set_title("Storage vs model feature fidelity")
    ax.grid(alpha=0.3, which="both")
    ax.set_ylim(0.88, 1.003)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "storage_feature_tradeoff.png", dpi=180)
    plt.close(fig)


def pixel_table(pixel: dict) -> str:
    src = pixel["source"]
    rows = [
        f'| 存储大小 | {mib(src["camera_hdf5_storage_bytes"]):.2f} MiB | '
        + " | ".join(f'{mib(pixel["variants"][v]["video_size_bytes"]):.2f} MiB' for v in VARIANTS)
        + " |",
        "| 占原始头部相机比例 | 100% | "
        + " | ".join(pct(pixel["variants"][v]["ratio_vs_hdf5_camera_storage"]) for v in VARIANTS)
        + " |",
        "| 空间节省 | 0% | "
        + " | ".join(pct(pixel["variants"][v]["space_saving_vs_raw_logical"]) for v in VARIANTS)
        + " |",
        "| 平均 PSNR | ∞ | "
        + " | ".join(f'{pixel["variants"][v]["pixel_metrics"]["psnr_db"]["mean"]:.3f} dB' for v in VARIANTS)
        + " |",
        "| 平均 SSIM | 1 | "
        + " | ".join(f'{pixel["variants"][v]["pixel_metrics"]["ssim"]["mean"]:.6f}' for v in VARIANTS)
        + " |",
        "| 平均 MAE（0–255） | 0 | "
        + " | ".join(f'{pixel["variants"][v]["pixel_metrics"]["mae_255"]["mean"]:.4f}' for v in VARIANTS)
        + " |",
    ]
    return "\n".join(rows)


def model_summary_rows(feature: dict) -> str:
    rows = []
    for model_id in MODELS:
        model = feature["models"][model_id]
        for variant in VARIANTS:
            metrics = model["variants"][variant]
            rows.append(
                f'| {MODEL_NAMES[model_id]} | {VARIANT_NAMES[variant]} | '
                f'{metrics["cls_cosine"]["mean"]:.6f} / {metrics["cls_cosine"]["min"]:.6f} | '
                f'{metrics["patch_token_cosine_mean"]["mean"]:.6f} | '
                f'{metrics["cls_relative_l2"]["mean"]:.4f} | '
                f'{metrics["cls_angle_degrees"]["mean"]:.3f}° |'
            )
    return "\n".join(rows)


def model_spec_rows(feature: dict) -> str:
    rows = []
    for model_id in MODELS:
        model = feature["models"][model_id]
        rows.append(
            f'| {model["display_name"]} | {model["parameter_count"] / 1e6:.1f}M | '
            f'{model["embedding_dimension"]} | {model["patch_size"]} | '
            f'{model["patch_grid"][0]}×{model["patch_grid"][1]} | {model["patch_token_count"]} |'
        )
    return "\n".join(rows)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    pixel = load_json(RESULTS / "pixel_summary.json")
    feature = load_json(RESULTS / "model_feature_summary.json")
    pixel_rows = load_csv(RESULTS / "per_frame_pixel_metrics.csv")
    feature_rows = load_csv(RESULTS / "per_frame_model_feature_metrics.csv")
    plot_storage(pixel)
    plot_pixel_timeseries(pixel_rows)
    plot_model_timeseries(
        feature_rows,
        key="cls_cosine",
        filename="model_cls_by_frame.png",
        ylabel="CLS cosine",
        title="Per-frame CLS feature fidelity",
    )
    plot_model_timeseries(
        feature_rows,
        key="patch_token_cosine_mean",
        filename="model_patch_by_frame.png",
        ylabel="Patch cosine",
        title="Per-frame patch-token feature fidelity",
    )
    plot_model_bars(feature)
    plot_storage_feature_tradeoff(pixel, feature)

    correlations = {}
    for model_id in MODELS:
        correlations[model_id] = {}
        for variant in VARIANTS:
            correlations[model_id][variant] = {
                "ssim_vs_cls_cosine": corr(
                    select_series(pixel_rows, variant=variant, key="ssim"),
                    select_series(feature_rows, model_id=model_id, variant=variant, key="cls_cosine"),
                ),
                "psnr_vs_cls_cosine": corr(
                    select_series(pixel_rows, variant=variant, key="psnr_db"),
                    select_series(feature_rows, model_id=model_id, variant=variant, key="cls_cosine"),
                ),
                "ssim_vs_patch_cosine": corr(
                    select_series(pixel_rows, variant=variant, key="ssim"),
                    select_series(
                        feature_rows,
                        model_id=model_id,
                        variant=variant,
                        key="patch_token_cosine_mean",
                    ),
                ),
            }
    with (RESULTS / "cross_metric_correlations.json").open("w", encoding="utf-8") as f:
        json.dump(correlations, f, indent=2, ensure_ascii=False)

    src = pixel["source"]
    p20 = pixel["variants"]["crf20_balanced"]
    m = feature["models"]
    v3s20 = m["dinov3_vits16"]["variants"]["crf20_balanced"]
    v3b20 = m["dinov3_vitb16"]["variants"]["crf20_balanced"]
    v2s20 = m["dinov2_vits14"]["variants"]["crf20_balanced"]
    v3s50 = m["dinov3_vits16"]["variants"]["crf50_max_loss"]
    v3b50 = m["dinov3_vitb16"]["variants"]["crf50_max_loss"]
    v2s50 = m["dinov2_vits14"]["variants"]["crf50_max_loss"]
    report = f"""# HDF5 单帧、LeRobot AV1 与 DINO 特征横向对比报告

生成时间：{datetime.now().astimezone().isoformat(timespec="seconds")}

## 结论摘要

本测试比较头部相机 `observations/images/top` 的 {src["frame_count"]} 帧 RGB 图像（{src["width"]}×{src["height"]}，30 FPS）。视频采用 LeRobot AV1 设置，测试 CRF 0、20、50；特征模型横向比较 DINOv3 ViT-S/16、DINOv3 ViT-B/16 和 DINOv2 ViT-S/14。

CRF 20 将头部相机数据从 {mib(src["camera_hdf5_storage_bytes"]):.2f} MiB 压到 {mib(p20["video_size_bytes"]):.2f} MiB（约缩小 {src["camera_hdf5_storage_bytes"] / p20["video_size_bytes"]:.1f} 倍），但三个模型的响应不同：

- CLS 保真度：DINOv2-S/14 最高（{v2s20["cls_cosine"]["mean"]:.6f}），其次 DINOv3-S/16（{v3s20["cls_cosine"]["mean"]:.6f}），DINOv3-B/16 最低（{v3b20["cls_cosine"]["mean"]:.6f}）。
- 位置对应 patch-token 保真度：DINOv3-S/16 最高（{v3s20["patch_token_cosine_mean"]["mean"]:.6f}），DINOv3-B/16 为 {v3b20["patch_token_cosine_mean"]["mean"]:.6f}，DINOv2-S/14 为 {v2s20["patch_token_cosine_mean"]["mean"]:.6f}。
- CRF 50 下三者均明显退化；DINOv3-B/16 的 CLS 最敏感（{v3b50["cls_cosine"]["mean"]:.6f}），DINOv2-S/14 的局部 patch 最敏感（{v2s50["patch_token_cosine_mean"]["mean"]:.6f}），DINOv3-S/16 整体最稳（CLS {v3s50["cls_cosine"]["mean"]:.6f}、patch {v3s50["patch_token_cosine_mean"]["mean"]:.6f}）。

因此不存在对所有模型都相同的“安全 CRF”。CRF 20 仍是容量/特征质量较合理的候选，但上线前应使用实际下游模型和策略成功率验证；CRF 50 不建议作为训练主数据。

![模型横向对比](figures/model_feature_comparison.png)

![存储与模型特征权衡](figures/storage_feature_tradeoff.png)

## 存储与像素结果

| 指标 | 原始 HDF5 单帧 | AV1 CRF 0 | AV1 CRF 20 | AV1 CRF 50 |
|---|---:|---:|---:|---:|
{pixel_table(pixel)}

![存储对比](figures/storage_comparison.png)

![逐帧像素指标](figures/pixel_metrics_by_frame.png)

## 特征模型设置

三模型采用同一个确定性预处理：RGB、短边缩放 256、中心裁剪 224×224、bicubic、ImageNet mean/std。

| 模型 | 参数量 | 特征维度 | Patch size | Patch 网格 | Token 数 |
|---|---:|---:|---:|---:|---:|
{model_spec_rows(feature)}

DINOv2 的 patch size 为 14，因此产生 16×16=256 个 token；DINOv3 的 patch size 为 16，产生 14×14=196 个 token。patch 指标只在各模型内部对同一空间位置的原始/压缩特征做比较，不跨模型直接匹配 token。

权重均从本地以 `strict=True` 加载：

- DINOv3 ViT-S/16：`{m["dinov3_vits16"]["checkpoint"]}`
- DINOv3 ViT-B/16：`{m["dinov3_vitb16"]["checkpoint"]}`
- DINOv2 ViT-S/14：`{m["dinov2_vits14"]["checkpoint"]}`

## 三模型 × 三压缩档结果

| 模型 | 视频档位 | CLS cosine 均值 / 最低 | Patch cosine 均值 | CLS relative L2 | CLS 夹角均值 |
|---|---|---:|---:|---:|---:|
{model_summary_rows(feature)}

![逐帧 CLS 特征](figures/model_cls_by_frame.png)

![逐帧 patch 特征](figures/model_patch_by_frame.png)

## 最差 CLS 帧

各模型、各压缩档的原始帧/解码帧/放大差异图：

"""
    for model_id in MODELS:
        report += f"\n- {MODEL_NAMES[model_id]}："
        links = []
        for variant in VARIANTS:
            path = f"figures/worst_cls_{model_id}_{variant}.png"
            frame = feature["models"][model_id]["variants"][variant]["worst_cls_cosine_frame"]
            links.append(f"[{VARIANT_NAMES[variant]}（帧 {frame}）]({path})")
        report += "、".join(links)
    report += f"""

## DINO patch 特征图

完整逐帧 patch-token 张量已经按模型分别保存。每个 NPZ 都包含 `original`、`crf0_min_loss`、`crf20_balanced`、`crf50_max_loss` 四个键，dtype 为 float16：

| 模型 | 每个键的形状 `[帧, token, 维度]` | NPZ 大小 | 文件 |
|---|---:|---:|---|
| DINOv3-S/16 | `{m["dinov3_vits16"]["patch_feature_maps"]["shape_per_key"]}` | {mib(Path(m["dinov3_vits16"]["patch_feature_maps"]["path"]).stat().st_size):.1f} MiB | [下载/打开](results/feature_maps/dinov3_vits16_patch_tokens_float16.npz) |
| DINOv3-B/16 | `{m["dinov3_vitb16"]["patch_feature_maps"]["shape_per_key"]}` | {mib(Path(m["dinov3_vitb16"]["patch_feature_maps"]["path"]).stat().st_size):.1f} MiB | [下载/打开](results/feature_maps/dinov3_vitb16_patch_tokens_float16.npz) |
| DINOv2-S/14 | `{m["dinov2_vits14"]["patch_feature_maps"]["shape_per_key"]}` | {mib(Path(m["dinov2_vits14"]["patch_feature_maps"]["path"]).stat().st_size):.1f} MiB | [下载/打开](results/feature_maps/dinov2_vits14_patch_tokens_float16.npz) |

PCA-RGB 可视化对同一模型、同一帧的原始/三档压缩特征使用共享 PCA 基底及共享颜色范围，因此同一张图内颜色可直接比较。不同模型或不同帧的 PCA 基底不同，不应按绝对颜色跨图比较。

- [DINOv3-S/16 第 110 帧特征图](figures/feature_maps/dinov3_vits16/frame_0110_pca.png)
- [DINOv3-B/16 第 110 帧特征图](figures/feature_maps/dinov3_vitb16/frame_0110_pca.png)
- [DINOv2-S/14 第 110 帧特征图](figures/feature_maps/dinov2_vits14/frame_0110_pca.png)

读取示例：

```python
import numpy as np

data = np.load("results/feature_maps/dinov3_vits16_patch_tokens_float16.npz")
tokens = data["crf20_balanced"]        # [220, 196, 384]
feature_maps = tokens.reshape(220, 14, 14, 384)
```

## 编码设置与实现说明

- HDF5：`{src["path"]}`
- 相机键：`{src["camera_key"]}`；`uint8` RGB，无 HDF5 压缩
- MP4：`libsvtav1`、`yuv420p`、GOP=2、30 FPS、fast-decode=0
- LeRobot 当前 RGB 默认值是 AV1 / yuv420p / GOP 2 / CRF 30 / preset 12；本实验将 CRF 改成 0、20、50。[LeRobot 编码参数文档](https://huggingface.co/docs/lerobot/main/video_encoding_parameters)
- 当前 SVT-AV1 3.1.2 把请求 preset 12 映射为实际 preset 10，三档一致。
- FFmpeg wrapper 会把直接传入的 CRF 0 当作未设置；代码使用 `svtav1-params=crf=0` 直传并由 SVT 日志确认。[FFmpeg wrapper 源码](https://www.ffmpeg.org/doxygen/trunk/libsvtav1_8c_source.html)
- CRF 0 仍不是 RGB 逐像素无损，因为 `yuv420p` 包含 RGB↔YUV 转换和 4:2:0 色度下采样。

## 可复现性与输出

```bash
cd {ROOT}
./run_all.sh --force
```

主要结果：

- `results/per_frame_pixel_metrics.csv`：3×220 条逐帧像素指标
- `results/pixel_summary.json`：像素和存储汇总
- `results/per_frame_model_feature_metrics.csv`：3 模型×3 压缩档×220 帧，共 1980 条特征指标
- `results/model_feature_summary.json`：多模型汇总
- `results/model_global_embeddings.npz`：各模型原始与解码全局特征
- `results/feature_maps/*.npz`：完整逐帧 patch-token 特征图
- `figures/feature_maps/<model>/`：共享 PCA 基底的特征图可视化
- `results/cross_metric_correlations.json`：逐模型的像素/特征相关性

## 局限

- 只有一个 7.33 秒片段和一个头部相机，不能直接外推到其他场景。
- 不同模型的特征空间和归一化统计不同；横向 cosine 数值用于比较各模型受同一压缩扰动的相对漂移，不代表模型能力高低。
- 本测试没有训练具体机器人策略，最终 CRF 应用策略成功率 A/B 测试确定。
- DINOv2 和 DINOv3 的 patch 网格不同，因此只比较各自位置对齐 token 的平均稳定性。
"""
    with (ROOT / "REPORT.md").open("w", encoding="utf-8") as f:
        f.write(report)
    print(f"Wrote {ROOT / 'REPORT.md'}")


if __name__ == "__main__":
    main()
