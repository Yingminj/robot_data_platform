#!/usr/bin/env python3
"""Batch-convert rosbag2 episodes to timestamp-audited ACT-style HDF5."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from conversion_common import (
    AlignmentConfig,
    ConversionError,
    align_rosbag,
    discover_rosbags,
    sort_and_limit,
    write_aligned_hdf5,
)
from robot_profile import (
    BUILTIN_PROFILES,
    DEFAULT_PROFILE,
    apply_topic_overrides,
    load_profile,
    parse_name_topic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="严格按固定控制频率对齐 rosbag2，并批量写入带源时间戳的 ACT HDF5。"
    )
    parser.add_argument("--input", type=Path, required=True, help="单个 rosbag 目录或包含多个 bag 的目录")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--output-name-template",
        default="{name}.hdf5",
        help="支持 {name} 和从 0 开始的 {index}，例如 episode_{index:06d}.hdf5",
    )
    parser.add_argument(
        "--profile",
        default=DEFAULT_PROFILE,
        help=f"机器人配置：内置 {sorted(BUILTIN_PROFILES)} 或 JSON 文件路径",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--alignment-mode",
        choices=("capture", "lerobot-loop"),
        default="lerobot-loop",
        help="lerobot-loop（默认）=按 bag 到达时间取 tick 之前最新帧，因果；"
        "capture=按 header.stamp 物理采集时间，会用到 tick 之后的数据，仅用于诊断",
    )
    parser.add_argument("--image-tolerance-ms", type=float, default=None, help="默认半帧")
    parser.add_argument(
        "--state-tolerance-ms", type=float, default=None, help="默认为观测到的 joint_states 周期的 1.5 倍"
    )
    parser.add_argument(
        "--action-tolerance-ms", type=float, default=None, help="默认一帧周期；30 FPS 时为 33.33 ms"
    )
    parser.add_argument("--action-pair-tolerance-ms", type=float, default=5.0)
    parser.add_argument("--end-effector-tolerance-ms", type=float, default=100.0)
    parser.add_argument("--invalid-frame-policy", choices=("fail", "drop"), default="fail")
    parser.add_argument(
        "--action-gap-policy",
        choices=("fail", "hold-last-command", "joint-state-fill"),
        default="hold-last-command",
        help="hold-last-command=断档时保持最后一条 joint_cmd（默认）；"
        "joint-state-fill=断档时按该手臂实测 joint_states 填充（旧版 .db3 遥操作会整段静默）；"
        "fail=拒绝该 episode",
    )
    parser.add_argument(
        "--grid-anchor",
        choices=("anchor-camera", "anchor-camera-ticks", "first-command"),
        default="anchor-camera-ticks",
        help="anchor-camera-ticks（默认）=直接以锚点相机帧时刻为 tick，图像陈旧度恒为 0；"
        "anchor-camera=从首条 joint_cmd 之前最近的锚点相机帧起按 1/fps 取 tick；"
        "first-command=从首条 joint_cmd 起按 1/fps 取 tick",
    )
    parser.add_argument("--max-hold-fraction", type=float, default=None)
    parser.add_argument("--max-hold-run-s", type=float, default=None)
    parser.add_argument("--max-tick-rate-deviation", type=float, default=0.1,
                        help="anchor-camera-ticks 下实测 tick 频率与 --fps 的最大相对偏差，默认 0.1")
    parser.add_argument("--max-decode-errors", type=int, default=0)
    parser.add_argument("--image-height", type=int, default=0, help="0 表示保留原高度")
    parser.add_argument("--image-width", type=int, default=0, help="0 表示保留原宽度")
    parser.add_argument(
        "--camera",
        action="append",
        metavar="NAME=TOPIC",
        help="可重复；一旦指定即替换 profile 中的相机映射",
    )
    parser.add_argument("--include-depth", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--depth", action="append", metavar="NAME=TOPIC")
    parser.add_argument("--anchor-camera", default=None)
    parser.add_argument("--compression", choices=("none", "gzip", "lzf"), default="gzip")
    parser.add_argument("--compression-level", type=int, default=4)
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sort-by", choices=("name", "mtime"), default="name")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--on-error", choices=("fail", "skip"), default="fail")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        profile = apply_topic_overrides(
            load_profile(args.profile),
            cameras=parse_name_topic(args.camera),
            depths=parse_name_topic(args.depth),
            anchor_camera=args.anchor_camera,
        )
        cfg = AlignmentConfig(
            fps=args.fps,
            mode=args.alignment_mode,
            image_tolerance_ms=args.image_tolerance_ms,
            state_tolerance_ms=args.state_tolerance_ms,
            action_tolerance_ms=args.action_tolerance_ms,
            action_pair_tolerance_ms=args.action_pair_tolerance_ms,
            end_effector_tolerance_ms=args.end_effector_tolerance_ms,
            image_height=args.image_height,
            image_width=args.image_width,
            invalid_frame_policy=args.invalid_frame_policy,
            include_depth=args.include_depth,
            max_decode_errors=args.max_decode_errors,
            action_gap_policy=args.action_gap_policy,
            grid_anchor=args.grid_anchor,
            max_hold_fraction=args.max_hold_fraction,
            max_hold_run_s=args.max_hold_run_s,
            max_tick_rate_deviation=args.max_tick_rate_deviation,
        )
        bags = sort_and_limit(discover_rosbags(args.input, args.recursive), args.sort_by, args.limit)
        if not bags:
            raise ConversionError(f"No rosbag2 episodes found under {args.input}")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        results: list[dict[str, object]] = []
        started = time.monotonic()
        for index, bag in enumerate(bags, 1):
            filename = args.output_name_template.format(name=bag.name, index=index - 1)
            if Path(filename).name != filename or not filename.endswith((".hdf5", ".h5")):
                raise ValueError(f"Invalid output filename from template: {filename!r}")
            output = args.output_dir / filename
            staging = output.with_name(f".{output.name}.incomplete")
            print(f"[{index}/{len(bags)}] {bag} -> {output}", flush=True)
            try:
                if output.exists():
                    if not args.overwrite:
                        raise FileExistsError(f"Output exists: {output}")
                    output.unlink()
                if staging.exists():
                    staging.unlink()
                episode = align_rosbag(bag, profile, cfg)
                write_aligned_hdf5(
                    staging,
                    episode,
                    profile,
                    compression=args.compression,
                    compression_level=args.compression_level,
                )
                staging.rename(output)
                results.append(
                    {"source": str(bag), "output": str(output), "status": "converted", **episode.audit}
                )
                print(f"  converted: {episode.frame_count} frames", flush=True)
            except Exception as exc:
                if staging.exists():
                    staging.unlink()
                results.append({"source": str(bag), "status": "failed", "error": str(exc)})
                print(f"  failed: {exc}", file=sys.stderr, flush=True)
                if args.on_error == "fail":
                    raise
        summary = {
            "converter": str(Path(__file__).resolve()),
            "elapsed_s": time.monotonic() - started,
            "episodes": results,
        }
        (args.output_dir / "conversion_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return 0 if any(item["status"] == "converted" for item in results) else 2
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
