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
    DEFAULT_CAMERA_TOPICS,
    DEFAULT_DEPTH_TOPICS,
    TopicConfig,
    align_rosbag,
    discover_rosbags,
    parse_name_topic,
    sort_and_limit,
    write_aligned_hdf5,
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
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--alignment-mode",
        choices=("capture", "lerobot-loop"),
        default="capture",
        help="capture=按 header.stamp 物理采集时间；lerobot-loop=按 bag 到达时间模拟最新帧读取",
    )
    parser.add_argument("--image-tolerance-ms", type=float, default=None, help="默认半帧")
    parser.add_argument("--state-tolerance-ms", type=float, default=10.0)
    parser.add_argument(
        "--action-tolerance-ms", type=float, default=None, help="默认一帧周期；30 FPS 时为 33.33 ms"
    )
    parser.add_argument("--action-pair-tolerance-ms", type=float, default=5.0)
    parser.add_argument("--gripper-tolerance-ms", type=float, default=100.0)
    parser.add_argument("--invalid-frame-policy", choices=("fail", "drop"), default="fail")
    parser.add_argument(
        "--action-gap-policy",
        choices=("fail", "hold"),
        default="fail",
        help="fail=拒绝缺失 joint_cmd 的行（默认）；hold=用下一 tick 的 qpos 补齐",
    )
    parser.add_argument(
        "--hold-fill-leading",
        action="store_true",
        help="在 hold 模式下，窗口计算时忽略 joint_cmd 时间范围，允许补齐前导/尾随断档",
    )
    parser.add_argument("--max-decode-errors", type=int, default=0)
    parser.add_argument("--image-height", type=int, default=0, help="0 表示保留原高度")
    parser.add_argument("--image-width", type=int, default=0, help="0 表示保留原宽度")
    parser.add_argument("--joint-state-order", choices=("named", "first14"), default="named")
    parser.add_argument(
        "--camera",
        action="append",
        metavar="NAME=TOPIC",
        help="可重复；一旦指定即替换默认三相机映射",
    )
    parser.add_argument("--include-depth", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--depth", action="append", metavar="NAME=TOPIC")
    parser.add_argument("--joint-state-topic", default="/joint_states")
    parser.add_argument("--joint-cmd-a-topic", default="/control/joint_cmd_A")
    parser.add_argument("--joint-cmd-b-topic", default="/control/joint_cmd_B")
    parser.add_argument("--gripper-l-topic", default="/control/gripperValueL")
    parser.add_argument("--gripper-r-topic", default="/control/gripperValueR")
    parser.add_argument("--compression", choices=("none", "gzip", "lzf"), default="none")
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
        cameras = parse_name_topic(args.camera, DEFAULT_CAMERA_TOPICS)
        depths = parse_name_topic(args.depth, DEFAULT_DEPTH_TOPICS)
        topics = TopicConfig(
            cameras=cameras,
            depths=depths,
            joint_states=args.joint_state_topic,
            joint_cmd_a=args.joint_cmd_a_topic,
            joint_cmd_b=args.joint_cmd_b_topic,
            gripper_l=args.gripper_l_topic,
            gripper_r=args.gripper_r_topic,
        )
        cfg = AlignmentConfig(
            fps=args.fps,
            mode=args.alignment_mode,
            image_tolerance_ms=args.image_tolerance_ms,
            state_tolerance_ms=args.state_tolerance_ms,
            action_tolerance_ms=args.action_tolerance_ms,
            action_pair_tolerance_ms=args.action_pair_tolerance_ms,
            gripper_tolerance_ms=args.gripper_tolerance_ms,
            image_height=args.image_height,
            image_width=args.image_width,
            joint_state_order=args.joint_state_order,
            invalid_frame_policy=args.invalid_frame_policy,
            include_depth=args.include_depth,
            max_decode_errors=args.max_decode_errors,
            action_gap_policy=args.action_gap_policy,
            hold_fill_leading=args.hold_fill_leading,
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
                episode = align_rosbag(bag, topics, cfg)
                write_aligned_hdf5(
                    staging,
                    episode,
                    topics,
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
