#!/usr/bin/env python3
"""Batch-convert rosbag2 episodes directly with LeRobot 0.6's v3 writer."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from conversion_common import (
    AlignmentConfig,
    ConversionError,
    align_rosbag,
    discover_rosbags,
    sort_and_limit,
)
from robot_profile import (
    BUILTIN_PROFILES,
    DEFAULT_PROFILE,
    apply_topic_overrides,
    load_profile,
    parse_name_topic,
)
from lerobot_v3_common import (
    RGBVideoConfig,
    append_episode,
    create_dataset,
    load_task_map,
    parse_preset,
    task_for_source,
    write_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将一个或多个 rosbag2 episode 直接转换为 LeRobotDataset v3。"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", default=None, help="默认 local/<output目录名>")
    parser.add_argument("--task", default="Unspecified task")
    parser.add_argument("--task-map", type=Path, default=None, help="JSON: bag名称或路径 -> task")
    parser.add_argument(
        "--profile",
        default=DEFAULT_PROFILE,
        help=f"机器人配置：内置 {sorted(BUILTIN_PROFILES)} 或 JSON 文件路径",
    )
    parser.add_argument("--robot-type", default=None, help="覆盖 profile 中的 robot_type")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--alignment-mode", choices=("capture", "lerobot-loop"), default="lerobot-loop"
    )
    parser.add_argument("--image-tolerance-ms", type=float, default=None)
    parser.add_argument(
        "--state-tolerance-ms", type=float, default=None, help="默认为观测到的 joint_states 周期的 1.5 倍"
    )
    parser.add_argument(
        "--action-tolerance-ms", type=float, default=None, help="默认一帧周期；30 FPS 时为 33.33 ms"
    )
    parser.add_argument("--action-pair-tolerance-ms", type=float, default=5.0)
    parser.add_argument(
        "--end-effector-tolerance-ms",
        type=float,
        default=100.0,
        help="夹爪/灵巧手状态与命令的最大时间偏差",
    )
    parser.add_argument("--invalid-frame-policy", choices=("fail", "drop"), default="fail")
    parser.add_argument(
        "--action-gap-policy",
        choices=("fail", "hold-last-command"),
        default="hold-last-command",
        help="hold-last-command=断档时保持最后一条 joint_cmd（默认）；fail=拒绝该 episode",
    )
    parser.add_argument(
        "--grid-anchor",
        choices=("anchor-camera", "anchor-camera-ticks", "first-command"),
        default="anchor-camera",
        help="第 0 帧对齐到首条 joint_cmd 之前最近的锚点相机帧（默认），或直接对齐到首条 joint_cmd",
    )
    parser.add_argument(
        "--max-hold-fraction",
        type=float,
        default=None,
        help="保持动作的行数占比上限，超出则拒绝该 episode",
    )
    parser.add_argument(
        "--max-hold-run-s",
        type=float,
        default=None,
        help="单段保持动作的最长时长（秒），超出则拒绝该 episode",
    )
    parser.add_argument("--max-decode-errors", type=int, default=0)
    parser.add_argument("--image-height", type=int, default=0)
    parser.add_argument("--image-width", type=int, default=0)
    parser.add_argument("--camera", action="append", metavar="NAME=TOPIC", help="覆盖 profile 相机映射")
    parser.add_argument("--include-depth", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--depth", action="append", metavar="NAME=TOPIC")
    parser.add_argument("--anchor-camera", default=None, help="用作网格锚点的相机名，默认取 profile 设置")
    parser.add_argument("--image-storage", choices=("video", "image"), default="video")
    parser.add_argument(
        "--video-codec",
        default="libsvtav1",
        help="与 LeRobot 默认一致；CRF 取值范围随编码器不同（AV1 0-63，x264 0-51）",
    )
    parser.add_argument("--video-pixel-format", default="yuv420p")
    parser.add_argument("--crf", type=float, default=0, help="0 表示无损；见 test_lerobot/REPORT.md")
    parser.add_argument("--gop", type=int, default=2)
    parser.add_argument("--preset", default=None)
    parser.add_argument("--fast-decode", type=int, default=0)
    parser.add_argument("--encoder-threads", type=int, default=None)
    parser.add_argument("--depth-crf", type=float, default=0)
    parser.add_argument("--image-writer-processes", type=int, default=0)
    parser.add_argument("--image-writer-threads", type=int, default=0)
    parser.add_argument("--include-velocity", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sort-by", choices=("name", "mtime"), default="name")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--on-error", choices=("fail", "skip"), default="fail")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.expanduser().resolve()
    staging = output.with_name(f".{output.name}.incomplete-{os.getpid()}")
    dataset = None
    try:
        if output.exists():
            if not args.overwrite:
                raise FileExistsError(f"Output exists: {output}")
            shutil.rmtree(output)
        if staging.exists():
            raise FileExistsError(f"Staging directory exists: {staging}")
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
        )
        print(
            f"profile={profile.name} state_dim={profile.state_dim} "
            f"cameras={sorted(profile.cameras)} anchor={profile.resolved_anchor_camera}",
            flush=True,
        )
        rgb_config = RGBVideoConfig(
            codec=args.video_codec,
            pixel_format=args.video_pixel_format,
            crf=args.crf,
            gop=args.gop,
            preset=parse_preset(args.preset),
            fast_decode=args.fast_decode,
            encoder_threads=args.encoder_threads,
        )
        bags = sort_and_limit(discover_rosbags(args.input, args.recursive), args.sort_by, args.limit)
        if not bags:
            raise ConversionError(f"No rosbag2 episodes found under {args.input}")
        task_map = load_task_map(args.task_map)
        repo_id = args.repo_id or f"local/{output.name}"
        features = None
        lerobot_version = None
        manifest_episodes: list[dict[str, object]] = []
        for index, bag in enumerate(bags, 1):
            print(f"[{index}/{len(bags)}] aligning {bag}", flush=True)
            try:
                episode = align_rosbag(bag, profile, cfg)
                if dataset is None:
                    dataset, features, lerobot_version = create_dataset(
                        repo_id=repo_id,
                        root=staging,
                        episode=episode,
                        robot_type=args.robot_type,
                        use_videos=args.image_storage == "video",
                        include_velocity=args.include_velocity,
                        include_depth=args.include_depth,
                        rgb_config=rgb_config,
                        depth_crf=args.depth_crf,
                        image_writer_processes=args.image_writer_processes,
                        image_writer_threads=args.image_writer_threads,
                    )
                task = task_for_source(bag, args.task, task_map)
                append_episode(
                    dataset,
                    episode,
                    features,
                    task,
                    include_velocity=args.include_velocity,
                    include_depth=args.include_depth,
                )
                manifest_episodes.append(
                    {"source": str(bag), "task": task, "status": "converted", **episode.audit}
                )
                print(f"  saved episode with {episode.frame_count} frames", flush=True)
            except Exception as exc:
                if dataset is not None:
                    try:
                        dataset.clear_episode_buffer(delete_images=True)
                    except Exception:
                        pass
                manifest_episodes.append({"source": str(bag), "status": "failed", "error": str(exc)})
                print(f"  failed: {exc}", file=sys.stderr, flush=True)
                if args.on_error == "fail":
                    raise
        if dataset is None:
            raise ConversionError("No episode was converted")
        dataset.finalize()
        write_manifest(
            staging,
            converter=Path(__file__),
            lerobot_version=lerobot_version,
            repo_id=repo_id,
            image_storage=args.image_storage,
            rgb_config=rgb_config,
            episodes=manifest_episodes,
            extra={"alignment": vars(cfg), "profile": profile.to_dict()},
        )
        staging.rename(output)
        print(f"converted {sum(item['status'] == 'converted' for item in manifest_episodes)} episode(s): {output}")
        return 0
    except Exception as exc:
        if dataset is not None:
            try:
                dataset.writer.stop_image_writer()
            except Exception:
                pass
        print(f"error: {exc}", file=sys.stderr)
        if staging.exists():
            print(f"incomplete output retained for diagnostics: {staging}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
