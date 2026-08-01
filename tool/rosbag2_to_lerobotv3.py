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
    DEFAULT_CAMERA_TOPICS,
    DEFAULT_DEPTH_TOPICS,
    TopicConfig,
    align_rosbag,
    discover_rosbags,
    parse_name_topic,
    sort_and_limit,
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
    parser.add_argument("--robot-type", default="marvin")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--alignment-mode", choices=("capture", "lerobot-loop"), default="lerobot-loop"
    )
    parser.add_argument("--image-tolerance-ms", type=float, default=None)
    parser.add_argument("--state-tolerance-ms", type=float, default=10.0)
    parser.add_argument(
        "--action-tolerance-ms", type=float, default=None, help="默认一帧周期；30 FPS 时为 33.33 ms"
    )
    parser.add_argument("--action-pair-tolerance-ms", type=float, default=5.0)
    parser.add_argument("--gripper-tolerance-ms", type=float, default=100.0)
    parser.add_argument("--invalid-frame-policy", choices=("fail", "drop"), default="fail")
    parser.add_argument("--max-decode-errors", type=int, default=0)
    parser.add_argument("--image-height", type=int, default=0)
    parser.add_argument("--image-width", type=int, default=0)
    parser.add_argument("--joint-state-order", choices=("named", "first14"), default="named")
    parser.add_argument("--camera", action="append", metavar="NAME=TOPIC")
    parser.add_argument("--include-depth", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--depth", action="append", metavar="NAME=TOPIC")
    parser.add_argument("--joint-state-topic", default="/joint_states")
    parser.add_argument("--joint-cmd-a-topic", default="/control/joint_cmd_A")
    parser.add_argument("--joint-cmd-b-topic", default="/control/joint_cmd_B")
    parser.add_argument("--gripper-l-topic", default="/control/gripperValueL")
    parser.add_argument("--gripper-r-topic", default="/control/gripperValueR")
    parser.add_argument("--image-storage", choices=("video", "image"), default="video")
    parser.add_argument("--video-codec", default="h264")
    parser.add_argument("--video-pixel-format", default="yuv420p")
    parser.add_argument("--crf", type=float, default=0)
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
                episode = align_rosbag(bag, topics, cfg)
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
            extra={"alignment": vars(cfg), "topics": vars(topics)},
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
