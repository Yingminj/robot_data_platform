"""``rdp convert`` -- rosbag2 episodes straight to a LeRobot v3 dataset."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from robot_data.align import align_rosbag
from robot_data.cli.args import (
    add_alignment_args,
    add_batch_args,
    add_dataset_args,
    add_profile_args,
    add_recipe_args,
    add_video_args,
    describe_selection,
    resolve_alignment,
    resolve_include_depth,
    resolve_profile,
    resolve_recipe,
    resolve_video,
)
from robot_data.discovery import discover_rosbags, sort_and_limit
from robot_data.errors import ConversionError
from robot_data.progress import set_progress_mode
from robot_data.writers.lerobot_v3 import (
    append_episode,
    create_dataset,
    load_task_map,
    task_for_source,
    write_manifest,
)

HELP = "把 rosbag2 episode 批量转换为 LeRobotDataset v3"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", type=Path, required=True, help="单个 rosbag 目录或其父目录")
    parser.add_argument("--output", type=Path, required=True)
    add_recipe_args(parser)
    add_profile_args(parser)
    add_alignment_args(parser)
    add_video_args(parser)
    add_dataset_args(parser)
    add_batch_args(parser)


def run(args: argparse.Namespace) -> int:
    set_progress_mode(args.progress)
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

        recipe = resolve_recipe(args)
        profile = resolve_profile(args, recipe)
        include_depth = resolve_include_depth(args, recipe)
        args.include_depth = include_depth
        cfg = resolve_alignment(args, recipe)
        rgb_config = resolve_video(args, recipe)
        print(describe_selection(profile, recipe, cfg), flush=True)

        bags = sort_and_limit(
            discover_rosbags(args.input, args.recursive), args.sort_by, args.limit
        )
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
                        robot_type=args.robot_type or (recipe.robot_type if recipe else None),
                        use_videos=args.image_storage == "video",
                        include_velocity=args.include_velocity,
                        include_depth=include_depth,
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
                    include_depth=include_depth,
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
                manifest_episodes.append(
                    {"source": str(bag), "status": "failed", "error": str(exc)}
                )
                print(f"  failed: {exc}", file=sys.stderr, flush=True)
                if args.on_error == "fail":
                    raise
        if dataset is None:
            raise ConversionError("No episode was converted")
        dataset.finalize()
        write_manifest(
            staging,
            converter="rdp convert",
            lerobot_version=lerobot_version,
            repo_id=repo_id,
            image_storage=args.image_storage,
            rgb_config=rgb_config,
            episodes=manifest_episodes,
            extra={
                "recipe": recipe.to_dict() if recipe else None,
                "alignment": vars(cfg),
                "profile": profile.to_dict(),
            },
        )
        staging.rename(output)
        converted = sum(item["status"] == "converted" for item in manifest_episodes)
        print(f"converted {converted} episode(s): {output}")
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
