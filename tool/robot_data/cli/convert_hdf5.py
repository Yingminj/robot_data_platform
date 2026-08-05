"""``rdp convert-hdf5`` -- aligned/legacy ACT HDF5 to a LeRobot v3 dataset."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from robot_data.cli.args import (
    add_batch_args,
    add_dataset_args,
    add_recipe_args,
    add_video_args,
    resolve_recipe,
    resolve_video,
)
from robot_data.discovery import discover_hdf5, sort_and_limit
from robot_data.errors import ConversionError
from robot_data.progress import EpisodeProgress, format_duration
from robot_data.writers.hdf5 import HDFSchema, assert_same_schema, inspect_hdf5, schema_stub
from robot_data.writers.lerobot_v3 import (
    append_hdf5_episode,
    create_dataset,
    load_task_map,
    quiet_encoder_logs,
    suppressed_stderr,
    task_for_source,
    write_manifest,
)

HELP = "用 LeRobot 0.6 官方 v3 writer 批量转换 ACT HDF5"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--allow-fps-override", action="store_true")
    parser.add_argument(
        "--allow-unaligned-source",
        action="store_true",
        help="允许转换缺少 schema_version 属性（对齐来源未知）的 HDF5",
    )
    parser.add_argument(
        "--strict-action",
        action="store_true",
        help="action 与 qpos 完全相同时直接报错（默认仅告警并继续转换）",
    )
    parser.add_argument(
        "--include-depth", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--verbose-encoder",
        action="store_true",
        help="保留 ffmpeg/libx264 与 tqdm 的原始日志（默认屏蔽以免打断进度显示）",
    )
    # HDF5 sources carry their own alignment and topic layout, so only the video
    # half of a recipe applies here.
    add_recipe_args(parser)
    add_video_args(parser)
    add_dataset_args(parser)
    add_batch_args(parser)


def run(args: argparse.Namespace) -> int:
    output = args.output.expanduser().resolve()
    staging = output.with_name(f".{output.name}.incomplete-{os.getpid()}")
    dataset = None
    progress: EpisodeProgress | None = None
    try:
        if args.fps <= 0:
            raise ValueError("fps must be positive")
        if output.exists():
            if not args.overwrite:
                raise FileExistsError(f"Output exists: {output}")
            shutil.rmtree(output)
        if staging.exists():
            raise FileExistsError(f"Staging directory exists: {staging}")
        paths = sort_and_limit(discover_hdf5(args.input, args.recursive), args.sort_by, args.limit)
        if not paths:
            raise ConversionError(f"No HDF5 files found under {args.input}")
        task_map = load_task_map(args.task_map)
        recipe = resolve_recipe(args)
        rgb_config = resolve_video(args, recipe)
        repo_id = args.repo_id or f"local/{output.name}"
        reference: HDFSchema | None = None
        features = None
        lerobot_version = None
        manifest_episodes: list[dict[str, Any]] = []
        quiet = not args.verbose_encoder
        if quiet:
            quiet_encoder_logs()
        progress = EpisodeProgress(args.progress, len(paths), args.input, output)
        progress.start()
        for path in paths:
            progress.set_current(path)
            try:
                schema = inspect_hdf5(
                    path,
                    requested_fps=args.fps,
                    allow_fps_override=args.allow_fps_override,
                    include_velocity=args.include_velocity,
                    include_depth=args.include_depth,
                    strict_action=args.strict_action,
                    allow_unaligned_source=args.allow_unaligned_source,
                )
                if schema.action_equals_state:
                    progress.log(
                        f"  warning: {path.name}: entire action array equals qpos "
                        "(no command signal; policy would learn a_t = s_t)"
                    )
                if reference is None:
                    reference = schema
                    with suppressed_stderr(quiet):
                        dataset, features, lerobot_version = create_dataset(
                            repo_id=repo_id,
                            root=staging,
                            episode=schema_stub(schema),
                            robot_type=args.robot_type,
                            use_videos=args.image_storage == "video",
                            include_velocity=args.include_velocity,
                            include_depth=args.include_depth,
                            rgb_config=rgb_config,
                            depth_crf=args.depth_crf,
                            image_writer_processes=args.image_writer_processes,
                            image_writer_threads=args.image_writer_threads,
                        )
                else:
                    assert_same_schema(reference, schema)
                task = task_for_source(path, args.task, task_map)
                with suppressed_stderr(quiet):
                    append_hdf5_episode(
                        dataset, schema, task, args.include_velocity, args.include_depth
                    )
                progress.episode_done(schema.frames, schema.action_equals_state)
                manifest_episodes.append(
                    {
                        "source": str(path),
                        "frames": schema.frames,
                        "fps": schema.fps,
                        "task": task,
                        "source_attributes": schema.attrs,
                        "action_equals_state": schema.action_equals_state,
                        "status": "converted",
                    }
                )
            except Exception as exc:
                if dataset is not None:
                    try:
                        dataset.clear_episode_buffer(delete_images=True)
                    except Exception:
                        pass
                manifest_episodes.append(
                    {"source": str(path), "status": "failed", "error": str(exc)}
                )
                progress.episode_failed()
                progress.log(f"  failed: {exc}")
                if args.on_error == "fail":
                    progress.finish()
                    raise
        if dataset is None:
            progress.finish()
            raise ConversionError("No episode was converted")
        with suppressed_stderr(quiet):
            dataset.finalize()
        progress.finish()
        degenerate = [
            item["source"] for item in manifest_episodes if item.get("action_equals_state")
        ]
        write_manifest(
            staging,
            converter="rdp convert-hdf5",
            lerobot_version=lerobot_version,
            repo_id=repo_id,
            image_storage=args.image_storage,
            rgb_config=rgb_config,
            episodes=manifest_episodes,
            extra={
                "recipe": recipe.to_dict() if recipe else None,
                "depth_crf": args.depth_crf,
                "fps": args.fps,
                "strict_action": args.strict_action,
                "allow_unaligned_source": args.allow_unaligned_source,
                "action_equals_state_episodes": degenerate,
            },
        )
        info = json.loads((staging / "meta" / "info.json").read_text(encoding="utf-8"))
        expected_frames = sum(
            int(item.get("frames", 0))
            for item in manifest_episodes
            if item["status"] == "converted"
        )
        if info.get("total_frames") != expected_frames:
            raise ConversionError(
                f"LeRobot metadata frame count mismatch: "
                f"{info.get('total_frames')} != {expected_frames}"
            )
        staging.rename(output)
        print(
            f"converted {info['total_episodes']} episode(s), {info['total_frames']} frames "
            f"in {format_duration(progress.elapsed)}: {output}"
        )
        if degenerate:
            print(
                f"warning: {len(degenerate)}/{len(manifest_episodes)} episode(s) have "
                "action == observation.state; see "
                "meta/conversion_manifest.json -> action_equals_state_episodes",
                file=sys.stderr,
            )
        return 0
    except Exception as exc:
        if progress is not None:
            progress.finish()
        if dataset is not None:
            try:
                dataset.writer.stop_image_writer()
            except Exception:
                pass
        print(f"error: {exc}", file=sys.stderr)
        if staging.exists():
            print(f"incomplete output retained for diagnostics: {staging}", file=sys.stderr)
        return 2
