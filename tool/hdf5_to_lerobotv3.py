#!/usr/bin/env python3
"""Batch-convert aligned/legacy ACT HDF5 episodes with LeRobot's official v3 API."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from conversion_common import AlignedEpisode, ConversionError, discover_hdf5, sort_and_limit
from lerobot_v3_common import (
    RGBVideoConfig,
    create_dataset,
    load_task_map,
    parse_preset,
    task_for_source,
    write_manifest,
)


@dataclass(frozen=True)
class HDFSchema:
    path: Path
    frames: int
    fps: int
    cameras: dict[str, tuple[int, int, int]]
    depths: dict[str, tuple[int, int]]
    attrs: dict[str, Any]


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def inspect_hdf5(
    path: Path,
    requested_fps: int,
    allow_fps_override: bool,
    include_velocity: bool,
    include_depth: bool,
) -> HDFSchema:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("Missing dependency: pip install h5py") from exc
    with h5py.File(path, "r") as source:
        required = ["action", "observations/qpos", "observations/images"]
        if include_velocity:
            required.append("observations/qvel")
        missing = [name for name in required if name not in source]
        if missing:
            raise ConversionError(f"{path}: missing HDF5 fields {missing}")
        frames = int(source["action"].shape[0])
        if frames < 1 or source["action"].shape != (frames, 16):
            raise ConversionError(f"{path}: action shape must be (T,16), got {source['action'].shape}")
        if source["observations/qpos"].shape != (frames, 16):
            raise ConversionError(f"{path}: qpos shape must be (T,16)")
        if include_velocity and source["observations/qvel"].shape != (frames, 16):
            raise ConversionError(f"{path}: qvel shape must be (T,16)")
        if np.array_equal(source["action"][:], source["observations/qpos"][:]):
            raise ConversionError(f"{path}: entire action array equals qpos; refusing unsafe dataset")
        cameras: dict[str, tuple[int, int, int]] = {}
        for name, dataset in source["observations/images"].items():
            if dataset.ndim != 4 or dataset.shape[0] != frames or dataset.shape[-1] != 3:
                raise ConversionError(f"{path}: invalid RGB dataset {name}: {dataset.shape}")
            if dataset.dtype != np.uint8:
                raise ConversionError(f"{path}: RGB {name} must be uint8, got {dataset.dtype}")
            cameras[name] = tuple(int(value) for value in dataset.shape[1:])
        if not cameras:
            raise ConversionError(f"{path}: no RGB cameras")
        depths: dict[str, tuple[int, int]] = {}
        if include_depth:
            if "observations/depths" not in source:
                raise ConversionError(f"{path}: depth requested but observations/depths is missing")
            for name, dataset in source["observations/depths"].items():
                if dataset.ndim != 3 or dataset.shape[0] != frames:
                    raise ConversionError(f"{path}: invalid depth dataset {name}: {dataset.shape}")
                depths[name] = tuple(int(value) for value in dataset.shape[1:])
        source_fps = int(source.attrs.get("fps", requested_fps))
        if source_fps != requested_fps and not allow_fps_override:
            raise ConversionError(
                f"{path}: HDF5 fps={source_fps}, requested fps={requested_fps}; "
                "use --allow-fps-override only if intentional"
            )
        attrs = {key: _json_value(value) for key, value in source.attrs.items()}
    return HDFSchema(path, frames, requested_fps, cameras, depths, attrs)


def schema_stub(schema: HDFSchema) -> AlignedEpisode:
    return AlignedEpisode(
        source=str(schema.path),
        fps=schema.fps,
        qpos=np.empty((1, 16), dtype=np.float32),
        qvel=np.empty((1, 16), dtype=np.float32),
        action=np.empty((1, 16), dtype=np.float32),
        images={
            name: np.empty((1, *shape), dtype=np.uint8) for name, shape in schema.cameras.items()
        },
        depths={
            name: np.empty((1, *shape), dtype=np.float32) for name, shape in schema.depths.items()
        },
        timestamps={},
        audit={},
    )


def assert_same_schema(reference: HDFSchema, current: HDFSchema) -> None:
    if reference.fps != current.fps:
        raise ConversionError(f"FPS differs: {current.path}")
    if reference.cameras != current.cameras:
        raise ConversionError(f"RGB camera schema differs: {current.path}")
    if reference.depths != current.depths:
        raise ConversionError(f"Depth camera schema differs: {current.path}")


def append_hdf5_episode(
    dataset: Any,
    schema: HDFSchema,
    task: str,
    include_velocity: bool,
    include_depth: bool,
) -> None:
    import h5py

    with h5py.File(schema.path, "r") as source:
        for index in range(schema.frames):
            frame: dict[str, Any] = {
                "observation.state": np.asarray(source["observations/qpos"][index], dtype=np.float32),
                "action": np.asarray(source["action"][index], dtype=np.float32),
                "task": task,
            }
            if include_velocity:
                frame["observation.velocity"] = np.asarray(
                    source["observations/qvel"][index], dtype=np.float32
                )
            for name in schema.cameras:
                frame[f"observation.images.{name}"] = source[f"observations/images/{name}"][index]
            if include_depth:
                for name in schema.depths:
                    depth = np.asarray(source[f"observations/depths/{name}"][index], dtype=np.float32)
                    frame[f"observation.depths.{name}"] = depth[..., None]
            dataset.add_frame(frame)
    dataset.save_episode(parallel_encoding=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用 LeRobot 0.6 官方 Dataset v3 writer 批量转换 ACT HDF5。"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--task", default="Unspecified task")
    parser.add_argument("--task-map", type=Path, default=None)
    parser.add_argument("--robot-type", default="marvin")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--allow-fps-override", action="store_true")
    parser.add_argument("--image-storage", choices=("video", "image"), default="video")
    parser.add_argument("--video-codec", default="h264")
    parser.add_argument("--video-pixel-format", default="yuv420p")
    parser.add_argument("--crf", type=float, default=0, help="RGB 视频 CRF，默认 0")
    parser.add_argument("--gop", type=int, default=2)
    parser.add_argument("--preset", default=None)
    parser.add_argument("--fast-decode", type=int, default=0)
    parser.add_argument("--encoder-threads", type=int, default=None)
    parser.add_argument("--include-depth", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--depth-crf", type=float, default=0)
    parser.add_argument("--include-velocity", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--image-writer-processes", type=int, default=0)
    parser.add_argument("--image-writer-threads", type=int, default=0)
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
        rgb_config = RGBVideoConfig(
            codec=args.video_codec,
            pixel_format=args.video_pixel_format,
            crf=args.crf,
            gop=args.gop,
            preset=parse_preset(args.preset),
            fast_decode=args.fast_decode,
            encoder_threads=args.encoder_threads,
        )
        repo_id = args.repo_id or f"local/{output.name}"
        reference: HDFSchema | None = None
        features = None
        lerobot_version = None
        manifest_episodes: list[dict[str, Any]] = []
        for index, path in enumerate(paths, 1):
            print(f"[{index}/{len(paths)}] {path}", flush=True)
            try:
                schema = inspect_hdf5(
                    path,
                    requested_fps=args.fps,
                    allow_fps_override=args.allow_fps_override,
                    include_velocity=args.include_velocity,
                    include_depth=args.include_depth,
                )
                if reference is None:
                    reference = schema
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
                append_hdf5_episode(dataset, schema, task, args.include_velocity, args.include_depth)
                manifest_episodes.append(
                    {
                        "source": str(path),
                        "frames": schema.frames,
                        "fps": schema.fps,
                        "task": task,
                        "source_attributes": schema.attrs,
                        "status": "converted",
                    }
                )
            except Exception as exc:
                if dataset is not None:
                    try:
                        dataset.clear_episode_buffer(delete_images=True)
                    except Exception:
                        pass
                manifest_episodes.append({"source": str(path), "status": "failed", "error": str(exc)})
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
            extra={"depth_crf": args.depth_crf, "fps": args.fps},
        )
        info = json.loads((staging / "meta" / "info.json").read_text(encoding="utf-8"))
        expected_frames = sum(
            int(item.get("frames", 0)) for item in manifest_episodes if item["status"] == "converted"
        )
        if info.get("total_frames") != expected_frames:
            raise ConversionError(
                f"LeRobot metadata frame count mismatch: {info.get('total_frames')} != {expected_frames}"
            )
        staging.rename(output)
        print(f"converted {info['total_episodes']} episode(s), {info['total_frames']} frames: {output}")
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
