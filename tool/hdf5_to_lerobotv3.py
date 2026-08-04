#!/usr/bin/env python3
"""Batch-convert aligned/legacy ACT HDF5 episodes with LeRobot's official v3 API."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

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
    state_dim: int
    state_names: tuple[str, ...]
    robot_type: str
    cameras: dict[str, tuple[int, int, int]]
    depths: dict[str, tuple[int, int]]
    attrs: dict[str, Any]
    action_equals_state: bool = False


def _format_duration(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    return f"{minutes}m{secs:02d}s"


def quiet_encoder_logs() -> None:
    """Silence tqdm bars and ffmpeg/libx264 chatter so the progress block stays intact."""
    os.environ.setdefault("TQDM_DISABLE", "1")
    with contextlib.suppress(Exception):
        from datasets.utils.logging import disable_progress_bar

        disable_progress_bar()
    with contextlib.suppress(Exception):
        import av.logging

        av.logging.set_level(av.logging.FATAL)


@contextlib.contextmanager
def suppressed_stderr(active: bool) -> Iterator[None]:
    """Redirect fd 2 to /dev/null; catches native ffmpeg/x264 writes, not just Python ones."""
    if not active:
        yield
        return
    sys.stderr.flush()
    saved = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(devnull)
        os.close(saved)


class ProgressReporter:
    """Single redrawing block on a TTY, one line per episode otherwise."""

    LINES = 4

    def __init__(self, mode: str, total: int, source: Path, output: Path) -> None:
        if mode == "auto":
            mode = "bar" if sys.stdout.isatty() else "plain"
        self.mode = mode
        self.total = total
        self.title = f"Converting {source.name} -> {output.name}"
        self.started = time.monotonic()
        self.done = 0
        self.frames = 0
        self.degenerate = 0
        self.failed = 0
        self.current = ""
        self.drawn = False

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def start(self) -> None:
        if self.mode == "bar":
            self._draw()  # the block carries its own title line
        elif self.mode != "none":
            print(self.title, flush=True)

    def set_current(self, path: Path) -> None:
        self.current = path.name
        if self.mode == "bar":
            self._draw()

    def episode_done(self, frames: int, degenerate: bool) -> None:
        self.done += 1
        self.frames += frames
        self.degenerate += int(degenerate)
        self._render_after_episode()

    def episode_failed(self) -> None:
        self.done += 1
        self.failed += 1
        self._render_after_episode()

    def log(self, message: str) -> None:
        """Emit a message without leaving a torn progress block behind."""
        if self.mode == "bar" and self.drawn:
            self._clear()
        print(message, file=sys.stderr, flush=True)
        if self.mode == "bar":
            self._draw()

    def finish(self) -> None:
        """Leave the final block on screen; safe to call more than once."""
        if self.mode == "bar" and self.drawn:
            self._draw()
            self.drawn = False
            print(flush=True)

    def _render_after_episode(self) -> None:
        if self.mode == "bar":
            self._draw()
        elif self.mode == "plain":
            percent = 100 * self.done / self.total if self.total else 0.0
            print(
                f"[{self.done}/{self.total}] {percent:5.1f}%  frames {self.frames:,}  "
                f"elapsed {_format_duration(self.elapsed)}  ETA {_format_duration(self._eta())}  {self.current}",
                flush=True,
            )

    def _eta(self) -> float:
        if self.done <= 0 or self.done >= self.total:
            return 0.0
        return self.elapsed / self.done * (self.total - self.done)

    def _clear(self) -> None:
        sys.stdout.write(f"\033[{self.LINES}A\033[J")
        sys.stdout.flush()
        self.drawn = False

    def _draw(self) -> None:
        if self.drawn:
            sys.stdout.write(f"\033[{self.LINES}A")
        ratio = self.done / self.total if self.total else 0.0
        width = 28
        filled = int(round(width * ratio))
        bar = "#" * filled + "." * (width - filled)
        eta = "--" if self.done == 0 else _format_duration(self._eta())
        status = f"current: {self.current}" if self.current else "current: -"
        if self.degenerate:
            status += f"  |  action==state: {self.degenerate} episode(s)"
        if self.failed:
            status += f"  |  failed: {self.failed}"
        columns = shutil.get_terminal_size((100, 24)).columns
        lines = [
            self.title,
            f"[{bar}]  {self.done}/{self.total} episodes  {100 * ratio:3.0f}%",
            f"frames {self.frames:,} | elapsed {_format_duration(self.elapsed)} | ETA {eta}",
            status,
        ]
        for line in lines:
            sys.stdout.write("\033[2K" + line[: max(columns - 1, 20)] + "\n")
        sys.stdout.flush()
        self.drawn = True


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
    strict_action: bool = False,
    allow_unaligned_source: bool = False,
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
        # The state width comes from the file, so grippers (1 DoF per side) and
        # dexterous hands (20 DoF per side) share this path unchanged.
        if frames < 1 or source["action"].ndim != 2:
            raise ConversionError(f"{path}: action must be (T,D), got {source['action'].shape}")
        state_dim = int(source["action"].shape[1])
        if source["observations/qpos"].shape != (frames, state_dim):
            raise ConversionError(
                f"{path}: qpos shape must be {(frames, state_dim)}, "
                f"got {source['observations/qpos'].shape}"
            )
        if include_velocity and source["observations/qvel"].shape != (frames, state_dim):
            raise ConversionError(
                f"{path}: qvel shape must be {(frames, state_dim)}, "
                f"got {source['observations/qvel'].shape}"
            )
        schema_version = source.attrs.get("schema_version")
        if schema_version is None and not allow_unaligned_source:
            raise ConversionError(
                f"{path}: no schema_version attribute, so alignment provenance is unknown; "
                "convert with rosbag2_to_hdf5_aligned.py or pass --allow-unaligned-source"
            )
        raw_names = source.attrs.get("state_names_json")
        if raw_names is not None:
            state_names = tuple(json.loads(_json_value(raw_names)))
            if len(state_names) != state_dim:
                raise ConversionError(
                    f"{path}: {len(state_names)} state names for {state_dim} dimensions"
                )
        else:
            state_names = tuple(f"state_{index}" for index in range(state_dim))
        robot_type = str(_json_value(source.attrs.get("robot_type", "unknown")))
        # action == qpos means the episode carries no command signal (a_t = s_t).
        # It is reported, not rejected; pass --strict-action to fail on it instead.
        action_equals_state = bool(np.array_equal(source["action"][:], source["observations/qpos"][:]))
        if action_equals_state and strict_action:
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
    return HDFSchema(
        path=path,
        frames=frames,
        fps=requested_fps,
        state_dim=state_dim,
        state_names=state_names,
        robot_type=robot_type,
        cameras=cameras,
        depths=depths,
        attrs=attrs,
        action_equals_state=action_equals_state,
    )


def schema_stub(schema: HDFSchema) -> AlignedEpisode:
    return AlignedEpisode(
        source=str(schema.path),
        fps=schema.fps,
        qpos=np.empty((1, schema.state_dim), dtype=np.float32),
        qvel=np.empty((1, schema.state_dim), dtype=np.float32),
        action=np.empty((1, schema.state_dim), dtype=np.float32),
        images={
            name: np.empty((1, *shape), dtype=np.uint8) for name, shape in schema.cameras.items()
        },
        depths={
            name: np.empty((1, *shape), dtype=np.float32) for name, shape in schema.depths.items()
        },
        timestamps={},
        audit={},
        state_names=schema.state_names,
        robot_type=schema.robot_type,
    )


def assert_same_schema(reference: HDFSchema, current: HDFSchema) -> None:
    if reference.fps != current.fps:
        raise ConversionError(f"FPS differs: {current.path}")
    if reference.state_dim != current.state_dim:
        raise ConversionError(
            f"state dimension differs: {current.state_dim} != {reference.state_dim} ({current.path})"
        )
    if reference.state_names != current.state_names:
        raise ConversionError(f"state names differ: {current.path}")
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
    parser.add_argument("--robot-type", default=None, help="默认取 HDF5 的 robot_type 属性")
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
    parser.add_argument("--image-storage", choices=("video", "image"), default="video")
    parser.add_argument(
        "--video-codec",
        default="h264",
        help="与 LeRobot 默认一致；CRF 取值范围随编码器不同（AV1 0-63，x264 0-51）",
    )
    parser.add_argument("--video-pixel-format", default="yuv420p")
    parser.add_argument("--crf", type=float, default=20, help="0 表示无损；见 test_lerobot/REPORT.md")
    parser.add_argument("--gop", type=int, default=2)
    parser.add_argument("--preset", default=None)
    parser.add_argument("--fast-decode", type=int, default=0)
    parser.add_argument("--encoder-threads", type=int, default=None)
    parser.add_argument("--include-depth", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--depth-crf", type=float, default=0)
    parser.add_argument("--include-velocity", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--image-writer-processes", type=int, default=0)
    parser.add_argument("--image-writer-threads", type=int, default=8)
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sort-by", choices=("name", "mtime"), default="name")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--on-error", choices=("fail", "skip"), default="fail")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--progress",
        choices=("auto", "bar", "plain", "none"),
        default="auto",
        help="auto=TTY 时用进度条，重定向时每 episode 一行；bar/plain/none 可强制",
    )
    parser.add_argument(
        "--verbose-encoder",
        action="store_true",
        help="保留 ffmpeg/libx264 与 tqdm 的原始日志（默认屏蔽以免打断进度显示）",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.expanduser().resolve()
    staging = output.with_name(f".{output.name}.incomplete-{os.getpid()}")
    dataset = None
    progress: ProgressReporter | None = None
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
        quiet = not args.verbose_encoder
        if quiet:
            quiet_encoder_logs()
        progress = ProgressReporter(args.progress, len(paths), args.input, output)
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
                    append_hdf5_episode(dataset, schema, task, args.include_velocity, args.include_depth)
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
                manifest_episodes.append({"source": str(path), "status": "failed", "error": str(exc)})
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
            converter=Path(__file__),
            lerobot_version=lerobot_version,
            repo_id=repo_id,
            image_storage=args.image_storage,
            rgb_config=rgb_config,
            episodes=manifest_episodes,
            extra={
                "depth_crf": args.depth_crf,
                "fps": args.fps,
                "strict_action": args.strict_action,
                "allow_unaligned_source": args.allow_unaligned_source,
                "action_equals_state_episodes": degenerate,
            },
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
        print(
            f"converted {info['total_episodes']} episode(s), {info['total_frames']} frames "
            f"in {_format_duration(progress.elapsed)}: {output}"
        )
        if degenerate:
            print(
                f"warning: {len(degenerate)}/{len(manifest_episodes)} episode(s) have action == observation.state; "
                "see meta/conversion_manifest.json -> action_equals_state_episodes",
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


if __name__ == "__main__":
    raise SystemExit(main())
