"""Official LeRobot v3 writer helpers shared by the conversion entry points.

LeRobot 0.6's Dataset v3 API is the only writer here; the RGB/depth encoder
settings that are tuned per dataset (codec, CRF, GOP) travel as an explicit
:class:`RGBVideoConfig` so a recipe can carry them and the manifest can record
exactly what produced the videos.
"""

from __future__ import annotations

import contextlib
import importlib.metadata
import json
import os
import sys
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from robot_data.align.config import AlignedEpisode
from robot_data.errors import ConversionError
from robot_data.progress import progress_bar, progress_enabled
from robot_data.writers.hdf5 import HDFSchema


@dataclass(frozen=True)
class RGBVideoConfig:
    codec: str = "h264"
    pixel_format: str = "yuv420p"
    crf: float = 0
    gop: int = 2
    preset: str | int | None = None
    fast_decode: int = 0
    encoder_threads: int | None = None


# LeRobot v3 concatenates episodes into one mp4 per camera until it reaches
# video_files_size_in_mb (default 200), so an episode's frames start at an
# arbitrary offset inside a shared file and its frame indices no longer match
# the episode's own. 1 is the smallest value the writer accepts and makes it
# roll over to a new file after every episode, giving one mp4 per episode.
VIDEO_FILE_SIZE_MB = 1


def _check_lerobot() -> str:
    try:
        version = importlib.metadata.version("lerobot")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError("Missing dependency: pip install 'lerobot[dataset]>=0.6,<0.7'") from exc
    parts = tuple(int(part) for part in version.split(".")[:2])
    if parts != (0, 6):
        raise RuntimeError(f"These converters target LeRobot 0.6.x, found {version}")
    return version


def parse_preset(value: str | None) -> str | int | None:
    if value is None or value.lower() == "none":
        return None
    return int(value) if value.lstrip("-").isdigit() else value


def build_rgb_encoder(config: RGBVideoConfig) -> Any:
    _check_lerobot()
    from lerobot.configs import RGBEncoderConfig

    return RGBEncoderConfig(
        vcodec=config.codec,
        pix_fmt=config.pixel_format,
        g=config.gop,
        crf=config.crf,
        preset=config.preset,
        fast_decode=config.fast_decode,
    )


def build_depth_encoder(crf: float, encoder_threads: int | None) -> Any:
    _check_lerobot()
    from lerobot.configs import DepthEncoderConfig

    # encoder_threads belongs to the Dataset writer; it is accepted here only
    # to keep RGB/depth CLI configuration symmetric.
    _ = encoder_threads
    return DepthEncoderConfig(crf=crf)


def episode_features(
    episode: AlignedEpisode,
    use_videos: bool,
    include_velocity: bool,
    include_depth: bool,
) -> dict[str, dict[str, Any]]:
    state_names = list(episode.state_names)
    state_dim = episode.state_dim
    if not state_names:
        state_names = [f"state_{index}" for index in range(state_dim)]
    if len(state_names) != state_dim:
        raise ConversionError(
            f"{len(state_names)} state names for {state_dim} state dimensions"
        )
    features: dict[str, dict[str, Any]] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": state_names,
        },
        "action": {"dtype": "float32", "shape": (state_dim,), "names": state_names},
    }
    if include_velocity:
        features["observation.velocity"] = {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": state_names,
        }
    visual_dtype = "video" if use_videos else "image"
    for name, images in episode.images.items():
        features[f"observation.images.{name}"] = {
            "dtype": visual_dtype,
            "shape": tuple(int(value) for value in images.shape[1:]),
            "names": ["height", "width", "channels"],
        }
    if include_depth:
        if not episode.depths:
            raise ConversionError("Depth output requested but aligned episode has no depth data")
        for name, depths in episode.depths.items():
            height, width = depths.shape[1:]
            features[f"observation.depths.{name}"] = {
                "dtype": visual_dtype,
                "shape": (int(height), int(width), 1),
                "names": ["height", "width", "channels"],
                "info": {"is_depth_map": True, "depth_unit": "m"},
            }
    return features


def assert_episode_schema(
    episode: AlignedEpisode,
    features: dict[str, dict[str, Any]],
    include_velocity: bool,
    include_depth: bool,
) -> None:
    expected = tuple(features["observation.state"]["shape"])
    if episode.qpos.shape != (episode.frame_count, *expected):
        raise ConversionError(f"qpos shape invalid: {episode.qpos.shape}, expected (T, {expected[0]})")
    if episode.action.shape != (episode.frame_count, *expected):
        raise ConversionError(f"action shape invalid: {episode.action.shape}, expected (T, {expected[0]})")
    if include_velocity and episode.qvel.shape != (episode.frame_count, *expected):
        raise ConversionError(f"qvel shape invalid: {episode.qvel.shape}, expected (T, {expected[0]})")
    for name, images in episode.images.items():
        expected = tuple(features[f"observation.images.{name}"]["shape"])
        if images.shape[1:] != expected or images.dtype != np.uint8:
            raise ConversionError(
                f"RGB schema differs for {name}: {images.shape[1:]} {images.dtype}, expected {expected} uint8"
            )
    expected_image_names = {
        key.removeprefix("observation.images.")
        for key in features
        if key.startswith("observation.images.")
    }
    if set(episode.images) != expected_image_names:
        raise ConversionError(
            f"Camera set differs: {sorted(episode.images)} != {sorted(expected_image_names)}"
        )
    if include_depth:
        expected_depth_names = {
            key.removeprefix("observation.depths.")
            for key in features
            if key.startswith("observation.depths.")
        }
        if set(episode.depths) != expected_depth_names:
            raise ConversionError("Depth camera set differs between episodes")


def create_dataset(
    repo_id: str,
    root: Path,
    episode: AlignedEpisode,
    robot_type: str | None,
    use_videos: bool,
    include_velocity: bool,
    include_depth: bool,
    rgb_config: RGBVideoConfig,
    depth_crf: float,
    image_writer_processes: int,
    image_writer_threads: int,
) -> tuple[Any, dict[str, dict[str, Any]], str]:
    version = _check_lerobot()
    from lerobot.datasets import LeRobotDataset

    features = episode_features(episode, use_videos, include_velocity, include_depth)
    rgb_encoder = build_rgb_encoder(rgb_config) if use_videos else None
    depth_encoder = build_depth_encoder(depth_crf, rgb_config.encoder_threads) if use_videos and include_depth else None
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=root,
        fps=episode.fps,
        robot_type=robot_type or episode.robot_type,
        features=features,
        use_videos=use_videos,
        image_writer_processes=image_writer_processes,
        image_writer_threads=image_writer_threads,
        rgb_encoder=rgb_encoder,
        depth_encoder=depth_encoder,
        encoder_threads=rgb_config.encoder_threads,
        video_files_size_in_mb=VIDEO_FILE_SIZE_MB,
    )
    return dataset, features, version


def append_episode(
    dataset: Any,
    episode: AlignedEpisode,
    features: dict[str, dict[str, Any]],
    task: str,
    include_velocity: bool,
    include_depth: bool,
) -> None:
    if dataset.fps != episode.fps:
        raise ConversionError(f"FPS differs between episodes: {episode.fps} != {dataset.fps}")
    assert_episode_schema(episode, features, include_velocity, include_depth)
    with progress_bar("write frames", episode.frame_count, unit="frame") as advance:
        for index in range(episode.frame_count):
            advance()
            frame: dict[str, Any] = {
                "observation.state": episode.qpos[index],
                "action": episode.action[index],
                "task": task,
            }
            if include_velocity:
                frame["observation.velocity"] = episode.qvel[index]
            for name, images in episode.images.items():
                frame[f"observation.images.{name}"] = images[index]
            if include_depth:
                for name, depths in episode.depths.items():
                    frame[f"observation.depths.{name}"] = depths[index][..., None]
            dataset.add_frame(frame)
    # save_episode() encodes the videos in one blocking call with no progress
    # hook of its own, so announce it rather than appearing to hang.
    if progress_enabled():
        print(f"  encoding {episode.frame_count} frames...", file=sys.stderr, flush=True)
    dataset.save_episode(parallel_encoding=False)


def write_manifest(
    root: Path,
    *,
    converter: str,
    lerobot_version: str,
    repo_id: str,
    image_storage: str,
    rgb_config: RGBVideoConfig,
    episodes: list[dict[str, Any]],
    extra: dict[str, Any] | None = None,
) -> None:
    manifest = {
        "format": "lerobot-dataset-v3",
        "converter": converter,
        "lerobot_version": lerobot_version,
        "repo_id": repo_id,
        "image_storage": image_storage,
        "rgb_encoder": asdict(rgb_config),
        "video_files_size_in_mb": VIDEO_FILE_SIZE_MB,
        "episodes": episodes,
        **(extra or {}),
    }
    path = root / "meta" / "conversion_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def task_for_source(source: Path, default_task: str, task_map: dict[str, str]) -> str:
    return task_map.get(source.name, task_map.get(str(source), default_task))


def load_task_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in value.items()):
        raise ValueError("task-map must be a JSON object mapping source name/path to task text")
    return value


# ---------------------------------------------------------------------------
# HDF5 source
# ---------------------------------------------------------------------------


def append_hdf5_episode(
    dataset: Any,
    schema: HDFSchema,
    task: str,
    include_velocity: bool,
    include_depth: bool,
) -> None:
    """Stream one HDF5 episode into an open LeRobot dataset."""
    import h5py

    with h5py.File(schema.path, "r") as source:
        with progress_bar("write frames", schema.frames, unit="frame") as advance:
            for index in range(schema.frames):
                advance()
                frame: dict[str, Any] = {
                    "observation.state": np.asarray(
                        source["observations/qpos"][index], dtype=np.float32
                    ),
                    "action": np.asarray(source["action"][index], dtype=np.float32),
                    "task": task,
                }
                if include_velocity:
                    frame["observation.velocity"] = np.asarray(
                        source["observations/qvel"][index], dtype=np.float32
                    )
                for name in schema.cameras:
                    frame[f"observation.images.{name}"] = source[
                        f"observations/images/{name}"
                    ][index]
                if include_depth:
                    for name in schema.depths:
                        depth = np.asarray(
                            source[f"observations/depths/{name}"][index], dtype=np.float32
                        )
                        frame[f"observation.depths.{name}"] = depth[..., None]
                dataset.add_frame(frame)
    dataset.save_episode(parallel_encoding=False)


# ---------------------------------------------------------------------------
# Encoder noise
# ---------------------------------------------------------------------------


def quiet_encoder_logs() -> None:
    """Silence tqdm bars and ffmpeg/libx264 chatter so a progress block stays intact."""
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
