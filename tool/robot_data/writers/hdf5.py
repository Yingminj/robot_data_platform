"""ACT-style HDF5: writing aligned episodes out, and reading them back in.

Both directions live here because they share one layout contract: what
:func:`write_aligned_hdf5` stores as attributes is exactly what
:func:`inspect_hdf5` reads back to rebuild an episode's schema.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from robot_data.align.config import AlignedEpisode
from robot_data.errors import ConversionError
from robot_data.profiles.schema import RobotProfile

HDF5_COMPRESSIONS = ("none", "gzip", "lzf")


def hdf5_compression_kwargs(compression: str, level: int) -> dict[str, Any]:
    if compression == "none":
        return {}
    if compression == "gzip":
        return {"compression": "gzip", "compression_opts": level, "shuffle": True}
    if compression == "lzf":
        return {"compression": "lzf", "shuffle": True}
    raise ValueError("HDF5 compression must be none, gzip, or lzf")


def write_aligned_hdf5(
    path: Path,
    episode: AlignedEpisode,
    profile: RobotProfile,
    compression: str = "gzip",
    compression_level: int = 4,
) -> None:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("Missing dependency: pip install h5py") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = hdf5_compression_kwargs(compression, compression_level)
    with h5py.File(path, "w") as output:
        output.attrs["sim"] = False
        output.attrs["schema_version"] = episode.audit["schema_version"]
        output.attrs["fps"] = episode.fps
        output.attrs["source_bag"] = episode.source
        output.attrs["alignment_mode"] = episode.audit["alignment_mode"]
        output.attrs["robot_type"] = episode.robot_type
        output.attrs["state_dim"] = episode.state_dim
        output.attrs["state_names_json"] = json.dumps(list(episode.state_names))
        output.attrs["profile_json"] = json.dumps(profile.to_dict(), sort_keys=True)
        output.attrs["alignment_audit_json"] = json.dumps(episode.audit, sort_keys=True)
        output.create_dataset("action", data=episode.action, **kwargs)
        observations = output.create_group("observations")
        observations.create_dataset("qpos", data=episode.qpos, **kwargs)
        observations.create_dataset("qvel", data=episode.qvel, **kwargs)
        images = observations.create_group("images")
        for name, values in episode.images.items():
            images.create_dataset(name, data=values, chunks=(1, *values.shape[1:]), **kwargs)
        if episode.depths:
            depths = observations.create_group("depths")
            depths.attrs["unit"] = "meter"
            for name, values in episode.depths.items():
                depths.create_dataset(name, data=values, chunks=(1, *values.shape[1:]), **kwargs)
        timestamp_group = output.create_group("timestamps")
        timestamp_group.attrs["unit"] = "nanosecond"
        for name, values in episode.timestamps.items():
            dtype = "bool" if values.dtype == bool else "int64"
            timestamp_group.create_dataset(name, data=values, dtype=dtype)


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


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


def json_value(value: Any) -> Any:
    """Coerce an HDF5 attribute into something ``json.dumps`` accepts."""
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
                "convert with 'rdp export-hdf5' or pass --allow-unaligned-source"
            )
        raw_names = source.attrs.get("state_names_json")
        if raw_names is not None:
            state_names = tuple(json.loads(json_value(raw_names)))
            if len(state_names) != state_dim:
                raise ConversionError(
                    f"{path}: {len(state_names)} state names for {state_dim} dimensions"
                )
        else:
            state_names = tuple(f"state_{index}" for index in range(state_dim))
        robot_type = str(json_value(source.attrs.get("robot_type", "unknown")))
        # action == qpos means the episode carries no command signal (a_t = s_t).
        # It is reported, not rejected; pass --strict-action to fail on it instead.
        action_equals_state = bool(
            np.array_equal(source["action"][:], source["observations/qpos"][:])
        )
        if action_equals_state and strict_action:
            raise ConversionError(
                f"{path}: entire action array equals qpos; refusing unsafe dataset"
            )
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
                raise ConversionError(
                    f"{path}: depth requested but observations/depths is missing"
                )
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
        attrs = {key: json_value(value) for key, value in source.attrs.items()}
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
    """A one-frame episode carrying only the schema, to create the dataset."""
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
