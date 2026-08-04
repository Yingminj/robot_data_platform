#!/usr/bin/env python3
"""Profile-driven alignment and I/O helpers for robot dataset conversion.

The invariant is one LeRobot-style control row::

    observation at tick k -> teleop action produced immediately after it

Raw sensor capture time (ROS ``header.stamp``) is used in ``capture`` mode.
``lerobot-loop`` mode instead uses rosbag record time as the availability time,
which more closely models LeRobot's asynchronous "latest camera frame" reads.

Episode windowing follows teleoperation activity rather than the bag extent:
the usable range runs from the first arm command to the last arm command, and
the grid is anchored to the newest anchor-camera frame at or before that first
command, so row 0 starts on a fresh image at the moment teleoperation begins.
Everything outside that range is discarded.  Gaps *inside* the range are filled
by holding the last issued command (zero-order hold), which is what a real
controller does when commands stop arriving; held rows are counted and can be
rejected via ``max_hold_fraction`` / ``max_hold_run_s``.

State and action dimensions come from the :mod:`robot_profile` description, not
from constants, so grippers and multi-DoF dexterous hands share one code path.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from robot_profile import (
    DEXHAND,
    FLOAT32,
    EndEffectorSpec,
    RobotProfile,
)
from ros_messages import (
    IMAGE_TYPES,
    MessageDecodeError,
    decode_depth,
    decode_image,
    header_stamp_ns_from_cdr,
    image_kind,
    parse_float32_cdr,
    parse_jointcmd_cdr,
    read_jointstate,
    resize_letterbox,
)


class ConversionError(RuntimeError):
    """A source episode cannot safely be converted."""


COMMAND_ECHO = "command_echo"
MEASURED = "measured"


@dataclass(frozen=True)
class AlignmentConfig:
    fps: int = 30
    mode: str = "lerobot-loop"
    image_tolerance_ms: float | None = None
    state_tolerance_ms: float | None = None
    action_tolerance_ms: float | None = None
    action_pair_tolerance_ms: float = 5.0
    end_effector_tolerance_ms: float = 100.0
    image_height: int = 0
    image_width: int = 0
    invalid_frame_policy: str = "fail"
    include_depth: bool = False
    max_decode_errors: int = 0
    action_gap_policy: str = "hold-last-command"
    grid_anchor: str = "anchor-camera-ticks"
    max_hold_fraction: float | None = None
    max_hold_run_s: float | None = None
    max_tick_rate_deviation: float = 0.1

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.max_tick_rate_deviation < 0:
            raise ValueError("max_tick_rate_deviation must be non-negative")
        if self.mode not in {"capture", "lerobot-loop"}:
            raise ValueError("mode must be capture or lerobot-loop")
        if self.invalid_frame_policy not in {"fail", "drop"}:
            raise ValueError("invalid_frame_policy must be fail or drop")
        if self.action_gap_policy not in {"fail", "hold-last-command", "joint-state-fill"}:
            raise ValueError(
                "action_gap_policy must be fail, hold-last-command or joint-state-fill"
            )
        if self.grid_anchor not in {"anchor-camera", "anchor-camera-ticks", "first-command"}:
            raise ValueError(
                "grid_anchor must be anchor-camera, anchor-camera-ticks or first-command"
            )
        tolerances = {
            "image_tolerance_ms": self.image_tolerance_ms,
            "state_tolerance_ms": self.state_tolerance_ms,
            "action_tolerance_ms": self.action_tolerance_ms,
            "action_pair_tolerance_ms": self.action_pair_tolerance_ms,
            "end_effector_tolerance_ms": self.end_effector_tolerance_ms,
        }
        invalid = [name for name, value in tolerances.items() if value is not None and value < 0]
        if invalid:
            raise ValueError(f"Alignment tolerances must be non-negative: {invalid}")
        if self.max_decode_errors < 0:
            raise ValueError("max_decode_errors must be non-negative")
        if bool(self.image_height) != bool(self.image_width):
            raise ValueError("image_height and image_width must both be zero or both be positive")
        if self.max_hold_fraction is not None and not 0 <= self.max_hold_fraction <= 1:
            raise ValueError("max_hold_fraction must be within [0, 1]")

    @property
    def resolved_image_tolerance_ms(self) -> float:
        if self.image_tolerance_ms is not None:
            return self.image_tolerance_ms
        return (500.0 if self.mode == "capture" else 1500.0) / self.fps

    @property
    def resolved_action_tolerance_ms(self) -> float:
        """Do not let a missing command select an action from a later control cycle."""
        return self.action_tolerance_ms if self.action_tolerance_ms is not None else 1000.0 / self.fps

    def resolve_state_tolerance_ms(self, source_period_ms: float) -> float:
        """Bound the age of the newest state sample preceding each tick.

        With "latest sample before tick" semantics that age is naturally spread
        over one source period, so a fixed default that happens to equal the
        publishing period rejects rows for ordinary jitter.  Default to 1.5
        source periods unless the caller pins a value.
        """
        if self.state_tolerance_ms is not None:
            return self.state_tolerance_ms
        return max(1.5 * source_period_ms, 1.0)


@dataclass
class AlignedEpisode:
    source: str
    fps: int
    qpos: np.ndarray
    qvel: np.ndarray
    action: np.ndarray
    images: dict[str, np.ndarray]
    depths: dict[str, np.ndarray]
    timestamps: dict[str, np.ndarray]
    audit: dict[str, Any]
    state_names: tuple[str, ...] = ()
    robot_type: str = "unknown"

    @property
    def frame_count(self) -> int:
        return int(self.action.shape[0])

    @property
    def state_dim(self) -> int:
        return int(self.qpos.shape[1])


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def discover_rosbags(source: Path, recursive: bool = True) -> list[Path]:
    """Find rosbag2 episode directories, for both sqlite3 and MCAP storage."""
    source = source.expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if source.is_file():
        if source.suffix not in {".db3", ".mcap"}:
            raise ValueError(f"Expected a rosbag directory, .db3 or .mcap, got {source}")
        return [source.parent]
    if (source / "metadata.yaml").is_file() or any(source.glob("*.db3")) or any(source.glob("*.mcap")):
        return [source]
    bags: set[Path] = set()
    patterns = ("**/metadata.yaml", "**/*.db3", "**/*.mcap")
    if not recursive:
        patterns = ("*/metadata.yaml", "*/*.db3", "*/*.mcap")
    for pattern in patterns:
        bags.update(path.parent for path in source.glob(pattern))
    return sorted(bags, key=lambda path: str(path))


def discover_hdf5(source: Path, recursive: bool = True) -> list[Path]:
    source = source.expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if source.is_file():
        return [source]
    patterns = ("**/*.hdf5", "**/*.h5") if recursive else ("*.hdf5", "*.h5")
    files: set[Path] = set()
    for pattern in patterns:
        files.update(source.glob(pattern))
    return sorted(files, key=lambda path: str(path))


def sort_and_limit(paths: list[Path], sort_by: str, limit: int) -> list[Path]:
    if sort_by == "mtime":
        paths = sorted(paths, key=lambda path: (path.stat().st_mtime_ns, str(path)))
    elif sort_by == "name":
        paths = sorted(paths, key=lambda path: str(path))
    else:
        raise ValueError("sort_by must be name or mtime")
    return paths[:limit] if limit > 0 else paths


# ---------------------------------------------------------------------------
# Index helpers
# ---------------------------------------------------------------------------


def _nearest_indices(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    right = np.searchsorted(source, target, side="left")
    right = np.clip(right, 0, len(source) - 1)
    left = np.clip(right - 1, 0, len(source) - 1)
    use_right = np.abs(source[right] - target) < np.abs(target - source[left])
    indices = np.where(use_right, right, left)
    return indices, np.abs(source[indices] - target)


def _previous_indices(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.searchsorted(source, target, side="right") - 1
    valid = indices >= 0
    safe = np.clip(indices, 0, len(source) - 1)
    age = target - source[safe]
    return safe, age, valid


def _next_indices(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.searchsorted(source, target, side="left")
    valid = indices < len(source)
    safe = np.clip(indices, 0, len(source) - 1)
    lead = source[safe] - target
    return safe, lead, valid


def _interpolate(
    source_t: np.ndarray, source_v: np.ndarray, target: np.ndarray, tolerance_ns: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    right = np.searchsorted(source_t, target, side="left")
    right_safe = np.clip(right, 0, len(source_t) - 1)
    exact = (right < len(source_t)) & (source_t[right_safe] == target)
    left = right - 1
    valid = exact | ((left >= 0) & (right < len(source_t)))
    left_safe = np.clip(left, 0, len(source_t) - 1)
    left_age = target - source_t[left_safe]
    right_lead = source_t[right_safe] - target
    valid &= exact | ((left_age <= tolerance_ns) & (right_lead <= tolerance_ns))
    denom = (source_t[right_safe] - source_t[left_safe]).astype(np.float64)
    alpha = np.divide(target - source_t[left_safe], denom, out=np.zeros_like(denom), where=denom != 0)
    output = source_v[left_safe] + alpha[:, None] * (source_v[right_safe] - source_v[left_safe])
    output[exact] = source_v[right_safe[exact]]
    selected_time = np.where(exact, source_t[right_safe], target)
    return output, selected_time, valid


def _backward_difference(values: np.ndarray, fps: int) -> np.ndarray:
    """Causal derivative: row k depends only on rows <= k."""
    output = np.zeros_like(values, dtype=np.float64)
    if values.shape[0] > 1:
        output[1:] = (values[1:] - values[:-1]) * float(fps)
        output[0] = output[1]
    return output


def _stats_ms(values_ns: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values_ns, dtype=np.float64) / 1e6
    if not values.size:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "max": None}
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous ``True`` runs as inclusive (start, end) index pairs."""
    if not mask.any():
        return []
    padded = np.concatenate(([False], mask, [False]))
    edges = np.diff(padded.astype(np.int8))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0] - 1
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


# ---------------------------------------------------------------------------
# Bag scanning
# ---------------------------------------------------------------------------


@dataclass
class _Stream:
    """One collected topic: timestamps plus optional per-message values."""

    times: list[int] = field(default_factory=list)
    values: list[np.ndarray] = field(default_factory=list)
    velocities: list[np.ndarray] = field(default_factory=list)
    fallbacks: int = 0
    errors: int = 0
    count: int = 0

    def finish(self, dim: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        times = np.asarray(self.times, dtype=np.int64)
        values = (
            np.asarray(self.values, dtype=np.float64)
            if self.values
            else np.empty((0, dim), dtype=np.float64)
        )
        velocities = (
            np.asarray(self.velocities, dtype=np.float64)
            if self.velocities
            else np.empty((0, dim), dtype=np.float64)
        )
        return times, values, velocities


@dataclass
class _RawBag:
    bag_dir: Path
    profile: RobotProfile
    arm_t: np.ndarray
    arm_pos: np.ndarray
    arm_vel: np.ndarray
    cmd_t: dict[str, np.ndarray]
    cmd_v: dict[str, np.ndarray]
    ee_state_t: dict[str, np.ndarray]
    ee_state_v: dict[str, np.ndarray]
    ee_cmd_t: dict[str, np.ndarray]
    ee_cmd_v: dict[str, np.ndarray]
    image_t: dict[str, np.ndarray]
    depth_t: dict[str, np.ndarray]
    image_kinds: dict[str, str]
    msgtypes: dict[str, str]
    topic_counts: dict[str, int]
    tolerant_parses: dict[str, int]


def open_bag_reader(bag_dir: Path, profile: RobotProfile) -> Any:
    """Open a bag, supplying type definitions the storage backend may lack.

    MCAP embeds its schemas.  rosbag2 sqlite3 (format version 5) does not --
    the ``.db3`` holds type *names* only -- so opening one without a typestore
    fails outright with "Bag contains no type definitions".  The profile's
    ``message_definitions`` fill that gap for custom types; standard ROS 2
    messages come from the bundled Humble typestore.  The default typestore is
    consulted only when the bag has no definitions of its own, so passing it
    never overrides a schema the bag does carry.
    """
    try:
        from rosbags.highlevel import AnyReader
        from rosbags.typesys import Stores, get_types_from_msg, get_typestore
    except ImportError as exc:
        raise RuntimeError("Missing dependency: pip install rosbags") from exc

    typestore = get_typestore(Stores.ROS2_HUMBLE)
    for typename, definition in (profile.message_definitions or {}).items():
        if typename in typestore.types:
            continue
        try:
            typestore.register(get_types_from_msg(definition, typename))
        except Exception as exc:
            raise ConversionError(
                f"profile {profile.name}: cannot register message definition for {typename}: {exc}"
            ) from exc
    return AnyReader([bag_dir], default_typestore=typestore)


def _ensure_monotonic(name: str, timestamps: np.ndarray) -> None:
    if timestamps.size and np.any(np.diff(timestamps) < 0):
        raise ConversionError(f"{name} timestamps are not monotonic")


def _scan_bag(bag_dir: Path, profile: RobotProfile, cfg: AlignmentConfig) -> _RawBag:
    profile = profile.with_depth(cfg.include_depth)
    required = profile.required_topics
    if cfg.include_depth:
        required |= set(profile.depths.values())

    camera_by_topic = {topic: name for name, topic in profile.cameras.items()}
    depth_by_topic = {topic: name for name, topic in profile.depths.items()} if cfg.include_depth else {}
    arm_cmd_topics = set(profile.arm.command_topics)
    ee_by_state = {e.state_topic: e for e in profile.end_effectors if e.state_topic}
    ee_by_command = {e.command_topic: e for e in profile.end_effectors}

    streams: dict[str, _Stream] = {topic: _Stream() for topic in required | set(depth_by_topic)}
    image_kinds: dict[str, str] = {}
    msgtypes: dict[str, str] = {}

    reader = open_bag_reader(bag_dir, profile)
    reader.open()
    try:
        available = {connection.topic for connection in reader.connections}
        missing = sorted(required - available)
        if missing:
            raise ConversionError(f"Missing required topics: {missing}")
        wanted = required | set(depth_by_topic)
        connections = [c for c in reader.connections if c.topic in wanted]
        for connection in connections:
            msgtypes[connection.topic] = connection.msgtype
            if connection.topic in camera_by_topic or connection.topic in depth_by_topic:
                try:
                    image_kinds[connection.topic] = image_kind(connection.msgtype)
                except MessageDecodeError as exc:
                    raise ConversionError(f"{connection.topic}: {exc}") from exc

        for connection, record_ns, rawdata in reader.messages(connections=connections):
            topic = connection.topic
            stream = streams[topic]
            stream.count += 1
            try:
                # Imagery: only timestamps are needed on this pass.
                if topic in camera_by_topic or topic in depth_by_topic:
                    stamp = (
                        header_stamp_ns_from_cdr(rawdata)
                        if cfg.mode == "capture"
                        else int(record_ns)
                    )
                    if stamp is None:
                        raise MessageDecodeError("image has no valid header.stamp")
                    stream.times.append(stamp)
                    continue

                if topic in arm_cmd_topics:
                    header_ns, positions = parse_jointcmd_cdr(rawdata, profile.arm.command_dim)
                    stamp = int(record_ns) if cfg.mode == "lerobot-loop" else header_ns
                    if not np.all(np.isfinite(positions)):
                        raise MessageDecodeError(f"{topic} carries non-finite positions")
                    stream.times.append(stamp)
                    stream.values.append(positions)
                    continue

                if topic == profile.arm.joint_states_topic:
                    message, fallback = read_jointstate(reader, rawdata, connection.msgtype)
                    stream.fallbacks += int(fallback)
                    stamp = int(record_ns) if cfg.mode == "lerobot-loop" else message.stamp_ns
                    expected = (
                        profile.arm.joint_names if profile.arm.joint_state_order == "named" else None
                    )
                    positions = message.ordered(expected, profile.arm.dim)
                    if not np.all(np.isfinite(positions)):
                        raise MessageDecodeError("arm JointState carries non-finite positions")
                    velocity = message.ordered_velocity(expected, profile.arm.dim)
                    stream.times.append(stamp)
                    stream.values.append(positions)
                    stream.velocities.append(
                        velocity if velocity is not None else np.full(profile.arm.dim, np.nan)
                    )
                    continue

                effector = ee_by_state.get(topic) or ee_by_command.get(topic)
                if effector is None:
                    continue
                is_state = topic in ee_by_state and ee_by_state[topic] is effector
                kind = effector.state_kind if is_state else effector.command_kind
                if kind == FLOAT32:
                    # std_msgs/Float32 has no Header; record time is its only clock.
                    stamp = int(record_ns)
                    values = np.asarray([parse_float32_cdr(rawdata)], dtype=np.float64)
                else:
                    message, fallback = read_jointstate(reader, rawdata, connection.msgtype)
                    stream.fallbacks += int(fallback)
                    stamp = int(record_ns) if cfg.mode == "lerobot-loop" else message.stamp_ns
                    expected = effector.joint_names if effector.joint_names and message.names else None
                    values = message.ordered(expected, effector.dim)
                if not np.all(np.isfinite(values)):
                    raise MessageDecodeError(f"{topic} carries non-finite values")
                stream.times.append(stamp)
                stream.values.append(values)
            except Exception:
                stream.errors += 1
    finally:
        reader.close()

    media_topics = set(camera_by_topic) | set(depth_by_topic)
    excessive = {
        topic: stream.errors
        for topic, stream in streams.items()
        if stream.errors > (0 if topic in media_topics else cfg.max_decode_errors)
    }
    if excessive:
        raise ConversionError(f"Decode/validation errors exceed limit: {excessive}")

    empty = sorted(topic for topic in required if not streams[topic].times)
    if empty:
        raise ConversionError(f"Required topics contain no valid messages: {empty}")

    for topic, stream in streams.items():
        _ensure_monotonic(topic, np.asarray(stream.times, dtype=np.int64))

    arm_t, arm_pos, arm_vel = streams[profile.arm.joint_states_topic].finish(profile.arm.dim)
    cmd_t: dict[str, np.ndarray] = {}
    cmd_v: dict[str, np.ndarray] = {}
    for topic in profile.arm.command_topics:
        times, values, _ = streams[topic].finish(profile.arm.command_dim)
        cmd_t[topic], cmd_v[topic] = times, values

    ee_state_t: dict[str, np.ndarray] = {}
    ee_state_v: dict[str, np.ndarray] = {}
    ee_cmd_t: dict[str, np.ndarray] = {}
    ee_cmd_v: dict[str, np.ndarray] = {}
    for effector in profile.end_effectors:
        times, values, _ = streams[effector.command_topic].finish(effector.dim)
        ee_cmd_t[effector.name], ee_cmd_v[effector.name] = times, values
        if effector.state_topic:
            times, values, _ = streams[effector.state_topic].finish(effector.dim)
            ee_state_t[effector.name], ee_state_v[effector.name] = times, values

    return _RawBag(
        bag_dir=bag_dir,
        profile=profile,
        arm_t=arm_t,
        arm_pos=arm_pos,
        arm_vel=arm_vel,
        cmd_t=cmd_t,
        cmd_v=cmd_v,
        ee_state_t=ee_state_t,
        ee_state_v=ee_state_v,
        ee_cmd_t=ee_cmd_t,
        ee_cmd_v=ee_cmd_v,
        image_t={
            name: np.asarray(streams[topic].times, dtype=np.int64)
            for name, topic in profile.cameras.items()
        },
        depth_t={
            name: np.asarray(streams[topic].times, dtype=np.int64)
            for name, topic in profile.depths.items()
        }
        if cfg.include_depth
        else {},
        image_kinds=image_kinds,
        msgtypes=msgtypes,
        topic_counts={topic: stream.count for topic, stream in streams.items()},
        tolerant_parses={
            topic: stream.fallbacks for topic, stream in streams.items() if stream.fallbacks
        },
    )


# ---------------------------------------------------------------------------
# Media decoding
# ---------------------------------------------------------------------------


def _decode_selected_media(
    raw: _RawBag,
    cfg: AlignmentConfig,
    image_indices: dict[str, np.ndarray],
    depth_indices: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    profile = raw.profile
    selected_by_topic: dict[str, set[int]] = {
        profile.cameras[name]: set(np.unique(indices).tolist())
        for name, indices in image_indices.items()
    }
    if cfg.include_depth:
        selected_by_topic.update(
            {
                profile.depths[name]: set(np.unique(indices).tolist())
                for name, indices in depth_indices.items()
            }
        )
    counters = {topic: 0 for topic in selected_by_topic}
    decoded: dict[str, dict[int, np.ndarray]] = {topic: {} for topic in selected_by_topic}
    camera_topics = set(profile.cameras.values())

    reader = open_bag_reader(raw.bag_dir, raw.profile)
    reader.open()
    try:
        connections = [c for c in reader.connections if c.topic in selected_by_topic]
        for connection, _, rawdata in reader.messages(connections=connections):
            topic = connection.topic
            index = counters[topic]
            counters[topic] += 1
            if index not in selected_by_topic[topic]:
                continue
            try:
                message = reader.deserialize(rawdata, connection.msgtype)
                if topic in camera_topics:
                    value = decode_image(message, connection.msgtype)
                    value = resize_letterbox(value, cfg.image_height, cfg.image_width)
                else:
                    value = decode_depth(message, connection.msgtype)
                    value = resize_letterbox(value, cfg.image_height, cfg.image_width, is_depth=True)
                decoded[topic][index] = value
            except Exception as exc:
                raise ConversionError(f"Failed to decode selected {topic} message {index}: {exc}") from exc
    finally:
        reader.close()

    images: dict[str, np.ndarray] = {}
    depths: dict[str, np.ndarray] = {}
    for name, indices in image_indices.items():
        topic = profile.cameras[name]
        missing = sorted(set(indices.tolist()) - set(decoded[topic]))
        if missing:
            raise ConversionError(f"Selected RGB frames missing for {name}: {missing[:10]}")
        images[name] = np.stack([decoded[topic][int(i)] for i in indices]).astype(np.uint8, copy=False)
    for name, indices in depth_indices.items():
        topic = profile.depths[name]
        missing = sorted(set(indices.tolist()) - set(decoded[topic]))
        if missing:
            raise ConversionError(f"Selected depth frames missing for {name}: {missing[:10]}")
        depths[name] = np.stack([decoded[topic][int(i)] for i in indices]).astype(np.float32, copy=False)
    return images, depths


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Window:
    start_ns: int
    end_ns: int
    command_start_ns: int
    command_end_ns: int
    anchor_ns: int
    bag_start_ns: int
    bag_end_ns: int


def _compute_window(raw: _RawBag, cfg: AlignmentConfig) -> _Window:
    """Teleoperation-activity window: first arm command to last arm command."""
    profile = raw.profile
    command_start = max(int(raw.cmd_t[topic][0]) for topic in profile.arm.command_topics)
    command_end = min(int(raw.cmd_t[topic][-1]) for topic in profile.arm.command_topics)
    if command_end <= command_start:
        raise ConversionError("Arm command topics do not overlap in time")

    observation_times: list[np.ndarray] = [raw.arm_t, *raw.image_t.values(), *raw.depth_t.values()]
    observation_times.extend(raw.ee_state_t.values())
    observation_times.extend(raw.ee_cmd_t.values())

    bag_start = min(int(values[0]) for values in [*observation_times, *raw.cmd_t.values()])
    bag_end = max(int(values[-1]) for values in [*observation_times, *raw.cmd_t.values()])

    # Anchor row 0 on the newest anchor-camera frame at or before teleop start,
    # so the episode opens on a fresh image rather than an arbitrary phase.
    anchor_name = profile.resolved_anchor_camera
    anchor_times = raw.image_t[anchor_name]
    if cfg.grid_anchor == "first-command":
        anchor = command_start
    else:
        position = int(np.searchsorted(anchor_times, command_start, side="right")) - 1
        anchor = int(anchor_times[position]) if position >= 0 else command_start

    start = max(anchor, *(int(values[0]) for values in observation_times))
    end = min(command_end, *(int(values[-1]) for values in observation_times))
    if end <= start:
        raise ConversionError(
            "Teleoperation window does not overlap observation topics "
            f"(command span {(command_end - command_start) / 1e9:.2f}s)"
        )
    return _Window(
        start_ns=start,
        end_ns=end,
        command_start_ns=command_start,
        command_end_ns=command_end,
        anchor_ns=anchor,
        bag_start_ns=bag_start,
        bag_end_ns=bag_end,
    )


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------


def _select_command(
    times: np.ndarray, values: np.ndarray, grid_ns: np.ndarray, action_tol: int, policy: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pick the command for each tick, holding the last one across gaps.

    Returns (indices, lead_ns, real_mask, valid_mask).
    """
    next_idx, lead, next_ok = _next_indices(times, grid_ns)
    real = next_ok & (lead <= action_tol)
    if policy == "fail":
        return next_idx, lead, real, real

    previous_idx, _, previous_ok = _previous_indices(times, grid_ns)
    # Inside a gap hold the last issued command; before the first command
    # (possible when the grid is anchored to a camera frame) use the first one.
    indices = np.where(real, next_idx, np.where(previous_ok, previous_idx, 0))
    return indices, lead, real, np.ones_like(real, dtype=bool)


def _effector_observation(
    effector: EndEffectorSpec,
    raw: _RawBag,
    grid_ns: np.ndarray,
    tolerance_ns: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Observed end-effector state; echoes the command when unmeasured."""
    if effector.has_measured_state:
        times = raw.ee_state_t[effector.name]
        values = raw.ee_state_v[effector.name]
        source = MEASURED
    else:
        times = raw.ee_cmd_t[effector.name]
        values = raw.ee_cmd_v[effector.name]
        source = COMMAND_ECHO
    indices, age, valid = _previous_indices(times, grid_ns)
    valid &= age <= tolerance_ns
    return values[indices], times[indices], valid, source


def align_rosbag(bag_dir: Path, profile: RobotProfile, cfg: AlignmentConfig) -> AlignedEpisode:
    """Convert one rosbag into strict, fixed-rate LeRobot-style control rows."""
    raw = _scan_bag(bag_dir.expanduser().resolve(), profile, cfg)
    profile = raw.profile
    window = _compute_window(raw, cfg)

    if cfg.grid_anchor == "anchor-camera-ticks":
        # One row per anchor-camera frame: the observation image is then always
        # exactly fresh instead of up to a full camera period stale, which is
        # what "latest frame before tick" costs when the camera rate and the
        # requested fps are nearly equal.
        anchor_times = raw.image_t[profile.resolved_anchor_camera]
        grid_ns = anchor_times[
            (anchor_times >= window.start_ns) & (anchor_times <= window.end_ns)
        ].astype(np.int64)
        frame_count = int(grid_ns.size)
    else:
        frame_count = int(math.floor((window.end_ns - window.start_ns) * cfg.fps / 1e9)) + 1
        grid_ns = window.start_ns + np.rint(np.arange(frame_count) * 1e9 / cfg.fps).astype(np.int64)
    if frame_count < 2:
        raise ConversionError(f"Teleoperation window produces only {frame_count} frame(s)")

    # With anchor-camera ticks the row spacing is the camera's real period, but
    # LeRobot derives its timestamp column as frame_index / fps.  If the camera
    # does not actually run at --fps the dataset clock silently drifts, so make
    # the mismatch an error rather than a property of the output.
    tick_period_ms = float(np.median(np.diff(grid_ns))) / 1e6 if grid_ns.size > 1 else 0.0
    measured_fps = 1000.0 / tick_period_ms if tick_period_ms > 0 else 0.0
    if cfg.grid_anchor == "anchor-camera-ticks" and measured_fps > 0:
        deviation = abs(measured_fps - cfg.fps) / cfg.fps
        if deviation > cfg.max_tick_rate_deviation:
            raise ConversionError(
                f"anchor camera {profile.resolved_anchor_camera!r} ticks at "
                f"{measured_fps:.2f} Hz but --fps is {cfg.fps} "
                f"({100 * deviation:.1f}% off): the dataset timestamp column would not match "
                f"the real row spacing. Set --fps {round(measured_fps)}, pick another "
                f"--anchor-camera, or use --grid-anchor anchor-camera for a true fixed-rate grid "
                f"(raise --max-tick-rate-deviation to accept the mismatch)."
            )

    arm_period_ms = (
        float(np.median(np.diff(raw.arm_t))) / 1e6 if raw.arm_t.size > 1 else 1000.0 / cfg.fps
    )
    state_tolerance_ms = cfg.resolve_state_tolerance_ms(arm_period_ms)

    image_tol = round(cfg.resolved_image_tolerance_ms * 1e6)
    state_tol = round(state_tolerance_ms * 1e6)
    action_tol = round(cfg.resolved_action_tolerance_ms * 1e6)
    pair_tol = round(cfg.action_pair_tolerance_ms * 1e6)
    effector_tol = round(cfg.end_effector_tolerance_ms * 1e6)

    valid_parts: dict[str, np.ndarray] = {}

    # -- imagery ---------------------------------------------------------
    image_indices: dict[str, np.ndarray] = {}
    image_offsets: dict[str, np.ndarray] = {}
    for name, source_t in raw.image_t.items():
        if cfg.mode == "capture":
            indices, offsets = _nearest_indices(source_t, grid_ns)
            valid = offsets <= image_tol
        else:
            indices, offsets, valid = _previous_indices(source_t, grid_ns)
            valid &= offsets <= image_tol
        image_indices[name] = indices
        image_offsets[name] = offsets
        valid_parts[f"image:{name}"] = valid

    depth_indices: dict[str, np.ndarray] = {}
    depth_offsets: dict[str, np.ndarray] = {}
    for name, source_t in raw.depth_t.items():
        if cfg.mode == "capture":
            indices, offsets = _nearest_indices(source_t, grid_ns)
            valid = offsets <= image_tol
        else:
            indices, offsets, valid = _previous_indices(source_t, grid_ns)
            valid &= offsets <= image_tol
        depth_indices[name] = indices
        depth_offsets[name] = offsets
        valid_parts[f"depth:{name}"] = valid

    # -- arm state -------------------------------------------------------
    if cfg.mode == "capture":
        arm_qpos, arm_state_t, arm_valid = _interpolate(raw.arm_t, raw.arm_pos, grid_ns, state_tol)
        if raw.arm_vel.size and np.all(np.isfinite(raw.arm_vel)):
            arm_qvel, _, velocity_valid = _interpolate(raw.arm_t, raw.arm_vel, grid_ns, state_tol)
            arm_valid &= velocity_valid
        else:
            arm_qvel = np.full_like(arm_qpos, np.nan)
    else:
        arm_idx, arm_age, arm_valid = _previous_indices(raw.arm_t, grid_ns)
        arm_valid &= arm_age <= state_tol
        arm_qpos = raw.arm_pos[arm_idx]
        arm_state_t = raw.arm_t[arm_idx]
        arm_qvel = raw.arm_vel[arm_idx] if raw.arm_vel.size else np.full_like(arm_qpos, np.nan)
    valid_parts["arm_state"] = arm_valid

    # -- end-effector observation ---------------------------------------
    effector_state: dict[str, np.ndarray] = {}
    effector_state_t: dict[str, np.ndarray] = {}
    effector_state_source: dict[str, str] = {}
    for effector in profile.end_effectors:
        values, times, valid, source = _effector_observation(effector, raw, grid_ns, effector_tol)
        effector_state[effector.name] = values
        effector_state_t[effector.name] = times
        effector_state_source[effector.name] = source
        valid_parts[f"state:{effector.name}"] = valid

    # -- arm action ------------------------------------------------------
    cmd_indices: dict[str, np.ndarray] = {}
    cmd_leads: dict[str, np.ndarray] = {}
    real_masks: dict[str, np.ndarray] = {}
    for topic in profile.arm.command_topics:
        indices, lead, real, valid = _select_command(
            raw.cmd_t[topic], raw.cmd_v[topic], grid_ns, action_tol, cfg.action_gap_policy
        )
        cmd_indices[topic] = indices
        cmd_leads[topic] = lead
        real_masks[topic] = real
        valid_parts[f"command:{topic}"] = valid

    selected_cmd_times = np.stack(
        [raw.cmd_t[topic][cmd_indices[topic]] for topic in profile.arm.command_topics]
    )
    command_skew = selected_cmd_times.max(axis=0) - selected_cmd_times.min(axis=0)
    all_real = np.logical_and.reduce([real_masks[t] for t in profile.arm.command_topics])
    any_real = np.logical_or.reduce([real_masks[t] for t in profile.arm.command_topics])
    # Skew only means anything when every arm has a genuine command this tick.
    pair_valid = ~all_real | (command_skew <= pair_tol)
    if cfg.action_gap_policy == "fail":
        valid_parts["command_pair_skew"] = pair_valid
    if cfg.action_gap_policy == "joint-state-fill":
        # Arms are teleoperated independently here, so a row still carries real
        # intent when any arm was commanded; requiring all of them would mask
        # out every row of a single-arm episode.  Per-arm detail stays in
        # audit.hold.joint_state_fill_rows.
        hold_mask = ~any_real
    else:
        hold_mask = ~(all_real & pair_valid)

    if cfg.action_gap_policy == "joint-state-fill":
        # A silent arm is filled from its own measured state, so an episode is
        # usable as long as *some* arm was genuinely teleoperated.
        if not any_real.any():
            raise ConversionError("Episode contains no genuine teleop command rows on any arm")
    elif cfg.action_gap_policy != "fail" and not all_real.any():
        raise ConversionError("Episode contains no genuine teleop command rows")

    # -- end-effector action --------------------------------------------
    reference_time = selected_cmd_times.max(axis=0)
    effector_action: dict[str, np.ndarray] = {}
    effector_action_t: dict[str, np.ndarray] = {}
    effector_action_age: dict[str, np.ndarray] = {}
    for effector in profile.end_effectors:
        times = raw.ee_cmd_t[effector.name]
        values = raw.ee_cmd_v[effector.name]
        indices, age, valid = _previous_indices(times, reference_time)
        valid &= age <= effector_tol
        effector_action[effector.name] = values[indices]
        effector_action_t[effector.name] = times[indices]
        effector_action_age[effector.name] = age
        valid_parts[f"action:{effector.name}"] = valid

    # -- validity --------------------------------------------------------
    valid_mask = np.ones(frame_count, dtype=bool)
    for part in valid_parts.values():
        valid_mask &= part
    invalid_counts = {name: int((~part).sum()) for name, part in valid_parts.items() if not part.all()}
    if not valid_mask.all() and cfg.invalid_frame_policy == "fail":
        raise ConversionError(
            f"{int((~valid_mask).sum())}/{frame_count} control rows violate alignment limits: "
            f"{invalid_counts}"
        )
    if not valid_mask.any():
        raise ConversionError("No valid control rows remain after alignment")

    def keep(values: np.ndarray) -> np.ndarray:
        return values[valid_mask]

    grid_ns = keep(grid_ns)
    arm_qpos = keep(arm_qpos)
    arm_qvel = keep(arm_qvel)
    arm_state_t = keep(arm_state_t)
    command_skew = keep(command_skew)
    hold_mask = keep(hold_mask)
    all_real = keep(all_real)
    any_real = keep(any_real)
    image_indices = {name: keep(indices) for name, indices in image_indices.items()}
    depth_indices = {name: keep(indices) for name, indices in depth_indices.items()}

    # -- assemble state / action ----------------------------------------
    # Each command topic drives a fixed slice of the arm, in declaration order:
    # topic i owns joint_names[i * command_dim : (i + 1) * command_dim].  That
    # mapping is enforced by ArmSpec, so a silent arm can be filled from its own
    # measured joints without guessing which columns belong to it.
    arm_blocks: list[np.ndarray] = []
    joint_state_fill_rows: dict[str, int] = {}
    for position, topic in enumerate(profile.arm.command_topics):
        block = raw.cmd_v[topic][keep(cmd_indices[topic])]
        if cfg.action_gap_policy == "joint-state-fill":
            filled = ~keep(real_masks[topic])
            if filled.any():
                start = position * profile.arm.command_dim
                stop = start + profile.arm.command_dim
                block = block.copy()
                # Older recordings drop joint_cmd while the arm is still held in
                # place by the controller; its measured position is then the
                # only honest statement of the commanded pose.
                block[filled] = arm_qpos[filled, start:stop]
            joint_state_fill_rows[topic] = int(filled.sum())
        arm_blocks.append(block)
    qpos_parts: list[np.ndarray] = [arm_qpos]
    action_parts: list[np.ndarray] = [np.concatenate(arm_blocks, axis=1)]
    for effector in profile.end_effectors:
        qpos_parts.append(keep(effector_state[effector.name]))
        action_parts.append(keep(effector_action[effector.name]))

    qpos = np.concatenate(qpos_parts, axis=1).astype(np.float32)
    action = np.concatenate(action_parts, axis=1).astype(np.float32)
    if qpos.shape[1] != profile.state_dim or action.shape[1] != profile.action_dim:
        raise ConversionError(
            f"Assembled width mismatch: state {qpos.shape[1]} / action {action.shape[1]} "
            f"vs profile {profile.state_dim} / {profile.action_dim}"
        )
    if np.array_equal(qpos, action):
        raise ConversionError("Entire action array equals observation state; refusing unsafe dataset")

    # -- velocity (causal) ----------------------------------------------
    if not np.all(np.isfinite(arm_qvel)):
        arm_qvel = _backward_difference(arm_qpos, cfg.fps)
    velocity_parts = [arm_qvel]
    for effector in profile.end_effectors:
        velocity_parts.append(_backward_difference(keep(effector_state[effector.name]), cfg.fps))
    qvel = np.concatenate(velocity_parts, axis=1).astype(np.float32)

    images, depths = _decode_selected_media(raw, cfg, image_indices, depth_indices)

    # -- hold accounting -------------------------------------------------
    hold_runs = _runs(hold_mask)
    hold_segments = [
        {
            "start_time_s": float(grid_ns[start] - grid_ns[0]) / 1e9,
            "end_time_s": float(grid_ns[end] - grid_ns[0]) / 1e9,
            "duration_s": float(grid_ns[end] - grid_ns[start]) / 1e9,
            "rows": end - start + 1,
        }
        for start, end in hold_runs
    ]
    max_hold_run_s = max((item["duration_s"] for item in hold_segments), default=0.0)
    hold_fraction = float(hold_mask.mean()) if hold_mask.size else 0.0
    if cfg.max_hold_fraction is not None and hold_fraction > cfg.max_hold_fraction:
        raise ConversionError(
            f"held-action fraction {hold_fraction:.3f} exceeds limit {cfg.max_hold_fraction:.3f}"
        )
    if cfg.max_hold_run_s is not None and max_hold_run_s > cfg.max_hold_run_s:
        raise ConversionError(
            f"longest held-action run {max_hold_run_s:.2f}s exceeds limit {cfg.max_hold_run_s:.2f}s"
        )

    timestamps: dict[str, np.ndarray] = {
        "grid_ns": grid_ns,
        "arm_state_ns": arm_state_t,
        "action_hold_mask": hold_mask,
    }
    for topic in profile.arm.command_topics:
        key = topic.strip("/").replace("/", "_")
        timestamps[f"command_{key}_ns"] = raw.cmd_t[topic][keep(cmd_indices[topic])]
    for effector in profile.end_effectors:
        timestamps[f"state_{effector.name}_ns"] = keep(effector_state_t[effector.name])
        timestamps[f"action_{effector.name}_ns"] = keep(effector_action_t[effector.name])
    for name, indices in image_indices.items():
        timestamps[f"image_{name}_ns"] = raw.image_t[name][indices]
    for name, indices in depth_indices.items():
        timestamps[f"depth_{name}_ns"] = raw.depth_t[name][indices]

    def unique_ratio(indices: np.ndarray) -> float:
        return float(len(np.unique(indices)) / len(indices)) if len(indices) else 0.0

    audit: dict[str, Any] = {
        "schema_version": "aligned-control-rows-v2",
        "source_bag": str(raw.bag_dir),
        "profile": profile.to_dict(),
        "alignment_mode": cfg.mode,
        "grid_anchor": cfg.grid_anchor,
        "action_gap_policy": cfg.action_gap_policy,
        "fps": cfg.fps,
        # Actual row spacing: equals fps for fixed-rate grids, and the anchor
        # camera's real rate for anchor-camera-ticks.
        "measured_tick_hz": measured_fps,
        "candidate_frames": frame_count,
        "output_frames": int(qpos.shape[0]),
        "dropped_frames": int(frame_count - qpos.shape[0]),
        "invalid_counts": invalid_counts,
        "window": {
            "bag_duration_s": float(window.bag_end_ns - window.bag_start_ns) / 1e9,
            "command_span_s": float(window.command_end_ns - window.command_start_ns) / 1e9,
            "converted_span_s": float(window.end_ns - window.start_ns) / 1e9,
            "command_coverage": float(window.command_end_ns - window.command_start_ns)
            / float(window.bag_end_ns - window.bag_start_ns),
            "teleop_start_offset_s": float(window.command_start_ns - window.bag_start_ns) / 1e9,
            "anchor_lead_ms": float(window.command_start_ns - window.anchor_ns) / 1e6,
            "anchor_camera": profile.resolved_anchor_camera,
        },
        "hold": {
            "rows": int(hold_mask.sum()),
            "fraction": hold_fraction,
            "max_run_s": max_hold_run_s,
            "segments": hold_segments,
            "real_command_rows": int(all_real.sum()),
            "any_arm_real_command_rows": int(any_real.sum()),
            # Per arm: rows whose action came from measured joints rather than a
            # command.  Those columns are an identity copy of the observation,
            # so training should treat them via action_hold_mask.
            "joint_state_fill_rows": joint_state_fill_rows,
        },
        "unique_ratio": {
            **{f"image_{name}": unique_ratio(indices) for name, indices in image_indices.items()},
            **{
                f"command_{topic.strip('/').replace('/', '_')}": unique_ratio(keep(cmd_indices[topic]))
                for topic in profile.arm.command_topics
            },
        },
        "end_effector_state_source": effector_state_source,
        "image_formats": {
            name: raw.image_kinds.get(topic, "unknown") for name, topic in profile.cameras.items()
        },
        "tolerant_cdr_parses": raw.tolerant_parses,
        "topic_counts": raw.topic_counts,
        "source_period_ms": {"arm_state": arm_period_ms},
        "limits_ms": {
            "image": cfg.resolved_image_tolerance_ms,
            "state": state_tolerance_ms,
            "action": cfg.resolved_action_tolerance_ms,
            "action_pair": cfg.action_pair_tolerance_ms,
            "end_effector": cfg.end_effector_tolerance_ms,
        },
        "metrics_ms": {
            # Lead is only meaningful where a genuine command exists; on held
            # rows it measures the distance to the far side of the gap.
            **{
                f"command_{topic.strip('/').replace('/', '_')}_lead": _stats_ms(
                    keep(cmd_leads[topic])[all_real]
                )
                for topic in profile.arm.command_topics
            },
            "command_pair_skew": _stats_ms(command_skew),
            **{
                f"action_{effector.name}_age": _stats_ms(keep(effector_action_age[effector.name]))
                for effector in profile.end_effectors
            },
            **{
                f"image_{name}_offset": _stats_ms(keep(offsets))
                for name, offsets in image_offsets.items()
            },
            **{
                f"depth_{name}_offset": _stats_ms(keep(offsets))
                for name, offsets in depth_offsets.items()
            },
        },
    }

    return AlignedEpisode(
        source=str(raw.bag_dir),
        fps=cfg.fps,
        qpos=qpos,
        qvel=qvel,
        action=action,
        images=images,
        depths=depths,
        timestamps=timestamps,
        audit=audit,
        state_names=tuple(profile.state_names),
        robot_type=profile.robot_type,
    )


# ---------------------------------------------------------------------------
# HDF5 output
# ---------------------------------------------------------------------------


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
