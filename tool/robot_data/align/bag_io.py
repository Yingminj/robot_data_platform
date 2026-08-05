"""Opening rosbags and reading them into flat per-topic arrays.

This is the lowest layer that knows about rosbag2: everything above it works on
numpy arrays of timestamps and values.  The quality checker imports from here
too, so inspecting a bag never drags in the alignment machinery.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from robot_data.align.config import AlignmentConfig
from robot_data.errors import ConversionError, MessageDecodeError
from robot_data.profiles.schema import (
    FLOAT32,
    FLOAT32MULTIARRAY,
    EndEffectorSpec,
    RobotProfile,
)
from robot_data.progress import connection_total, tracked
from robot_data.ros.cdr import (
    header_stamp_ns_from_cdr,
    parse_float32_cdr,
    parse_float32multiarray_cdr,
    parse_jointcmd_cdr,
    read_jointstate,
)
from robot_data.ros.media import image_kind

# How a topic named by the profile failed to deliver: not advertised by the
# recording at all, or advertised but carrying no valid message.
TOPIC_ABSENT = "absent"
TOPIC_EMPTY = "empty"


def open_bag_reader(bag_dir: Path, profile: RobotProfile | None = None) -> Any:
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
    definitions = (profile.message_definitions if profile else None) or {}
    for typename, definition in definitions.items():
        if typename in typestore.types:
            continue
        try:
            typestore.register(get_types_from_msg(definition, typename))
        except Exception as exc:
            name = profile.name if profile else "?"
            raise ConversionError(
                f"profile {name}: cannot register message definition for {typename}: {exc}"
            ) from exc
    return AnyReader([bag_dir], default_typestore=typestore)


# ---------------------------------------------------------------------------
# Timestamp-only scan (quality checking and topic inventory)
# ---------------------------------------------------------------------------


@dataclass
class TopicScan:
    """Timestamps for one topic; payloads are never decoded."""

    topic: str
    msgtype: str
    receive_ns: list[int] = field(default_factory=list)
    header_ns: list[int | None] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.receive_ns)


@dataclass
class BagScan:
    bag_dir: Path
    topics: dict[str, TopicScan]
    start_ns: int
    end_ns: int

    @property
    def duration_ns(self) -> int:
        return max(self.end_ns - self.start_ns, 0)


def scan_timestamps(
    bag_dir: Path,
    profile: RobotProfile | None,
    header_topics: set[str] | None = None,
    only_topics: set[str] | None = None,
) -> BagScan:
    """Collect ``(receive_ns, header_ns)`` per topic without decoding payloads.

    ``only_topics`` restricts the read; passing ``None`` reads every topic in
    the bag, which is what the topic inventory needs to show streams the profile
    does not claim.  ``header.stamp`` is parsed only for ``header_topics``:
    headerless messages such as ``std_msgs/Float32`` start with payload bytes
    that can masquerade as a plausible timestamp.
    """
    header_topics = header_topics or set()
    reader = open_bag_reader(bag_dir, profile)
    reader.open()
    try:
        connections = [
            connection
            for connection in reader.connections
            if only_topics is None or connection.topic in only_topics
        ]
        topics = {
            connection.topic: TopicScan(connection.topic, connection.msgtype)
            for connection in connections
        }
        for connection, record_ns, rawdata in reader.messages(connections=connections):
            scan = topics[connection.topic]
            stamp = None
            if connection.topic in header_topics:
                try:
                    stamp = header_stamp_ns_from_cdr(rawdata)
                except Exception:
                    stamp = None
            scan.receive_ns.append(int(record_ns))
            scan.header_ns.append(stamp)
        start = getattr(reader, "start_time", 0) or 0
        end = getattr(reader, "end_time", 0) or 0
    finally:
        reader.close()

    for scan in topics.values():
        order = np.argsort(np.asarray(scan.receive_ns, dtype=np.int64), kind="stable")
        scan.receive_ns = [scan.receive_ns[index] for index in order]
        scan.header_ns = [scan.header_ns[index] for index in order]
    if not start or not end:
        flat = [value for scan in topics.values() for value in scan.receive_ns]
        start, end = (min(flat), max(flat)) if flat else (0, 0)
    return BagScan(bag_dir=bag_dir, topics=topics, start_ns=int(start), end_ns=int(end))


# ---------------------------------------------------------------------------
# Full scan (conversion)
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
class RawBag:
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
    # ``topic -> "absent" | "empty"`` for profile topics the recording did not
    # deliver; their columns are reconstructed downstream.
    missing_topics: dict[str, str] = field(default_factory=dict)

    def has_arm_command(self, topic: str) -> bool:
        return topic not in self.missing_topics

    def has_effector_command(self, effector: EndEffectorSpec) -> bool:
        return effector.command_topic not in self.missing_topics

    def has_effector_state(self, effector: EndEffectorSpec) -> bool:
        return bool(effector.state_topic) and effector.state_topic not in self.missing_topics


def _ensure_monotonic(name: str, timestamps: np.ndarray) -> None:
    if timestamps.size and np.any(np.diff(timestamps) < 0):
        raise ConversionError(f"{name} timestamps are not monotonic")


def scan_bag(bag_dir: Path, profile: RobotProfile, cfg: AlignmentConfig) -> RawBag:
    """Read every topic the profile names into flat arrays."""
    profile = profile.with_depth(cfg.include_depth)
    required = profile.required_topics
    if cfg.include_depth:
        required |= set(profile.depths.values())

    camera_by_topic = {topic: name for name, topic in profile.cameras.items()}
    depth_by_topic = (
        {topic: name for name, topic in profile.depths.items()} if cfg.include_depth else {}
    )
    arm_cmd_topics = set(profile.arm.command_topics)
    ee_by_state = {e.state_topic: e for e in profile.end_effectors if e.state_topic}
    ee_by_command = {e.command_topic: e for e in profile.end_effectors}

    streams: dict[str, _Stream] = {topic: _Stream() for topic in required | set(depth_by_topic)}
    image_kinds: dict[str, str] = {}
    msgtypes: dict[str, str] = {}

    # Recording generations differ in which of the profile's topics they
    # actually publish, so classify what has to be present up front rather than
    # rejecting anything the profile names.  ``essential`` has no substitute;
    # ``fillable`` (arm and end-effector commands) can be reconstructed from
    # measured state under --missing-topic-policy fill; ``optional`` (measured
    # end-effector state) only enriches the observation when it exists.
    essential = profile.essential_topics | set(depth_by_topic)
    optional = profile.optional_topics
    fillable = required - essential - optional
    missing_topics: dict[str, str] = {}

    reader = open_bag_reader(bag_dir, profile)
    reader.open()
    try:
        available = {connection.topic for connection in reader.connections}
        absent_essential = sorted(essential - available)
        if absent_essential:
            raise ConversionError(f"Missing required topics: {absent_essential}")
        absent_fillable = sorted(fillable - available)
        if absent_fillable and cfg.missing_topic_policy != "fill":
            raise ConversionError(
                f"Missing required topics: {absent_fillable} "
                "(use --missing-topic-policy fill to reconstruct them from measured state)"
            )
        for topic in absent_fillable:
            missing_topics[topic] = TOPIC_ABSENT
        for topic in sorted(optional - available):
            missing_topics[topic] = TOPIC_ABSENT
        wanted = (required | set(depth_by_topic)) & available
        connections = [c for c in reader.connections if c.topic in wanted]
        for connection in connections:
            msgtypes[connection.topic] = connection.msgtype
            if connection.topic in camera_by_topic or connection.topic in depth_by_topic:
                try:
                    image_kinds[connection.topic] = image_kind(connection.msgtype)
                except MessageDecodeError as exc:
                    raise ConversionError(f"{connection.topic}: {exc}") from exc

        for connection, record_ns, rawdata in tracked(
            reader.messages(connections=connections),
            f"scan {bag_dir.name}",
            connection_total(connections),
        ):
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
                        profile.arm.joint_names
                        if profile.arm.joint_state_order == "named"
                        else None
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
                elif kind == FLOAT32MULTIARRAY:
                    # Also headerless, and positional: the profile says which
                    # components of the vector are the effector's position.
                    stamp = int(record_ns)
                    payload = parse_float32multiarray_cdr(rawdata)
                    indices = effector.state_indices if is_state else effector.command_indices
                    if payload.size <= max(indices):
                        raise MessageDecodeError(
                            f"{topic} carries {payload.size} values, need index {max(indices)}"
                        )
                    values = payload[list(indices)]
                else:
                    message, fallback = read_jointstate(reader, rawdata, connection.msgtype)
                    stream.fallbacks += int(fallback)
                    stamp = int(record_ns) if cfg.mode == "lerobot-loop" else message.stamp_ns
                    expected = (
                        effector.joint_names if effector.joint_names and message.names else None
                    )
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

    empty = {topic for topic in required if not streams[topic].times} - set(missing_topics)
    blocking = sorted(empty & essential)
    if blocking:
        raise ConversionError(f"Required topics contain no valid messages: {blocking}")
    stalled = sorted(empty & fillable)
    if stalled and cfg.missing_topic_policy != "fill":
        raise ConversionError(
            f"Required topics contain no valid messages: {stalled} "
            "(use --missing-topic-policy fill to reconstruct them from measured state)"
        )
    for topic in sorted(empty):
        missing_topics[topic] = TOPIC_EMPTY
    if not set(profile.arm.command_topics) - set(missing_topics):
        raise ConversionError(
            "No arm command topic carries any message; the bag records no teleoperation"
        )

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

    return RawBag(
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
        missing_topics=missing_topics,
    )
