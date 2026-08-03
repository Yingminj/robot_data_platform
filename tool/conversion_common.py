#!/usr/bin/env python3
"""Shared, strict alignment and I/O helpers for robot dataset conversion.

The important invariant is one LeRobot-style control row:

    observation at tick k -> teleop action produced immediately after it

Raw sensor capture time (ROS ``header.stamp``) is used in ``capture`` mode.
``lerobot-loop`` mode instead uses rosbag record time as the availability time,
which more closely models LeRobot's asynchronous "latest camera frame" reads.
Neither mode silently substitutes state for a missing action.
"""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DEFAULT_CAMERA_TOPICS = {
    "top": "/camera_head/camera_head/color/image_raw",
    "wrist_L": "/camera_left_wrist/camera_left_wrist/color/image_rect_raw",
    "wrist_R": "/camera_right_wrist/camera_right_wrist/color/image_rect_raw",
}
DEFAULT_DEPTH_TOPICS = {
    "top": "/camera_head/camera_head/aligned_depth_to_color/image_raw",
    "wrist_L": "/camera_left_wrist/camera_left_wrist/aligned_depth_to_color/image_raw",
    "wrist_R": "/camera_right_wrist/camera_right_wrist/aligned_depth_to_color/image_raw",
}
DEFAULT_JOINT_STATE_TOPIC = "/joint_states"
DEFAULT_JOINT_CMD_A_TOPIC = "/control/joint_cmd_A"
DEFAULT_JOINT_CMD_B_TOPIC = "/control/joint_cmd_B"
DEFAULT_GRIPPER_L_TOPIC = "/control/gripperValueL"
DEFAULT_GRIPPER_R_TOPIC = "/control/gripperValueR"

JOINTCMD_DEFINITION = """std_msgs/Header header
float64[7] positions
"""


class ConversionError(RuntimeError):
    """A source episode cannot safely be converted."""


@dataclass(frozen=True)
class TopicConfig:
    cameras: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_CAMERA_TOPICS))
    depths: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_DEPTH_TOPICS))
    joint_states: str = DEFAULT_JOINT_STATE_TOPIC
    joint_cmd_a: str = DEFAULT_JOINT_CMD_A_TOPIC
    joint_cmd_b: str = DEFAULT_JOINT_CMD_B_TOPIC
    gripper_l: str = DEFAULT_GRIPPER_L_TOPIC
    gripper_r: str = DEFAULT_GRIPPER_R_TOPIC


@dataclass(frozen=True)
class AlignmentConfig:
    fps: int = 30
    mode: str = "capture"
    image_tolerance_ms: float | None = None
    state_tolerance_ms: float = 10.0
    action_tolerance_ms: float | None = None
    action_pair_tolerance_ms: float = 5.0
    gripper_tolerance_ms: float = 100.0
    image_height: int = 0
    image_width: int = 0
    joint_state_order: str = "named"
    invalid_frame_policy: str = "fail"
    include_depth: bool = False
    max_decode_errors: int = 0
    action_gap_policy: str = "fail"
    hold_fill_leading: bool = False

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.mode not in {"capture", "lerobot-loop"}:
            raise ValueError("mode must be capture or lerobot-loop")
        if self.invalid_frame_policy not in {"fail", "drop"}:
            raise ValueError("invalid_frame_policy must be fail or drop")
        if self.action_gap_policy not in {"fail", "hold"}:
            raise ValueError("action_gap_policy must be fail or hold")
        if self.joint_state_order not in {"named", "first14"}:
            raise ValueError("joint_state_order must be named or first14")
        if bool(self.image_height) != bool(self.image_width):
            raise ValueError("image_height and image_width must both be zero or both be positive")
        tolerances = {
            "image_tolerance_ms": self.image_tolerance_ms,
            "state_tolerance_ms": self.state_tolerance_ms,
            "action_tolerance_ms": self.action_tolerance_ms,
            "action_pair_tolerance_ms": self.action_pair_tolerance_ms,
            "gripper_tolerance_ms": self.gripper_tolerance_ms,
        }
        invalid = [name for name, value in tolerances.items() if value is not None and value < 0]
        if invalid:
            raise ValueError(f"Alignment tolerances must be non-negative: {invalid}")
        if self.max_decode_errors < 0:
            raise ValueError("max_decode_errors must be non-negative")

    @property
    def resolved_image_tolerance_ms(self) -> float:
        if self.image_tolerance_ms is not None:
            return self.image_tolerance_ms
        return (500.0 if self.mode == "capture" else 1500.0) / self.fps

    @property
    def resolved_action_tolerance_ms(self) -> float:
        """Do not let a missing command select an action from a later control cycle."""
        return self.action_tolerance_ms if self.action_tolerance_ms is not None else 1000.0 / self.fps


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

    @property
    def frame_count(self) -> int:
        return int(self.action.shape[0])


def parse_name_topic(items: list[str] | None, defaults: dict[str, str]) -> dict[str, str]:
    if not items:
        return dict(defaults)
    output: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected NAME=TOPIC, got {item!r}")
        name, topic = item.split("=", 1)
        name, topic = name.strip(), topic.strip()
        if not name or not topic.startswith("/"):
            raise ValueError(f"Invalid NAME=TOPIC mapping: {item!r}")
        if name in output:
            raise ValueError(f"Duplicate camera/depth name: {name}")
        output[name] = topic
    return output


def discover_rosbags(source: Path, recursive: bool = True) -> list[Path]:
    source = source.expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if source.is_file():
        if source.suffix != ".db3":
            raise ValueError(f"Expected a rosbag directory or .db3, got {source}")
        return [source.parent]
    if (source / "metadata.yaml").is_file() or any(source.glob("*.db3")):
        return [source]
    pattern = "**/metadata.yaml" if recursive else "*/metadata.yaml"
    bags = {path.parent for path in source.glob(pattern)}
    db_pattern = "**/*.db3" if recursive else "*/*.db3"
    bags.update(path.parent for path in source.glob(db_pattern))
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


def header_stamp_ns(message: Any) -> int | None:
    header = getattr(message, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return None
    sec = int(getattr(stamp, "sec", 0))
    nanosec = int(getattr(stamp, "nanosec", 0))
    if sec <= 0 or not 0 <= nanosec < 1_000_000_000:
        return None
    return sec * 1_000_000_000 + nanosec


def header_stamp_ns_from_cdr(rawdata: bytes | memoryview) -> int | None:
    """Parse Header.stamp from messages whose first field is std_msgs/Header."""
    data = memoryview(rawdata)
    if len(data) < 12:
        return None
    endian = "<" if data[1] == 1 else ">"
    sec, nanosec = struct.unpack_from(f"{endian}iI", data, 4)
    if sec <= 0 or nanosec >= 1_000_000_000:
        return None
    return sec * 1_000_000_000 + nanosec


def _extract_named(values: Iterable[float], names: Iterable[str]) -> np.ndarray:
    mapping = dict(zip(names, values, strict=False))
    expected = [f"Joint{i}_{side}" for side in ("L", "R") for i in range(1, 8)]
    missing = [name for name in expected if name not in mapping]
    if missing:
        raise ValueError(f"JointState missing required names: {missing}")
    return np.asarray([mapping[name] for name in expected], dtype=np.float64)


def parse_joint_state(message: Any, order: str) -> tuple[np.ndarray, np.ndarray | None]:
    names = list(getattr(message, "name", []))
    positions = list(getattr(message, "position", []))
    velocities = list(getattr(message, "velocity", []))
    if order == "named":
        qpos = _extract_named(positions, names)
        qvel = _extract_named(velocities, names) if len(velocities) == len(names) and names else None
    else:
        if len(positions) < 14:
            raise ValueError(f"JointState has {len(positions)} positions, expected at least 14")
        qpos = np.asarray(positions[:14], dtype=np.float64)
        qvel = np.asarray(velocities[:14], dtype=np.float64) if len(velocities) >= 14 else None
    if not np.all(np.isfinite(qpos)):
        raise ValueError("JointState contains non-finite positions")
    return qpos, qvel


def decode_ros_image(message: Any) -> np.ndarray:
    import cv2

    encoding = str(getattr(message, "encoding", "")).lower()
    width, height = int(message.width), int(message.height)
    step = int(getattr(message, "step", 0))
    raw = bytes(message.data)
    if width <= 0 or height <= 0:
        raise ValueError("Image has invalid dimensions")
    if encoding in {"rgb8", "bgr8"}:
        row_bytes = step or width * 3
        array = np.frombuffer(raw, dtype=np.uint8).reshape(height, row_bytes)[:, : width * 3]
        image = array.reshape(height, width, 3)
        if encoding == "bgr8":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif encoding in {"rgba8", "bgra8"}:
        row_bytes = step or width * 4
        array = np.frombuffer(raw, dtype=np.uint8).reshape(height, row_bytes)[:, : width * 4]
        image = array.reshape(height, width, 4)
        code = cv2.COLOR_RGBA2RGB if encoding == "rgba8" else cv2.COLOR_BGRA2RGB
        image = cv2.cvtColor(image, code)
    elif encoding in {"mono8", "8uc1"}:
        row_bytes = step or width
        mono = np.frombuffer(raw, dtype=np.uint8).reshape(height, row_bytes)[:, :width]
        image = cv2.cvtColor(mono, cv2.COLOR_GRAY2RGB)
    else:
        raise ValueError(f"Unsupported RGB encoding: {encoding}")
    return np.ascontiguousarray(image, dtype=np.uint8)


def decode_ros_depth(message: Any) -> np.ndarray:
    encoding = str(getattr(message, "encoding", "")).lower()
    width, height = int(message.width), int(message.height)
    step = int(getattr(message, "step", 0))
    if encoding in {"16uc1", "mono16"}:
        dtype, scale = np.uint16, 0.001
    elif encoding == "32fc1":
        dtype, scale = np.float32, 1.0
    elif encoding == "64fc1":
        dtype, scale = np.float64, 1.0
    else:
        raise ValueError(f"Unsupported depth encoding: {encoding}")
    row_values = (step // np.dtype(dtype).itemsize) if step else width
    depth = np.frombuffer(bytes(message.data), dtype=dtype).reshape(height, row_values)[:, :width]
    depth = depth.astype(np.float32) * scale
    depth[~np.isfinite(depth)] = 0
    depth[depth < 0] = 0
    return np.ascontiguousarray(depth)


def resize_letterbox(array: np.ndarray, height: int, width: int, is_depth: bool = False) -> np.ndarray:
    if not height or not width or array.shape[:2] == (height, width):
        return array
    import cv2

    old_h, old_w = array.shape[:2]
    scale = min(width / old_w, height / old_h)
    new_w, new_h = round(old_w * scale), round(old_h * scale)
    interpolation = cv2.INTER_NEAREST if is_depth else cv2.INTER_LINEAR
    resized = cv2.resize(array, (new_w, new_h), interpolation=interpolation)
    shape = (height, width) if is_depth else (height, width, array.shape[2])
    output = np.zeros(shape, dtype=array.dtype)
    y, x = (height - new_h) // 2, (width - new_w) // 2
    output[y : y + new_h, x : x + new_w] = resized
    return output


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
    alpha = np.divide(
        target - source_t[left_safe], denom, out=np.zeros_like(denom), where=denom != 0
    )
    output = source_v[left_safe] + alpha[:, None] * (source_v[right_safe] - source_v[left_safe])
    output[exact] = source_v[right_safe[exact]]
    selected_time = np.where(exact, source_t[right_safe], target)
    return output, selected_time, valid


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


def _compute_hold_segments(
    hold_mask: np.ndarray, grid_ns: np.ndarray, arm_qpos: np.ndarray, fps: int
) -> list[dict[str, Any]]:
    """Compute contiguous hold segments with drift metrics."""
    if not hold_mask.any():
        return []

    segments: list[dict[str, Any]] = []
    in_segment = False
    segment_start_idx = 0

    for i in range(len(hold_mask)):
        if hold_mask[i] and not in_segment:
            in_segment = True
            segment_start_idx = i
        elif not hold_mask[i] and in_segment:
            # Segment ended at i-1
            segment_end_idx = i - 1
            segments.append(_build_segment_info(segment_start_idx, segment_end_idx, grid_ns, arm_qpos, fps))
            in_segment = False

    # Handle segment extending to the end
    if in_segment:
        segment_end_idx = len(hold_mask) - 1
        segments.append(_build_segment_info(segment_start_idx, segment_end_idx, grid_ns, arm_qpos, fps))

    return segments


def _build_segment_info(
    start_idx: int, end_idx: int, grid_ns: np.ndarray, arm_qpos: np.ndarray, fps: int
) -> dict[str, Any]:
    """Build segment info with drift metrics."""
    segment_rows = end_idx - start_idx + 1
    start_time_s = float(grid_ns[start_idx]) / 1e9
    end_time_s = float(grid_ns[end_idx]) / 1e9

    # Compute max single-step joint drift in this segment
    if segment_rows > 1:
        segment_qpos = arm_qpos[start_idx : end_idx + 1]
        diffs = np.abs(segment_qpos[1:] - segment_qpos[:-1])
        max_joint_drift_rad = float(np.max(diffs))
    else:
        max_joint_drift_rad = 0.0

    return {
        "start_time_s": start_time_s,
        "end_time_s": end_time_s,
        "duration_s": end_time_s - start_time_s,
        "rows": segment_rows,
        "max_joint_drift_rad": max_joint_drift_rad,
    }


@dataclass
class _RawBag:
    bag_dir: Path
    joint_t: np.ndarray
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    cmd_a_t: np.ndarray
    cmd_a: np.ndarray
    cmd_b_t: np.ndarray
    cmd_b: np.ndarray
    grip_l_t: np.ndarray
    grip_l: np.ndarray
    grip_r_t: np.ndarray
    grip_r: np.ndarray
    image_t: dict[str, np.ndarray]
    depth_t: dict[str, np.ndarray]
    topic_counts: dict[str, int]
    timestamp_sources: dict[str, str]


def _rosbags_reader(bag_dir: Path) -> Any:
    try:
        from rosbags.highlevel import AnyReader
        from rosbags.typesys import Stores, get_types_from_msg, get_typestore
    except ImportError as exc:
        raise RuntimeError("Missing dependency: pip install rosbags") from exc
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    typestore.register(get_types_from_msg(JOINTCMD_DEFINITION, "marvin_msgs/msg/Jointcmd"))
    return AnyReader([bag_dir], default_typestore=typestore)


def _ensure_monotonic(name: str, timestamps: np.ndarray) -> None:
    if timestamps.size and np.any(np.diff(timestamps) < 0):
        raise ConversionError(f"{name} timestamps are not monotonic")


def _scan_bag(bag_dir: Path, topics: TopicConfig, cfg: AlignmentConfig) -> _RawBag:
    requested = {
        topics.joint_states,
        topics.joint_cmd_a,
        topics.joint_cmd_b,
        topics.gripper_l,
        topics.gripper_r,
        *topics.cameras.values(),
    }
    if cfg.include_depth:
        requested.update(topics.depths.values())

    joint_t: list[int] = []
    joint_pos: list[np.ndarray] = []
    joint_vel: list[np.ndarray] = []
    cmd_a_t: list[int] = []
    cmd_a: list[np.ndarray] = []
    cmd_b_t: list[int] = []
    cmd_b: list[np.ndarray] = []
    grip_l_t: list[int] = []
    grip_l: list[float] = []
    grip_r_t: list[int] = []
    grip_r: list[float] = []
    image_t = {name: [] for name in topics.cameras}
    depth_t = {name: [] for name in topics.depths} if cfg.include_depth else {}
    camera_by_topic = {topic: name for name, topic in topics.cameras.items()}
    depth_by_topic = {topic: name for name, topic in topics.depths.items()} if cfg.include_depth else {}
    errors: dict[str, int] = {topic: 0 for topic in requested}
    counts: dict[str, int] = {topic: 0 for topic in requested}

    reader = _rosbags_reader(bag_dir)
    reader.open()
    try:
        available = set(reader.topics)
        missing = sorted(requested - available)
        if missing:
            raise ConversionError(f"Missing required topics: {missing}")
        connections = [connection for connection in reader.connections if connection.topic in requested]
        for connection, record_ns, rawdata in reader.messages(connections=connections):
            topic = connection.topic
            counts[topic] += 1
            try:
                if topic in camera_by_topic:
                    timestamp = (
                        header_stamp_ns_from_cdr(rawdata) if cfg.mode == "capture" else int(record_ns)
                    )
                    if timestamp is None:
                        raise ValueError("image has no valid header.stamp")
                    image_t[camera_by_topic[topic]].append(timestamp)
                    continue
                if topic in depth_by_topic:
                    timestamp = (
                        header_stamp_ns_from_cdr(rawdata) if cfg.mode == "capture" else int(record_ns)
                    )
                    if timestamp is None:
                        raise ValueError("depth image has no valid header.stamp")
                    depth_t[depth_by_topic[topic]].append(timestamp)
                    continue

                message = reader.deserialize(rawdata, connection.msgtype)
                header_ns = header_stamp_ns(message)
                timestamp = int(record_ns) if cfg.mode == "lerobot-loop" else header_ns
                if topic in {topics.gripper_l, topics.gripper_r}:
                    # std_msgs/Float32 has no Header; record time is its only clock.
                    timestamp = int(record_ns)
                if timestamp is None:
                    raise ValueError("message has no valid header.stamp")

                if topic == topics.joint_states:
                    position, velocity = parse_joint_state(message, cfg.joint_state_order)
                    joint_t.append(timestamp)
                    joint_pos.append(position)
                    joint_vel.append(
                        velocity if velocity is not None else np.full(14, np.nan, dtype=np.float64)
                    )
                elif topic == topics.joint_cmd_a:
                    value = np.asarray(message.positions, dtype=np.float64)
                    if value.shape != (7,) or not np.all(np.isfinite(value)):
                        raise ValueError(f"joint_cmd_A shape/value invalid: {value.shape}")
                    cmd_a_t.append(timestamp)
                    cmd_a.append(value)
                elif topic == topics.joint_cmd_b:
                    value = np.asarray(message.positions, dtype=np.float64)
                    if value.shape != (7,) or not np.all(np.isfinite(value)):
                        raise ValueError(f"joint_cmd_B shape/value invalid: {value.shape}")
                    cmd_b_t.append(timestamp)
                    cmd_b.append(value)
                elif topic == topics.gripper_l:
                    value = float(message.data)
                    if not math.isfinite(value):
                        raise ValueError("non-finite left gripper value")
                    grip_l_t.append(timestamp)
                    grip_l.append(value)
                elif topic == topics.gripper_r:
                    value = float(message.data)
                    if not math.isfinite(value):
                        raise ValueError("non-finite right gripper value")
                    grip_r_t.append(timestamp)
                    grip_r.append(value)
            except Exception:
                errors[topic] += 1
    finally:
        reader.close()

    media_topics = set(camera_by_topic) | set(depth_by_topic)
    excessive = {
        topic: count
        for topic, count in errors.items()
        if count > (0 if topic in media_topics else cfg.max_decode_errors)
    }
    if excessive:
        raise ConversionError(f"Decode/validation errors exceed limit: {excessive}")

    def array1(values: list[Any], dtype: Any) -> np.ndarray:
        return np.asarray(values, dtype=dtype)

    required_nonempty = {
        topics.joint_states: joint_t,
        topics.joint_cmd_a: cmd_a_t,
        topics.joint_cmd_b: cmd_b_t,
        topics.gripper_l: grip_l_t,
        topics.gripper_r: grip_r_t,
        **{topics.cameras[name]: values for name, values in image_t.items()},
    }
    if cfg.include_depth:
        required_nonempty.update({topics.depths[name]: values for name, values in depth_t.items()})
    empty = [topic for topic, values in required_nonempty.items() if not values]
    if empty:
        raise ConversionError(f"Required topics contain no valid messages: {empty}")

    arrays_to_check = {
        topics.joint_states: array1(joint_t, np.int64),
        topics.joint_cmd_a: array1(cmd_a_t, np.int64),
        topics.joint_cmd_b: array1(cmd_b_t, np.int64),
        topics.gripper_l: array1(grip_l_t, np.int64),
        topics.gripper_r: array1(grip_r_t, np.int64),
        **{topics.cameras[name]: array1(values, np.int64) for name, values in image_t.items()},
        **{topics.depths[name]: array1(values, np.int64) for name, values in depth_t.items()},
    }
    for name, values in arrays_to_check.items():
        _ensure_monotonic(name, values)

    return _RawBag(
        bag_dir=bag_dir,
        joint_t=arrays_to_check[topics.joint_states],
        joint_pos=np.asarray(joint_pos, dtype=np.float64),
        joint_vel=np.asarray(joint_vel, dtype=np.float64),
        cmd_a_t=arrays_to_check[topics.joint_cmd_a],
        cmd_a=np.asarray(cmd_a, dtype=np.float64),
        cmd_b_t=arrays_to_check[topics.joint_cmd_b],
        cmd_b=np.asarray(cmd_b, dtype=np.float64),
        grip_l_t=arrays_to_check[topics.gripper_l],
        grip_l=np.asarray(grip_l, dtype=np.float64),
        grip_r_t=arrays_to_check[topics.gripper_r],
        grip_r=np.asarray(grip_r, dtype=np.float64),
        image_t={name: arrays_to_check[topic] for name, topic in topics.cameras.items()},
        depth_t={name: arrays_to_check[topic] for name, topic in topics.depths.items()}
        if cfg.include_depth
        else {},
        topic_counts=counts,
        timestamp_sources={
            "images": "header.stamp" if cfg.mode == "capture" else "rosbag record timestamp",
            "joint_states": "header.stamp" if cfg.mode == "capture" else "rosbag record timestamp",
            "joint_commands": "header.stamp" if cfg.mode == "capture" else "rosbag record timestamp",
            "grippers": "rosbag record timestamp (std_msgs/Float32 has no Header)",
        },
    )


def _decode_selected_media(
    raw: _RawBag,
    topics: TopicConfig,
    cfg: AlignmentConfig,
    image_indices: dict[str, np.ndarray],
    depth_indices: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    selected_by_topic: dict[str, set[int]] = {
        topics.cameras[name]: set(np.unique(indices).tolist()) for name, indices in image_indices.items()
    }
    if cfg.include_depth:
        selected_by_topic.update(
            {topics.depths[name]: set(np.unique(indices).tolist()) for name, indices in depth_indices.items()}
        )
    counters = {topic: 0 for topic in selected_by_topic}
    decoded: dict[str, dict[int, np.ndarray]] = {topic: {} for topic in selected_by_topic}
    camera_topics = set(topics.cameras.values())

    reader = _rosbags_reader(raw.bag_dir)
    reader.open()
    try:
        connections = [connection for connection in reader.connections if connection.topic in selected_by_topic]
        for connection, _, rawdata in reader.messages(connections=connections):
            topic = connection.topic
            index = counters[topic]
            counters[topic] += 1
            if index not in selected_by_topic[topic]:
                continue
            try:
                message = reader.deserialize(rawdata, connection.msgtype)
                if topic in camera_topics:
                    value = decode_ros_image(message)
                    value = resize_letterbox(value, cfg.image_height, cfg.image_width)
                else:
                    value = decode_ros_depth(message)
                    value = resize_letterbox(
                        value, cfg.image_height, cfg.image_width, is_depth=True
                    )
                decoded[topic][index] = value
            except Exception as exc:
                raise ConversionError(f"Failed to decode selected {topic} message {index}: {exc}") from exc
    finally:
        reader.close()

    images: dict[str, np.ndarray] = {}
    depths: dict[str, np.ndarray] = {}
    for name, indices in image_indices.items():
        topic = topics.cameras[name]
        missing = sorted(set(indices.tolist()) - set(decoded[topic]))
        if missing:
            raise ConversionError(f"Selected RGB frames missing for {name}: {missing[:10]}")
        images[name] = np.stack([decoded[topic][int(index)] for index in indices]).astype(
            np.uint8, copy=False
        )
    for name, indices in depth_indices.items():
        topic = topics.depths[name]
        missing = sorted(set(indices.tolist()) - set(decoded[topic]))
        if missing:
            raise ConversionError(f"Selected depth frames missing for {name}: {missing[:10]}")
        depths[name] = np.stack([decoded[topic][int(index)] for index in indices]).astype(
            np.float32, copy=False
        )
    return images, depths


def align_rosbag(
    bag_dir: Path, topics: TopicConfig, cfg: AlignmentConfig
) -> AlignedEpisode:
    """Convert one rosbag into strict, fixed-rate LeRobot-style rows."""
    raw = _scan_bag(bag_dir.expanduser().resolve(), topics, cfg)

    # Window computation: exclude joint_cmd topics when hold_fill_leading is enabled
    if cfg.action_gap_policy == "hold" and cfg.hold_fill_leading:
        window_times = [
            raw.joint_t,
            raw.grip_l_t,
            raw.grip_r_t,
            *raw.image_t.values(),
            *raw.depth_t.values(),
        ]
    else:
        window_times = [
            raw.joint_t,
            raw.cmd_a_t,
            raw.cmd_b_t,
            raw.grip_l_t,
            raw.grip_r_t,
            *raw.image_t.values(),
            *raw.depth_t.values(),
        ]
    start_ns = max(int(values[0]) for values in window_times)
    end_ns = min(int(values[-1]) for values in window_times)
    if end_ns <= start_ns:
        raise ConversionError("Required topic time ranges do not overlap")
    frame_count = int(math.floor((end_ns - start_ns) * cfg.fps / 1e9)) + 1
    if frame_count < 2:
        raise ConversionError(f"Common time range produces only {frame_count} frame(s)")
    grid_ns = start_ns + np.rint(np.arange(frame_count) * 1e9 / cfg.fps).astype(np.int64)

    image_tol = round(cfg.resolved_image_tolerance_ms * 1e6)
    state_tol = round(cfg.state_tolerance_ms * 1e6)
    action_tol = round(cfg.resolved_action_tolerance_ms * 1e6)
    pair_tol = round(cfg.action_pair_tolerance_ms * 1e6)
    gripper_tol = round(cfg.gripper_tolerance_ms * 1e6)

    valid_parts: dict[str, np.ndarray] = {}
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

    if cfg.mode == "capture":
        arm_qpos, joint_selected_t, joint_valid = _interpolate(
            raw.joint_t, raw.joint_pos, grid_ns, state_tol
        )
        if np.all(np.isfinite(raw.joint_vel)):
            arm_qvel, _, velocity_valid = _interpolate(
                raw.joint_t, raw.joint_vel, grid_ns, state_tol
            )
            joint_valid &= velocity_valid
        else:
            arm_qvel = np.full_like(arm_qpos, np.nan)
    else:
        joint_indices, joint_age, joint_valid = _previous_indices(raw.joint_t, grid_ns)
        joint_valid &= joint_age <= state_tol
        arm_qpos = raw.joint_pos[joint_indices]
        joint_selected_t = raw.joint_t[joint_indices]
        arm_qvel = raw.joint_vel[joint_indices]
    valid_parts["joint_state"] = joint_valid

    # Observation gripper state is the most recently commanded endpoint value.
    obs_grip_l_idx, obs_grip_l_age, obs_grip_l_valid = _previous_indices(raw.grip_l_t, grid_ns)
    obs_grip_r_idx, obs_grip_r_age, obs_grip_r_valid = _previous_indices(raw.grip_r_t, grid_ns)
    obs_grip_l_valid &= obs_grip_l_age <= gripper_tol
    obs_grip_r_valid &= obs_grip_r_age <= gripper_tol
    valid_parts["observation_gripper_L"] = obs_grip_l_valid
    valid_parts["observation_gripper_R"] = obs_grip_r_valid

    # LeRobot row semantics: teleop actions are obtained after observation.
    cmd_a_idx, cmd_a_lead, cmd_a_valid = _next_indices(raw.cmd_a_t, grid_ns)
    cmd_b_idx, cmd_b_lead, cmd_b_valid = _next_indices(raw.cmd_b_t, grid_ns)
    cmd_a_valid &= cmd_a_lead <= action_tol
    cmd_b_valid &= cmd_b_lead <= action_tol
    command_skew = np.abs(raw.cmd_a_t[cmd_a_idx] - raw.cmd_b_t[cmd_b_idx])
    pair_valid = command_skew <= pair_tol

    # Gripper topics are slower than arm commands and represent a held endpoint
    # value.  Use the latest value available when each teleop arm command is made.
    action_grip_l_idx, action_grip_l_age, action_grip_l_valid = _previous_indices(
        raw.grip_l_t, raw.cmd_a_t[cmd_a_idx]
    )
    action_grip_r_idx, action_grip_r_age, action_grip_r_valid = _previous_indices(
        raw.grip_r_t, raw.cmd_b_t[cmd_b_idx]
    )
    action_grip_l_valid &= action_grip_l_age <= gripper_tol
    action_grip_r_valid &= action_grip_r_age <= gripper_tol

    # Hold mode: action gaps are filled with next-tick qpos instead of failing
    action_hold_mask: np.ndarray | None = None
    hold_audit: dict[str, Any] = {}
    if cfg.action_gap_policy == "hold":
        # Determine which rows need hold fill: either arm invalid OR pair invalid
        cmd_a_real = cmd_a_valid & pair_valid
        cmd_b_real = cmd_b_valid & pair_valid
        action_hold_mask = ~(cmd_a_real & cmd_b_real)

        # Count real teleop command rows (at least one arm has real command)
        real_cmd_rows = int((cmd_a_real | cmd_b_real).sum())
        if real_cmd_rows == 0:
            raise ConversionError("episode contains no real teleop command rows")

        hold_audit = {
            "hold_rows": int(action_hold_mask.sum()),
            "hold_fraction": float(action_hold_mask.sum() / frame_count),
            "hold_rows_by_arm": {
                "A": int((~cmd_a_real).sum()),
                "B": int((~cmd_b_real).sum()),
                "both": int((~cmd_a_real & ~cmd_b_real).sum()),
            },
            "real_command_rows": real_cmd_rows,
        }

        # Do not add joint_cmd checks to valid_parts in hold mode
        valid_parts["action_gripper_L"] = action_grip_l_valid
        valid_parts["action_gripper_R"] = action_grip_r_valid
    else:
        # Fail mode: original strict checking
        valid_parts["joint_cmd_A"] = cmd_a_valid
        valid_parts["joint_cmd_B"] = cmd_b_valid
        valid_parts["joint_cmd_pair"] = pair_valid
        valid_parts["action_gripper_L"] = action_grip_l_valid
        valid_parts["action_gripper_R"] = action_grip_r_valid

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
    joint_selected_t = keep(joint_selected_t)
    obs_grip_l_idx = keep(obs_grip_l_idx)
    obs_grip_r_idx = keep(obs_grip_r_idx)
    cmd_a_idx = keep(cmd_a_idx)
    cmd_b_idx = keep(cmd_b_idx)
    action_grip_l_idx = keep(action_grip_l_idx)
    action_grip_r_idx = keep(action_grip_r_idx)
    command_skew = keep(command_skew)
    cmd_a_lead = keep(cmd_a_lead)
    cmd_b_lead = keep(cmd_b_lead)
    action_grip_l_age = keep(action_grip_l_age)
    action_grip_r_age = keep(action_grip_r_age)
    image_indices = {name: keep(indices) for name, indices in image_indices.items()}
    depth_indices = {name: keep(indices) for name, indices in depth_indices.items()}
    if action_hold_mask is not None:
        action_hold_mask = keep(action_hold_mask)

    obs_grip_l = raw.grip_l[obs_grip_l_idx, None]
    obs_grip_r = raw.grip_r[obs_grip_r_idx, None]
    qpos = np.concatenate([arm_qpos, obs_grip_l, obs_grip_r], axis=1).astype(np.float32)

    # Build action: in hold mode, use next-tick qpos for missing commands
    if cfg.action_gap_policy == "hold":
        cmd_a_real = keep(cmd_a_valid & pair_valid)
        cmd_b_real = keep(cmd_b_valid & pair_valid)

        # Prepare next-tick arm_qpos for hold fill
        arm_qpos_next = np.empty_like(arm_qpos)
        arm_qpos_next[:-1] = arm_qpos[1:]
        arm_qpos_next[-1] = arm_qpos[-1]  # Last row copies itself

        # Build action with conditional selection
        action_left = np.where(cmd_a_real[:, None], raw.cmd_a[cmd_a_idx], arm_qpos_next[:, :7])
        action_right = np.where(cmd_b_real[:, None], raw.cmd_b[cmd_b_idx], arm_qpos_next[:, 7:14])

        action = np.concatenate(
            [
                action_left,
                action_right,
                raw.grip_l[action_grip_l_idx, None],
                raw.grip_r[action_grip_r_idx, None],
            ],
            axis=1,
        ).astype(np.float32)

        # Compute hold segments for audit
        hold_segments = _compute_hold_segments(action_hold_mask, grid_ns, arm_qpos, cfg.fps)
        hold_audit["hold_segments"] = hold_segments
    else:
        # Fail mode: use real commands only
        action = np.concatenate(
            [
                raw.cmd_a[cmd_a_idx],
                raw.cmd_b[cmd_b_idx],
                raw.grip_l[action_grip_l_idx, None],
                raw.grip_r[action_grip_r_idx, None],
            ],
            axis=1,
        ).astype(np.float32)
    if np.array_equal(qpos, action):
        raise ConversionError("Entire action array equals observation state; refusing unsafe dataset")

    if not np.all(np.isfinite(arm_qvel)):
        arm_qvel = np.gradient(arm_qpos, 1.0 / cfg.fps, axis=0)
    grip_velocity = np.gradient(
        np.concatenate([obs_grip_l, obs_grip_r], axis=1), 1.0 / cfg.fps, axis=0
    )
    qvel = np.concatenate([arm_qvel, grip_velocity], axis=1).astype(np.float32)

    images, depths = _decode_selected_media(
        raw, topics, cfg, image_indices=image_indices, depth_indices=depth_indices
    )
    timestamps: dict[str, np.ndarray] = {
        "grid_ns": grid_ns,
        "joint_state_ns": joint_selected_t,
        "joint_cmd_A_ns": raw.cmd_a_t[cmd_a_idx],
        "joint_cmd_B_ns": raw.cmd_b_t[cmd_b_idx],
        "observation_gripper_L_ns": raw.grip_l_t[obs_grip_l_idx],
        "observation_gripper_R_ns": raw.grip_r_t[obs_grip_r_idx],
        "action_gripper_L_ns": raw.grip_l_t[action_grip_l_idx],
        "action_gripper_R_ns": raw.grip_r_t[action_grip_r_idx],
    }
    if action_hold_mask is not None:
        timestamps["action_hold_mask"] = action_hold_mask
    for name, indices in image_indices.items():
        timestamps[f"image_{name}_ns"] = raw.image_t[name][indices]
    for name, indices in depth_indices.items():
        timestamps[f"depth_{name}_ns"] = raw.depth_t[name][indices]

    audit = {
        "schema_version": "aligned-control-rows-v1",
        "source_bag": str(raw.bag_dir),
        "alignment_mode": cfg.mode,
        "action_gap_policy": cfg.action_gap_policy,
        "fps": cfg.fps,
        "candidate_frames": frame_count,
        "output_frames": int(qpos.shape[0]),
        "dropped_frames": int(frame_count - qpos.shape[0]),
        "invalid_counts": invalid_counts,
        "timestamp_sources": raw.timestamp_sources,
        "topic_counts": raw.topic_counts,
        "limits_ms": {
            "image": cfg.resolved_image_tolerance_ms,
            "state": cfg.state_tolerance_ms,
            "action": cfg.resolved_action_tolerance_ms,
            "action_pair": cfg.action_pair_tolerance_ms,
            "gripper": cfg.gripper_tolerance_ms,
        },
        "metrics_ms": {
            "cmd_A_after_observation": _stats_ms(cmd_a_lead),
            "cmd_B_after_observation": _stats_ms(cmd_b_lead),
            "cmd_A_B_skew": _stats_ms(command_skew),
            "gripper_L_age_at_cmd_A": _stats_ms(action_grip_l_age),
            "gripper_R_age_at_cmd_B": _stats_ms(action_grip_r_age),
            **{
                f"image_{name}_offset": _stats_ms(keep(offsets))
                for name, offsets in image_offsets.items()
            },
            **{
                f"depth_{name}_offset": _stats_ms(keep(offsets))
                for name, offsets in depth_offsets.items()
            },
        },
        **hold_audit,
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
    )


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
    topics: TopicConfig,
    compression: str = "none",
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
        output.attrs["schema_version"] = "aligned-control-rows-v1"
        output.attrs["fps"] = episode.fps
        output.attrs["source_bag"] = episode.source
        output.attrs["alignment_mode"] = episode.audit["alignment_mode"]
        output.attrs["topics_json"] = json.dumps(
            {
                "cameras": topics.cameras,
                "depths": topics.depths if episode.depths else {},
                "joint_states": topics.joint_states,
                "joint_cmd_A": topics.joint_cmd_a,
                "joint_cmd_B": topics.joint_cmd_b,
                "gripper_L": topics.gripper_l,
                "gripper_R": topics.gripper_r,
            },
            sort_keys=True,
        )
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
            timestamp_group.create_dataset(name, data=values, dtype="int64")
