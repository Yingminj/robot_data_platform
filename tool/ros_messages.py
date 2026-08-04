#!/usr/bin/env python3
"""ROS 2 message decoding for rosbag conversion.

Covers the parts of the pipeline that must tolerate real recordings:

* ``sensor_msgs/Image`` and ``sensor_msgs/CompressedImage`` are both accepted
  and normalised to RGB uint8; the format is detected from the connection's
  message type rather than configured by hand.
* ``sensor_msgs/JointState`` messages produced by some publishers carry
  trailing bytes left over from a reused serialisation buffer.  ROS 2's own
  deserialiser ignores them, but ``rosbags`` asserts on unconsumed input.  A
  tolerant CDR reader is used as a fallback so those topics remain readable,
  and the number of fallbacks is reported so the underlying recorder bug stays
  visible instead of being silently absorbed.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any

import numpy as np


RAW_IMAGE_TYPES = frozenset({"sensor_msgs/msg/Image"})
COMPRESSED_IMAGE_TYPES = frozenset({"sensor_msgs/msg/CompressedImage"})
IMAGE_TYPES = RAW_IMAGE_TYPES | COMPRESSED_IMAGE_TYPES

RAW_IMAGE = "raw_image"
COMPRESSED_IMAGE = "compressed_image"

# CDR alignment is measured from the end of the 4-byte encapsulation header.
_CDR_ORIGIN = 4


class MessageDecodeError(ValueError):
    """A message could not be decoded."""


def image_kind(msgtype: str) -> str:
    """Classify a connection message type as raw or compressed imagery."""
    if msgtype in RAW_IMAGE_TYPES:
        return RAW_IMAGE
    if msgtype in COMPRESSED_IMAGE_TYPES:
        return COMPRESSED_IMAGE
    raise MessageDecodeError(f"not an image message type: {msgtype}")


# ---------------------------------------------------------------------------
# Tolerant CDR primitives
# ---------------------------------------------------------------------------


class _CDRReader:
    """Minimal little/big-endian CDR reader with ROS 2 alignment rules."""

    __slots__ = ("data", "endian", "pos")

    def __init__(self, rawdata: bytes | memoryview) -> None:
        data = memoryview(rawdata)
        if len(data) < _CDR_ORIGIN:
            raise MessageDecodeError("message is shorter than the CDR encapsulation header")
        self.data = data
        self.endian = "<" if data[1] & 1 else ">"
        self.pos = _CDR_ORIGIN

    @property
    def remaining(self) -> int:
        return len(self.data) - self.pos

    def align(self, width: int) -> None:
        offset = self.pos - _CDR_ORIGIN
        self.pos = _CDR_ORIGIN + (offset + width - 1) // width * width

    def _unpack(self, fmt: str, width: int) -> Any:
        self.align(width)
        if self.remaining < width:
            raise MessageDecodeError("truncated CDR payload")
        value = struct.unpack_from(self.endian + fmt, self.data, self.pos)[0]
        self.pos += width
        return value

    def int32(self) -> int:
        return int(self._unpack("i", 4))

    def uint32(self) -> int:
        return int(self._unpack("I", 4))

    def float32(self) -> float:
        return float(self._unpack("f", 4))

    def string(self) -> str:
        length = self.uint32()
        if length > self.remaining:
            raise MessageDecodeError("CDR string length exceeds payload")
        raw = bytes(self.data[self.pos : self.pos + max(length - 1, 0)])
        self.pos += length
        return raw.decode("utf-8", errors="replace")

    def float64_sequence(self) -> np.ndarray:
        count = self.uint32()
        if count == 0:
            return np.empty(0, dtype=np.float64)
        self.align(8)
        if count * 8 > self.remaining:
            raise MessageDecodeError(
                f"CDR sequence declares {count} float64 but only {self.remaining // 8} remain"
            )
        values = np.frombuffer(self.data, dtype=np.dtype(self.endian + "f8"), count=count, offset=self.pos)
        self.pos += count * 8
        return np.asarray(values, dtype=np.float64)

    def float64_array(self, count: int) -> np.ndarray:
        """Fixed-size array: no length prefix, elements aligned to 8 bytes."""
        if count == 0:
            return np.empty(0, dtype=np.float64)
        self.align(8)
        if count * 8 > self.remaining:
            raise MessageDecodeError("CDR fixed array exceeds payload")
        values = np.frombuffer(self.data, dtype=np.dtype(self.endian + "f8"), count=count, offset=self.pos)
        self.pos += count * 8
        return np.asarray(values, dtype=np.float64)

    def header_stamp_ns(self) -> tuple[int, str]:
        sec = self.int32()
        nanosec = self.uint32()
        frame_id = self.string()
        if sec <= 0 or not 0 <= nanosec < 1_000_000_000:
            raise MessageDecodeError(f"implausible header stamp: sec={sec} nanosec={nanosec}")
        return sec * 1_000_000_000 + nanosec, frame_id


def header_stamp_ns_from_cdr(rawdata: bytes | memoryview) -> int | None:
    """Read ``header.stamp`` from any message whose first field is a Header."""
    data = memoryview(rawdata)
    if len(data) < 12:
        return None
    endian = "<" if data[1] & 1 else ">"
    sec, nanosec = struct.unpack_from(f"{endian}iI", data, _CDR_ORIGIN)
    if sec <= 0 or nanosec >= 1_000_000_000:
        return None
    return sec * 1_000_000_000 + nanosec


# ---------------------------------------------------------------------------
# JointState
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JointStateMessage:
    stamp_ns: int
    names: tuple[str, ...]
    position: np.ndarray
    velocity: np.ndarray
    effort: np.ndarray
    trailing_bytes: int = 0

    def ordered(self, expected: tuple[str, ...] | None, dim: int) -> np.ndarray:
        """Return ``dim`` positions, reordered by name when names are present."""
        if expected and self.names:
            mapping = dict(zip(self.names, self.position, strict=False))
            missing = [name for name in expected if name not in mapping]
            if missing:
                raise MessageDecodeError(f"JointState missing required names: {missing[:8]}")
            return np.asarray([mapping[name] for name in expected], dtype=np.float64)
        if self.position.size < dim:
            raise MessageDecodeError(
                f"JointState carries {self.position.size} positions, expected at least {dim}"
            )
        return np.asarray(self.position[:dim], dtype=np.float64)

    def ordered_velocity(self, expected: tuple[str, ...] | None, dim: int) -> np.ndarray | None:
        if self.velocity.size == 0:
            return None
        if expected and self.names and self.velocity.size == len(self.names):
            mapping = dict(zip(self.names, self.velocity, strict=False))
            if all(name in mapping for name in expected):
                return np.asarray([mapping[name] for name in expected], dtype=np.float64)
        if self.velocity.size < dim:
            return None
        return np.asarray(self.velocity[:dim], dtype=np.float64)


def parse_jointstate_cdr(rawdata: bytes | memoryview) -> JointStateMessage:
    """Parse ``sensor_msgs/JointState`` directly, tolerating trailing bytes."""
    reader = _CDRReader(rawdata)
    stamp_ns, _ = reader.header_stamp_ns()
    count = reader.uint32()
    if count > 4096:
        raise MessageDecodeError(f"implausible JointState name count: {count}")
    names = tuple(reader.string() for _ in range(count))
    position = reader.float64_sequence()
    velocity = reader.float64_sequence()
    effort = reader.float64_sequence()
    return JointStateMessage(
        stamp_ns=stamp_ns,
        names=names,
        position=position,
        velocity=velocity,
        effort=effort,
        trailing_bytes=reader.remaining,
    )


def read_jointstate(reader: Any, rawdata: bytes | memoryview, msgtype: str) -> tuple[JointStateMessage, bool]:
    """Deserialise a JointState, falling back to the tolerant parser.

    Returns the message and whether the tolerant fallback was needed.
    """
    try:
        message = reader.deserialize(rawdata, msgtype)
    except Exception:
        return parse_jointstate_cdr(rawdata), True
    stamp = message.header.stamp
    return (
        JointStateMessage(
            stamp_ns=int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec),
            names=tuple(message.name),
            position=np.asarray(message.position, dtype=np.float64),
            velocity=np.asarray(message.velocity, dtype=np.float64),
            effort=np.asarray(message.effort, dtype=np.float64),
        ),
        False,
    )


# ---------------------------------------------------------------------------
# Jointcmd (header + fixed-size float64 array)
# ---------------------------------------------------------------------------


def parse_jointcmd_cdr(rawdata: bytes | memoryview, dim: int) -> tuple[int, np.ndarray]:
    """Parse a ``Header`` + ``float64[dim] positions`` command message."""
    reader = _CDRReader(rawdata)
    stamp_ns, _ = reader.header_stamp_ns()
    return stamp_ns, reader.float64_array(dim)


def parse_float32_cdr(rawdata: bytes | memoryview) -> float:
    """Parse a headerless ``std_msgs/Float32``."""
    return _CDRReader(rawdata).float32()


# ---------------------------------------------------------------------------
# Imagery
# ---------------------------------------------------------------------------


def _decode_raw_image(message: Any) -> np.ndarray:
    import cv2

    encoding = str(getattr(message, "encoding", "")).lower()
    width, height = int(message.width), int(message.height)
    step = int(getattr(message, "step", 0))
    if width <= 0 or height <= 0:
        raise MessageDecodeError("image has invalid dimensions")
    raw = bytes(message.data)
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
        raise MessageDecodeError(f"unsupported raw RGB encoding: {encoding}")
    return np.ascontiguousarray(image, dtype=np.uint8)


def _decode_compressed_image(message: Any) -> np.ndarray:
    import cv2

    fmt = str(getattr(message, "format", "")).lower()
    if "compresseddepth" in fmt.replace(" ", ""):
        raise MessageDecodeError(
            "compressedDepth streams are not supported; record depth as sensor_msgs/Image"
        )
    buffer = np.frombuffer(bytes(message.data), dtype=np.uint8)
    decoded = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if decoded is None:
        raise MessageDecodeError(f"cv2 could not decode CompressedImage (format={fmt!r})")
    return np.ascontiguousarray(cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB), dtype=np.uint8)


def decode_image(message: Any, msgtype: str) -> np.ndarray:
    """Decode either image message type to contiguous RGB uint8."""
    kind = image_kind(msgtype)
    return _decode_raw_image(message) if kind == RAW_IMAGE else _decode_compressed_image(message)


def decode_depth(message: Any, msgtype: str) -> np.ndarray:
    """Decode a depth image to float32 metres."""
    if msgtype in COMPRESSED_IMAGE_TYPES:
        raise MessageDecodeError(
            "compressed depth is not supported; record depth as sensor_msgs/Image"
        )
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
        raise MessageDecodeError(f"unsupported depth encoding: {encoding}")
    row_values = (step // np.dtype(dtype).itemsize) if step else width
    depth = np.frombuffer(bytes(message.data), dtype=dtype).reshape(height, row_values)[:, :width]
    depth = depth.astype(np.float32) * scale
    depth[~np.isfinite(depth)] = 0
    depth[depth < 0] = 0
    return np.ascontiguousarray(depth)


def resize_letterbox(array: np.ndarray, height: int, width: int, is_depth: bool = False) -> np.ndarray:
    """Scale to fit inside (height, width), padding the remainder with zeros."""
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
