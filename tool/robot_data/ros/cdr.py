"""Tolerant CDR decoding for the non-image ROS 2 messages in a recording.

``sensor_msgs/JointState`` messages produced by some publishers carry trailing
bytes left over from a reused serialisation buffer.  ROS 2's own deserialiser
ignores them, but ``rosbags`` asserts on unconsumed input.  A tolerant CDR
reader is used as a fallback so those topics remain readable, and the number of
fallbacks is reported so the underlying recorder bug stays visible instead of
being silently absorbed.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any

import numpy as np

from robot_data.errors import MessageDecodeError

# CDR alignment is measured from the end of the 4-byte encapsulation header.
_CDR_ORIGIN = 4


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
        values = np.frombuffer(
            self.data, dtype=np.dtype(self.endian + "f8"), count=count, offset=self.pos
        )
        self.pos += count * 8
        return np.asarray(values, dtype=np.float64)

    def float32_sequence(self) -> np.ndarray:
        count = self.uint32()
        if count == 0:
            return np.empty(0, dtype=np.float64)
        self.align(4)
        if count * 4 > self.remaining:
            raise MessageDecodeError(
                f"CDR sequence declares {count} float32 but only {self.remaining // 4} remain"
            )
        values = np.frombuffer(
            self.data, dtype=np.dtype(self.endian + "f4"), count=count, offset=self.pos
        )
        self.pos += count * 4
        return np.asarray(values, dtype=np.float64)

    def float64_array(self, count: int) -> np.ndarray:
        """Fixed-size array: no length prefix, elements aligned to 8 bytes."""
        if count == 0:
            return np.empty(0, dtype=np.float64)
        self.align(8)
        if count * 8 > self.remaining:
            raise MessageDecodeError("CDR fixed array exceeds payload")
        values = np.frombuffer(
            self.data, dtype=np.dtype(self.endian + "f8"), count=count, offset=self.pos
        )
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


def read_jointstate(
    reader: Any, rawdata: bytes | memoryview, msgtype: str
) -> tuple[JointStateMessage, bool]:
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
# Jointcmd (header + fixed-size float64 array) and headerless std_msgs
# ---------------------------------------------------------------------------


def parse_jointcmd_cdr(rawdata: bytes | memoryview, dim: int) -> tuple[int, np.ndarray]:
    """Parse a ``Header`` + ``float64[dim] positions`` command message."""
    reader = _CDRReader(rawdata)
    stamp_ns, _ = reader.header_stamp_ns()
    return stamp_ns, reader.float64_array(dim)


def parse_float32_cdr(rawdata: bytes | memoryview) -> float:
    """Parse a headerless ``std_msgs/Float32``."""
    return _CDRReader(rawdata).float32()


def parse_float32multiarray_cdr(rawdata: bytes | memoryview) -> np.ndarray:
    """Parse a headerless ``std_msgs/Float32MultiArray`` down to its ``data``.

    The ``layout`` prefix is read and discarded: gripper feedback topics publish
    a flat vector whose meaning is positional, so the profile picks components
    by index rather than by the (usually empty) dimension labels.
    """
    reader = _CDRReader(rawdata)
    for _ in range(reader.uint32()):  # layout.dim[]
        reader.string()  # label
        reader.uint32()  # size
        reader.uint32()  # stride
    reader.uint32()  # layout.data_offset
    return reader.float32_sequence()
