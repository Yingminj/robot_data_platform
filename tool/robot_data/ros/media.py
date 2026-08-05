"""Image and depth decoding for rosbag conversion.

``sensor_msgs/Image`` and ``sensor_msgs/CompressedImage`` are both accepted and
normalised to RGB uint8; the format is detected from the connection's message
type rather than configured by hand, so a profile does not need to know whether
a given generation of the rig published raw or JPEG frames.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from robot_data.errors import MessageDecodeError

RAW_IMAGE_TYPES = frozenset({"sensor_msgs/msg/Image"})
COMPRESSED_IMAGE_TYPES = frozenset({"sensor_msgs/msg/CompressedImage"})
IMAGE_TYPES = RAW_IMAGE_TYPES | COMPRESSED_IMAGE_TYPES

RAW_IMAGE = "raw_image"
COMPRESSED_IMAGE = "compressed_image"


def image_kind(msgtype: str) -> str:
    """Classify a connection message type as raw or compressed imagery."""
    if msgtype in RAW_IMAGE_TYPES:
        return RAW_IMAGE
    if msgtype in COMPRESSED_IMAGE_TYPES:
        return COMPRESSED_IMAGE
    raise MessageDecodeError(f"not an image message type: {msgtype}")


def is_image_type(msgtype: str) -> bool:
    return msgtype in IMAGE_TYPES


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


def resize_letterbox(
    array: np.ndarray, height: int, width: int, is_depth: bool = False
) -> np.ndarray:
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
