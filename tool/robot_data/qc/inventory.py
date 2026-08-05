"""What a recording actually contains, next to what the profile expects.

Conversion failures in this pipeline are overwhelmingly one mistake: the profile
names a camera topic the recording does not publish, or publishes under a
slightly different name (``/quad_tile`` vs ``/quad_tile/compressed``,
``/joint_states`` vs ``/joint_states``).  The converter can only report that
a required topic is missing; it cannot say what the bag has *instead*.

This module answers that.  It lists every topic in the recording, marks which
ones the profile claims and in what role, and flags the two mismatches worth
interrupting for: a claimed topic that is absent or empty, and an unclaimed
image topic -- almost always the camera the profile should have pointed at.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from robot_data.align.bag_io import BagScan
from robot_data.profiles.schema import RobotProfile
from robot_data.ros.media import IMAGE_TYPES

# What the inventory concluded about one topic.
CLAIMED = "claimed"  # the profile names it and it carries messages
CLAIMED_EMPTY = "claimed-empty"  # the profile names it, advertised but silent
CLAIMED_ABSENT = "claimed-absent"  # the profile names it, not in the bag at all
UNCLAIMED = "unclaimed"  # present in the bag, not named by the profile


@dataclass
class TopicRow:
    topic: str
    status: str
    role: str
    msgtype: str
    count: int
    hz: float | None
    duration_s: float | None
    cameras: list[str] = field(default_factory=list)

    @property
    def is_image(self) -> bool:
        return self.msgtype in IMAGE_TYPES

    def to_dict(self) -> dict[str, Any]:
        return {
            "topic": self.topic,
            "status": self.status,
            "role": self.role,
            "msgtype": self.msgtype,
            "count": self.count,
            "hz": self.hz,
            "duration_s": self.duration_s,
            "cameras": list(self.cameras),
        }


@dataclass
class Inventory:
    rows: list[TopicRow]
    profile_name: str
    problems: list[str]
    suggestions: list[str]

    @property
    def matches(self) -> bool:
        """Whether every topic the profile needs is present and non-empty."""
        return not self.problems

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile_name,
            "matches_profile": self.matches,
            "problems": list(self.problems),
            "suggestions": list(self.suggestions),
            "topics": [row.to_dict() for row in self.rows],
        }


def _rate(times: list[int]) -> tuple[float | None, float | None]:
    if len(times) < 2:
        return None, None
    span_s = (times[-1] - times[0]) / 1e9
    if span_s <= 0:
        return None, 0.0
    return (len(times) - 1) / span_s, span_s


def build_inventory(
    scan: BagScan, profile: RobotProfile, include_depth: bool = False
) -> Inventory:
    """Cross the bag's real topic list against what ``profile`` names."""
    roles = profile.topic_roles(include_depth=include_depth)
    essential = profile.essential_topics
    optional = profile.optional_topics
    rows: list[TopicRow] = []

    for topic in sorted(set(roles) | set(scan.topics)):
        seen = scan.topics.get(topic)
        role = roles.get(topic, "-")
        cameras = profile.cameras_for_topic(topic)
        if seen is None:
            rows.append(
                TopicRow(topic, CLAIMED_ABSENT, role, "-", 0, None, None, cameras)
            )
            continue
        hz, duration = _rate(seen.receive_ns)
        if topic not in roles:
            status = UNCLAIMED
        elif seen.count == 0:
            status = CLAIMED_EMPTY
        else:
            status = CLAIMED
        rows.append(
            TopicRow(topic, status, role, seen.msgtype, seen.count, hz, duration, cameras)
        )

    # Order so the reader sees the profile's own topics first, then whatever
    # else the recording carries.
    order = {CLAIMED: 0, CLAIMED_EMPTY: 1, CLAIMED_ABSENT: 2, UNCLAIMED: 3}
    rows.sort(key=lambda row: (order[row.status], row.role, row.topic))

    problems: list[str] = []
    suggestions: list[str] = []
    broken = [row for row in rows if row.status in {CLAIMED_ABSENT, CLAIMED_EMPTY}]
    spare_images = [row for row in rows if row.status == UNCLAIMED and row.is_image]

    for row in broken:
        if row.topic in optional:
            # A measured end-effector feedback topic only enriches the
            # observation; the converter degrades to a command echo instead of
            # rejecting, so this is worth stating but is not a mismatch.
            continue
        where = "not advertised by the recording" if row.status == CLAIMED_ABSENT else "advertised but empty"
        detail = f" (cameras {', '.join(row.cameras)})" if row.cameras else ""
        problems.append(f"{row.role} topic {row.topic} is {where}{detail}")
        if row.topic in essential and row.role == "camera" and spare_images:
            names = ", ".join(f"{item.topic} ({item.count} msgs)" for item in spare_images)
            suggestions.append(
                f"the recording does carry image topics the profile does not use: {names}"
            )
        elif row.role in {"arm_command", "ee_command"}:
            suggestions.append(
                f"{row.topic} can be reconstructed from measured state with "
                "--missing-topic-policy fill --action-gap-policy joint-state-fill "
                "(those action columns become an identity copy of the observation)"
            )

    if spare_images and not any(row.role == "camera" for row in broken):
        names = ", ".join(item.topic for item in spare_images)
        suggestions.append(
            f"image topics present but unused by this profile: {names} "
            "-- confirm you are converting the camera set you intend to train on"
        )

    # Deduplicate while keeping the order the problems were found in.
    suggestions = list(dict.fromkeys(suggestions))
    return Inventory(
        rows=rows, profile_name=profile.name, problems=problems, suggestions=suggestions
    )


_STATUS_LABEL = {
    CLAIMED: "使用",
    CLAIMED_EMPTY: "为空",
    CLAIMED_ABSENT: "缺失",
    UNCLAIMED: "未用",
}


def print_inventory(inventory: Inventory, recipe_name: str | None = None) -> None:
    """Print the topic table plus the profile-match verdict."""
    header = f"话题清单（profile {inventory.profile_name}"
    if recipe_name:
        header += f"，recipe {recipe_name}"
    print(header + "）：")
    print(f"  {'状态':<6}{'角色':<13}{'数量':>8}  {'实测Hz':>8}  {'时长s':>8}  {'消息类型':<38}话题")
    for row in inventory.rows:
        hz = "-" if row.hz is None else f"{row.hz:.1f}"
        duration = "-" if row.duration_s is None else f"{row.duration_s:.1f}"
        label = _STATUS_LABEL[row.status]
        suffix = f"   -> {', '.join(row.cameras)}" if row.cameras else ""
        print(
            f"  {label:<6}{row.role:<13}{row.count:>8}  {hz:>8}  {duration:>8}  "
            f"{row.msgtype:<38}{row.topic}{suffix}"
        )

    if inventory.matches:
        print("\n  ✅ profile 与录制内容一致：所需话题全部存在且非空。")
    else:
        print("\n  ❌ profile 与录制内容不匹配：")
        for problem in inventory.problems:
            print(f"     - {problem}")
    for suggestion in inventory.suggestions:
        print(f"     提示：{suggestion}")


def camera_geometry(
    scan: BagScan, profile: RobotProfile
) -> dict[str, dict[str, Any]]:
    """Decode one frame per camera topic to report its real pixel size.

    A mosaic profile crops fractionally, so the only way to know whether the
    tiles land where the profile says they do is to look at the frame the
    recording actually contains.
    """
    from robot_data.align.bag_io import open_bag_reader
    from robot_data.ros.media import decode_image

    wanted = {topic for topic in profile.cameras.values() if topic in scan.topics}
    result: dict[str, dict[str, Any]] = {}
    if not wanted:
        return result
    reader = open_bag_reader(scan.bag_dir, profile)
    reader.open()
    try:
        connections = [c for c in reader.connections if c.topic in wanted]
        remaining = set(wanted)
        for connection, _, rawdata in reader.messages(connections=connections):
            if connection.topic not in remaining:
                continue
            entry: dict[str, Any] = {"msgtype": connection.msgtype}
            try:
                message = reader.deserialize(rawdata, connection.msgtype)
                frame = decode_image(message, connection.msgtype)
                entry["height"], entry["width"] = int(frame.shape[0]), int(frame.shape[1])
                entry["tiles"] = {
                    name: dict(
                        zip(
                            ("x0", "y0", "x1", "y1"),
                            profile.camera_tiles[name].pixel_bounds(frame.shape),
                        )
                    )
                    | {
                        "output": [
                            profile.camera_tiles[name].height,
                            profile.camera_tiles[name].width,
                        ]
                    }
                    for name in profile.cameras_for_topic(connection.topic)
                    if name in profile.camera_tiles
                }
            except Exception as exc:
                entry["error"] = str(exc)
            result[connection.topic] = entry
            remaining.discard(connection.topic)
            if not remaining:
                break
    finally:
        reader.close()
    return result


def print_camera_geometry(
    geometry: dict[str, dict[str, Any]], profile: RobotProfile
) -> None:
    if not geometry:
        return
    print("\n相机分辨率与裁切（各解码一帧）：")
    for topic in sorted(geometry):
        entry = geometry[topic]
        names = ", ".join(profile.cameras_for_topic(topic)) or "-"
        if "error" in entry:
            print(f"  {topic} -> {names}: 解码失败：{entry['error']}")
            continue
        print(f"  {topic} -> {names}: {entry['width']}x{entry['height']} ({entry['msgtype']})")
        for name, tile in (entry.get("tiles") or {}).items():
            height, width = tile["output"]
            crop_w, crop_h = tile["x1"] - tile["x0"], tile["y1"] - tile["y0"]
            resize = (
                "原尺寸" if (crop_h, crop_w) == (height, width) else f"缩放到 {width}x{height}"
            )
            print(
                f"      {name:<8} 裁切 x[{tile['x0']}:{tile['x1']}] y[{tile['y0']}:{tile['y1']}]"
                f" = {crop_w}x{crop_h}，{resize}"
            )


def summarise_hdf5_layout(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Group an HDF5 layout listing by top-level branch, for the check report."""
    totals: dict[str, dict[str, Any]] = {}
    for entry in entries:
        branch = entry["path"].split("/")[0]
        bucket = totals.setdefault(branch, {"datasets": 0, "nbytes": 0, "storage_bytes": 0})
        bucket["datasets"] += 1
        bucket["nbytes"] += int(entry["nbytes"])
        bucket["storage_bytes"] += int(entry["storage_bytes"])
    return totals


def unique_frame_ratio(values: np.ndarray) -> float:
    return float(len(np.unique(values)) / len(values)) if len(values) else 0.0
