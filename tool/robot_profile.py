#!/usr/bin/env python3
"""Declarative robot/topic profiles for rosbag conversion.

A profile describes *what the robot is* -- which topics carry arm state and
commands, what end effectors are attached and how many degrees of freedom they
have, and which cameras exist.  Everything downstream (state/action dimensions,
LeRobot feature names, HDF5 layout) is derived from the profile rather than
hardcoded, so a new robot is a config change instead of a code change.

Two end-effector kinds are supported:

``gripper``
    A single scalar opening value per side.  Historically published as
    ``std_msgs/Float32`` with no header, so record time is the only clock.

``dexhand``
    A multi-DoF hand published as ``sensor_msgs/JointState``.  Both a measured
    state topic and a command topic may exist; when the state topic is absent
    the observation falls back to echoing the command, which is recorded in the
    conversion audit because it makes that part of the observation a copy of
    the action.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


class ProfileError(ValueError):
    """A robot profile is internally inconsistent or unusable."""


GRIPPER = "gripper"
DEXHAND = "dexhand"
END_EFFECTOR_KINDS = (GRIPPER, DEXHAND)

FLOAT32 = "float32"
JOINTSTATE = "jointstate"
PAYLOAD_KINDS = (FLOAT32, JOINTSTATE)


@dataclass(frozen=True)
class ArmSpec:
    """Arm joint state and the command topics that drive it."""

    joint_states_topic: str
    joint_names: tuple[str, ...]
    command_topics: tuple[str, ...]
    command_dim: int
    joint_state_order: str = "named"

    def __post_init__(self) -> None:
        if not self.joint_names:
            raise ProfileError("arm.joint_names must not be empty")
        if len(set(self.joint_names)) != len(self.joint_names):
            raise ProfileError("arm.joint_names contains duplicates")
        if not self.command_topics:
            raise ProfileError("arm.command_topics must not be empty")
        if len(set(self.command_topics)) != len(self.command_topics):
            raise ProfileError("arm.command_topics contains duplicates")
        if self.command_dim <= 0:
            raise ProfileError("arm.command_dim must be positive")
        if self.joint_state_order not in {"named", "positional"}:
            raise ProfileError("arm.joint_state_order must be named or positional")
        expected = self.command_dim * len(self.command_topics)
        if expected != len(self.joint_names):
            raise ProfileError(
                f"arm command width {expected} "
                f"({len(self.command_topics)} topics x {self.command_dim}) "
                f"does not match {len(self.joint_names)} joint names"
            )

    @property
    def dim(self) -> int:
        return len(self.joint_names)


@dataclass(frozen=True)
class EndEffectorSpec:
    """One end effector (one side of the robot)."""

    name: str
    kind: str
    dim: int
    command_topic: str
    command_kind: str
    state_topic: str | None = None
    state_kind: str = JOINTSTATE
    joint_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise ProfileError("end effector name must not be empty")
        if self.kind not in END_EFFECTOR_KINDS:
            raise ProfileError(f"end effector kind must be one of {END_EFFECTOR_KINDS}")
        if self.dim <= 0:
            raise ProfileError(f"end effector {self.name}: dim must be positive")
        if self.kind == GRIPPER and self.dim != 1:
            raise ProfileError(f"end effector {self.name}: gripper dim must be 1")
        if self.command_kind not in PAYLOAD_KINDS:
            raise ProfileError(f"end effector {self.name}: command_kind must be one of {PAYLOAD_KINDS}")
        if self.state_kind not in PAYLOAD_KINDS:
            raise ProfileError(f"end effector {self.name}: state_kind must be one of {PAYLOAD_KINDS}")
        if self.command_kind == FLOAT32 and self.dim != 1:
            raise ProfileError(f"end effector {self.name}: float32 payload implies dim 1")
        if self.joint_names and len(self.joint_names) != self.dim:
            raise ProfileError(
                f"end effector {self.name}: {len(self.joint_names)} joint names for dim {self.dim}"
            )

    @property
    def has_measured_state(self) -> bool:
        return self.state_topic is not None

    @property
    def feature_names(self) -> tuple[str, ...]:
        if self.joint_names:
            return self.joint_names
        if self.kind == GRIPPER:
            return (f"gripper_{self.name}",)
        return tuple(f"{self.name}_joint{index}" for index in range(1, self.dim + 1))


@dataclass(frozen=True)
class RobotProfile:
    """Complete description of one recording setup."""

    name: str
    robot_type: str
    arm: ArmSpec
    end_effectors: tuple[EndEffectorSpec, ...] = ()
    cameras: dict[str, str] = None  # type: ignore[assignment]
    depths: dict[str, str] = None  # type: ignore[assignment]
    anchor_camera: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "cameras", dict(self.cameras or {}))
        object.__setattr__(self, "depths", dict(self.depths or {}))
        if not self.cameras:
            raise ProfileError(f"profile {self.name}: at least one camera is required")
        names = [effector.name for effector in self.end_effectors]
        if len(set(names)) != len(names):
            raise ProfileError(f"profile {self.name}: duplicate end effector names {names}")
        for key, topic in {**self.cameras, **self.depths}.items():
            if not topic.startswith("/"):
                raise ProfileError(f"profile {self.name}: camera {key} topic must be absolute")
        extra = set(self.depths) - set(self.cameras)
        if extra:
            raise ProfileError(f"profile {self.name}: depth streams without RGB counterpart: {sorted(extra)}")
        if self.anchor_camera is not None and self.anchor_camera not in self.cameras:
            raise ProfileError(
                f"profile {self.name}: anchor_camera {self.anchor_camera!r} is not a declared camera"
            )

    # -- derived layout -------------------------------------------------

    @property
    def resolved_anchor_camera(self) -> str:
        if self.anchor_camera is not None:
            return self.anchor_camera
        return next(iter(self.cameras))

    @property
    def state_dim(self) -> int:
        return self.arm.dim + sum(effector.dim for effector in self.end_effectors)

    @property
    def action_dim(self) -> int:
        return self.state_dim

    @property
    def state_names(self) -> list[str]:
        names = list(self.arm.joint_names)
        for effector in self.end_effectors:
            names.extend(effector.feature_names)
        return names

    @property
    def action_names(self) -> list[str]:
        return self.state_names

    def state_slice(self, effector: EndEffectorSpec) -> tuple[int, int]:
        offset = self.arm.dim
        for candidate in self.end_effectors:
            if candidate.name == effector.name:
                return offset, offset + candidate.dim
            offset += candidate.dim
        raise ProfileError(f"unknown end effector {effector.name}")

    @property
    def required_topics(self) -> set[str]:
        topics = {self.arm.joint_states_topic, *self.arm.command_topics, *self.cameras.values()}
        for effector in self.end_effectors:
            topics.add(effector.command_topic)
            if effector.state_topic:
                topics.add(effector.state_topic)
        return topics

    def with_depth(self, include_depth: bool) -> RobotProfile:
        return self if include_depth else replace(self, depths={})

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "robot_type": self.robot_type,
            "arm": {
                "joint_states_topic": self.arm.joint_states_topic,
                "joint_names": list(self.arm.joint_names),
                "command_topics": list(self.arm.command_topics),
                "command_dim": self.arm.command_dim,
                "joint_state_order": self.arm.joint_state_order,
            },
            "end_effectors": [
                {
                    "name": effector.name,
                    "kind": effector.kind,
                    "dim": effector.dim,
                    "command_topic": effector.command_topic,
                    "command_kind": effector.command_kind,
                    "state_topic": effector.state_topic,
                    "state_kind": effector.state_kind,
                    "joint_names": list(effector.joint_names),
                }
                for effector in self.end_effectors
            ],
            "cameras": dict(self.cameras),
            "depths": dict(self.depths),
            "anchor_camera": self.resolved_anchor_camera,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "state_names": self.state_names,
        }


# ---------------------------------------------------------------------------
# Built-in profiles
# ---------------------------------------------------------------------------

_MARVIN_ARM_JOINTS = tuple(f"Joint{index}_{side}" for side in ("L", "R") for index in range(1, 8))

_DEXHAND_JOINTS = tuple(
    f"{side}_finger{finger}_joint{joint}"
    for side in ("left", "right")
    for finger in range(1, 6)
    for joint in range(1, 5)
)


def _dexhand_names(side: str) -> tuple[str, ...]:
    return tuple(name for name in _DEXHAND_JOINTS if name.startswith(f"{side}_"))


BUILTIN_PROFILES: dict[str, RobotProfile] = {
    # Legacy setup: raw sensor_msgs/Image cameras, scalar Float32 grippers.
    "marvin-gripper": RobotProfile(
        name="marvin-gripper",
        robot_type="marvin",
        arm=ArmSpec(
            joint_states_topic="/joint_states",
            joint_names=_MARVIN_ARM_JOINTS,
            command_topics=("/control/joint_cmd_A", "/control/joint_cmd_B"),
            command_dim=7,
        ),
        end_effectors=(
            EndEffectorSpec(
                name="L",
                kind=GRIPPER,
                dim=1,
                command_topic="/control/gripperValueL",
                command_kind=FLOAT32,
                state_topic=None,
            ),
            EndEffectorSpec(
                name="R",
                kind=GRIPPER,
                dim=1,
                command_topic="/control/gripperValueR",
                command_kind=FLOAT32,
                state_topic=None,
            ),
        ),
        cameras={
            "top": "/camera_head/camera_head/color/image_raw",
            "wrist_L": "/camera_left_wrist/camera_left_wrist/color/image_rect_raw",
            "wrist_R": "/camera_right_wrist/camera_right_wrist/color/image_rect_raw",
        },
        depths={
            "top": "/camera_head/camera_head/aligned_depth_to_color/image_raw",
            "wrist_L": "/camera_left_wrist/camera_left_wrist/aligned_depth_to_color/image_raw",
            "wrist_R": "/camera_right_wrist/camera_right_wrist/aligned_depth_to_color/image_raw",
        },
        anchor_camera="top",
    ),
    # Current MCAP setup: compressed JPEG cameras, 20-DoF dexterous hands.
    "tj-dexhand": RobotProfile(
        name="tj-dexhand",
        robot_type="marvin",
        arm=ArmSpec(
            joint_states_topic="/tj/joint_states",
            joint_names=_MARVIN_ARM_JOINTS,
            command_topics=("/tj/control/joint_cmd_A", "/tj/control/joint_cmd_B"),
            command_dim=7,
        ),
        end_effectors=(
            EndEffectorSpec(
                name="left_hand",
                kind=DEXHAND,
                dim=20,
                command_topic="/hand_left/joint_commands",
                command_kind=JOINTSTATE,
                # /hand_left/joint_states is absent from current recordings.
                state_topic=None,
                joint_names=_dexhand_names("left"),
            ),
            EndEffectorSpec(
                name="right_hand",
                kind=DEXHAND,
                dim=20,
                command_topic="/hand_right/joint_commands",
                command_kind=JOINTSTATE,
                state_topic="/hand_right/joint_states",
                joint_names=_dexhand_names("right"),
            ),
        ),
        cameras={
            "top": "/head_camera/camera/color/image_raw/compressed",
            "wrist_L": "/wrist_left_camera/camera/color/image_raw/compressed",
            "wrist_R": "/wrist_right_camera/camera/color/image_raw/compressed",
        },
        depths={},
        anchor_camera="top",
    ),
}

DEFAULT_PROFILE = "tj-dexhand"


def _spec_from_dict(payload: dict[str, Any]) -> EndEffectorSpec:
    return EndEffectorSpec(
        name=str(payload["name"]),
        kind=str(payload.get("kind", GRIPPER)),
        dim=int(payload["dim"]),
        command_topic=str(payload["command_topic"]),
        command_kind=str(payload.get("command_kind", JOINTSTATE)),
        state_topic=payload.get("state_topic") or None,
        state_kind=str(payload.get("state_kind", JOINTSTATE)),
        joint_names=tuple(payload.get("joint_names") or ()),
    )


def profile_from_dict(payload: dict[str, Any]) -> RobotProfile:
    """Build a profile from a plain dict, e.g. parsed JSON."""
    try:
        arm_payload = payload["arm"]
        arm = ArmSpec(
            joint_states_topic=str(arm_payload["joint_states_topic"]),
            joint_names=tuple(arm_payload["joint_names"]),
            command_topics=tuple(arm_payload["command_topics"]),
            command_dim=int(arm_payload["command_dim"]),
            joint_state_order=str(arm_payload.get("joint_state_order", "named")),
        )
        return RobotProfile(
            name=str(payload.get("name", "custom")),
            robot_type=str(payload.get("robot_type", "custom")),
            arm=arm,
            end_effectors=tuple(_spec_from_dict(item) for item in payload.get("end_effectors", ())),
            cameras=dict(payload.get("cameras") or {}),
            depths=dict(payload.get("depths") or {}),
            anchor_camera=payload.get("anchor_camera"),
        )
    except KeyError as exc:
        raise ProfileError(f"profile is missing required key: {exc}") from exc


def load_profile(name_or_path: str) -> RobotProfile:
    """Resolve a built-in profile name or a path to a JSON profile file."""
    if name_or_path in BUILTIN_PROFILES:
        return BUILTIN_PROFILES[name_or_path]
    path = Path(name_or_path).expanduser()
    if path.is_file():
        return profile_from_dict(json.loads(path.read_text(encoding="utf-8")))
    raise ProfileError(
        f"unknown profile {name_or_path!r}; built-ins are {sorted(BUILTIN_PROFILES)} "
        "or pass a path to a JSON profile"
    )


def parse_name_topic(items: list[str] | None) -> dict[str, str] | None:
    """Parse repeated ``NAME=TOPIC`` CLI arguments; ``None`` when unspecified."""
    if not items:
        return None
    output: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ProfileError(f"Expected NAME=TOPIC, got {item!r}")
        name, topic = (part.strip() for part in item.split("=", 1))
        if not name or not topic.startswith("/"):
            raise ProfileError(f"Invalid NAME=TOPIC mapping: {item!r}")
        if name in output:
            raise ProfileError(f"Duplicate camera/depth name: {name}")
        output[name] = topic
    return output


def apply_topic_overrides(
    profile: RobotProfile,
    cameras: dict[str, str] | None = None,
    depths: dict[str, str] | None = None,
    anchor_camera: str | None = None,
) -> RobotProfile:
    """Return a copy of ``profile`` with camera mappings replaced."""
    updates: dict[str, Any] = {}
    if cameras:
        updates["cameras"] = dict(cameras)
    if depths is not None:
        updates["depths"] = dict(depths)
    if anchor_camera:
        updates["anchor_camera"] = anchor_camera
    if not updates:
        return profile
    if "cameras" in updates and profile.anchor_camera not in updates["cameras"]:
        updates.setdefault("anchor_camera", next(iter(updates["cameras"])))
    if "cameras" in updates and "depths" not in updates:
        updates["depths"] = {
            name: topic for name, topic in profile.depths.items() if name in updates["cameras"]
        }
    return replace(profile, **updates)
