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

Every profile lives in ``robot_data/profiles/builtin/*.json``; there are no
profiles defined in Python, so adding one never means editing this file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

from robot_data.errors import ProfileError

BUILTIN_DIR = Path(__file__).resolve().parent / "builtin"

GRIPPER = "gripper"
DEXHAND = "dexhand"
END_EFFECTOR_KINDS = (GRIPPER, DEXHAND)

FLOAT32 = "float32"
JOINTSTATE = "jointstate"
FLOAT32MULTIARRAY = "float32multiarray"
PAYLOAD_KINDS = (FLOAT32, JOINTSTATE, FLOAT32MULTIARRAY)

DEFAULT_PROFILE = "tj-dexhand"


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
    # Which components of a ``float32multiarray`` payload carry the effector's
    # position, in output order.  Gripper feedback topics publish a flat vector
    # (position, velocity, current, two temperatures ...) whose meaning is
    # positional, so the profile names the indices instead of guessing.
    state_indices: tuple[int, ...] = ()
    command_indices: tuple[int, ...] = ()

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
            raise ProfileError(
                f"end effector {self.name}: command_kind must be one of {PAYLOAD_KINDS}"
            )
        if self.state_kind not in PAYLOAD_KINDS:
            raise ProfileError(
                f"end effector {self.name}: state_kind must be one of {PAYLOAD_KINDS}"
            )
        if self.command_kind == FLOAT32 and self.dim != 1:
            raise ProfileError(f"end effector {self.name}: float32 payload implies dim 1")
        if self.joint_names and len(self.joint_names) != self.dim:
            raise ProfileError(
                f"end effector {self.name}: {len(self.joint_names)} joint names for dim {self.dim}"
            )
        for label, kind, indices in (
            ("state", self.state_kind if self.state_topic else None, self.state_indices),
            ("command", self.command_kind, self.command_indices),
        ):
            if kind == FLOAT32MULTIARRAY:
                if len(indices) != self.dim:
                    raise ProfileError(
                        f"end effector {self.name}: {label}_kind float32multiarray needs "
                        f"{self.dim} {label}_indices, got {len(indices)}"
                    )
                if any(index < 0 for index in indices):
                    raise ProfileError(
                        f"end effector {self.name}: {label}_indices must be non-negative"
                    )
            elif indices:
                raise ProfileError(
                    f"end effector {self.name}: {label}_indices only apply to float32multiarray"
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
class CameraTile:
    """One camera carved out of a stitched mosaic topic.

    Newer recordings publish a single composited image (``/quad_tile`` or
    ``/quad_tile/compressed``) instead of one topic per camera, so the cameras
    have to be recovered by cropping.  Bounds are *fractions* of the mosaic
    rather than pixel counts, which is how the deployment-side splitter does it
    (``vlahost/lebot_client.py:split_hero3_image``): the mosaic may be rescaled
    before it reaches a consumer, and a fractional spec survives that.

    ``width``/``height`` give the size each crop is resized to.  For the hero
    layout the wrist crops are already at their native size and the resize is a
    no-op, while the head crop is 2x oversized in the mosaic and is genuinely
    scaled back down.
    """

    left: float = 0.0
    top: float = 0.0
    right: float = 1.0
    bottom: float = 1.0
    width: int = 0
    height: int = 0

    def __post_init__(self) -> None:
        for field, value in (
            ("left", self.left),
            ("top", self.top),
            ("right", self.right),
            ("bottom", self.bottom),
        ):
            if not 0.0 <= value <= 1.0:
                raise ProfileError(f"camera tile {field} must be within [0, 1], got {value}")
        if self.left >= self.right:
            raise ProfileError(f"camera tile left {self.left} must be < right {self.right}")
        if self.top >= self.bottom:
            raise ProfileError(f"camera tile top {self.top} must be < bottom {self.bottom}")
        if bool(self.width) != bool(self.height):
            raise ProfileError(
                "camera tile width and height must both be zero or both be positive"
            )
        if self.width < 0 or self.height < 0:
            raise ProfileError("camera tile width and height must not be negative")

    def pixel_bounds(self, shape: tuple[int, ...]) -> tuple[int, int, int, int]:
        """Resolve the fractional crop against a concrete ``(height, width)``."""
        height, width = shape[0], shape[1]
        x0, x1 = int(round(self.left * width)), int(round(self.right * width))
        y0, y1 = int(round(self.top * height)), int(round(self.bottom * height))
        # Clamp so a degenerate mosaic yields an empty-but-valid slice error
        # rather than a silently transposed one.
        x0, x1 = max(0, min(x0, width - 1)), max(1, min(x1, width))
        y0, y1 = max(0, min(y0, height - 1)), max(1, min(y1, height))
        if x1 <= x0 or y1 <= y0:
            raise ProfileError(
                f"camera tile resolves to an empty crop on a {width}x{height} mosaic"
            )
        return x0, y0, x1, y1

    def apply(self, image: Any) -> Any:
        """Crop ``image`` and resize the result to the declared output size."""
        x0, y0, x1, y1 = self.pixel_bounds(image.shape)
        crop = image[y0:y1, x0:x1]
        if not self.width or crop.shape[:2] == (self.height, self.width):
            return crop
        import cv2

        # INTER_LINEAR mirrors the deployment splitter's `_resize_camera`, so
        # training frames and live inference frames go through the same filter.
        return cv2.resize(crop, (self.width, self.height), interpolation=cv2.INTER_LINEAR)


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
    # ``camera name -> CameraTile`` for cameras that share a stitched topic.
    # Empty for the one-topic-per-camera setups, where each camera is its own
    # whole frame.
    camera_tiles: dict[str, CameraTile] = None  # type: ignore[assignment]
    # ``typename -> .msg definition text`` for message types the bag does not
    # carry itself.  MCAP embeds its schemas, but rosbag2 sqlite3 (format
    # version 5) stores only type *names*, so custom types must be supplied
    # here or the bag cannot be opened at all.
    message_definitions: dict[str, str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        object.__setattr__(self, "cameras", dict(self.cameras or {}))
        object.__setattr__(self, "depths", dict(self.depths or {}))
        object.__setattr__(self, "camera_tiles", dict(self.camera_tiles or {}))
        object.__setattr__(self, "message_definitions", dict(self.message_definitions or {}))
        if not self.cameras:
            raise ProfileError(f"profile {self.name}: at least one camera is required")
        unknown_tiles = set(self.camera_tiles) - set(self.cameras)
        if unknown_tiles:
            raise ProfileError(
                f"profile {self.name}: camera_tiles for undeclared cameras {sorted(unknown_tiles)}"
            )
        # Several cameras may share one topic only when each says which slice of
        # it it owns; otherwise they would silently be the same image.
        shared: dict[str, list[str]] = {}
        for camera, topic in self.cameras.items():
            shared.setdefault(topic, []).append(camera)
        for topic, names in shared.items():
            if len(names) > 1 and not set(names) <= set(self.camera_tiles):
                raise ProfileError(
                    f"profile {self.name}: cameras {sorted(names)} share topic {topic} "
                    "but not all of them declare a camera_tiles entry"
                )
        for camera, topic in self.depths.items():
            if camera in self.camera_tiles:
                raise ProfileError(
                    f"profile {self.name}: camera {camera} is a mosaic tile and cannot carry depth"
                )
        names = [effector.name for effector in self.end_effectors]
        if len(set(names)) != len(names):
            raise ProfileError(f"profile {self.name}: duplicate end effector names {names}")
        for key, topic in {**self.cameras, **self.depths}.items():
            if not topic.startswith("/"):
                raise ProfileError(f"profile {self.name}: camera {key} topic must be absolute")
        extra = set(self.depths) - set(self.cameras)
        if extra:
            raise ProfileError(
                f"profile {self.name}: depth streams without RGB counterpart: {sorted(extra)}"
            )
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

    @property
    def essential_topics(self) -> set[str]:
        """Topics with no honest substitute if the recording lacks them.

        Everything else in :attr:`required_topics` -- the arm command topics and
        the end-effector streams -- can be reconstructed from measured state
        when the recording leaves it silent, which is what
        ``--missing-topic-policy fill`` does.  Observations cannot: there is
        nothing to fill an absent camera or an absent ``joint_states`` with
        except invented data.
        """
        return {self.arm.joint_states_topic, *self.cameras.values()}

    @property
    def optional_topics(self) -> set[str]:
        """Declared streams that merely enrich the output when present.

        An end-effector ``state_topic`` upgrades that effector's observation
        from a command echo to a real measurement.  Recording generations differ
        in whether they publish one, so its absence degrades to the documented
        ``command_echo`` behaviour instead of rejecting the bag.
        """
        return {effector.state_topic for effector in self.end_effectors if effector.state_topic}

    def topic_roles(self, include_depth: bool = True) -> dict[str, str]:
        """``topic -> role`` for every topic the profile names.

        The roles are the vocabulary both the quality checker and the topic
        inventory report in, so a topic is described the same way wherever it
        turns up.
        """
        roles: dict[str, str] = {}
        for topic in self.cameras.values():
            roles[topic] = "camera"
        if include_depth:
            for topic in self.depths.values():
                roles.setdefault(topic, "depth")
        roles[self.arm.joint_states_topic] = "arm_state"
        for topic in self.arm.command_topics:
            roles[topic] = "arm_command"
        for effector in self.end_effectors:
            roles[effector.command_topic] = "ee_command"
            if effector.state_topic:
                roles[effector.state_topic] = "ee_state"
        return roles

    def cameras_for_topic(self, topic: str) -> list[str]:
        """Camera names served by ``topic`` (several when it is a mosaic)."""
        return sorted(name for name, value in self.cameras.items() if value == topic)

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
                    "state_indices": list(effector.state_indices),
                    "command_indices": list(effector.command_indices),
                }
                for effector in self.end_effectors
            ],
            "cameras": dict(self.cameras),
            "depths": dict(self.depths),
            "camera_tiles": {
                name: {
                    "left": tile.left,
                    "top": tile.top,
                    "right": tile.right,
                    "bottom": tile.bottom,
                    "width": tile.width,
                    "height": tile.height,
                }
                for name, tile in self.camera_tiles.items()
            },
            "message_definitions": dict(self.message_definitions),
            "anchor_camera": self.resolved_anchor_camera,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "state_names": self.state_names,
        }


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


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
        state_indices=tuple(int(index) for index in payload.get("state_indices") or ()),
        command_indices=tuple(int(index) for index in payload.get("command_indices") or ()),
    )


def _tile_from_dict(payload: dict[str, Any]) -> CameraTile:
    return CameraTile(
        left=float(payload.get("left", 0.0)),
        top=float(payload.get("top", 0.0)),
        right=float(payload.get("right", 1.0)),
        bottom=float(payload.get("bottom", 1.0)),
        width=int(payload.get("width", 0)),
        height=int(payload.get("height", 0)),
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
            end_effectors=tuple(
                _spec_from_dict(item) for item in payload.get("end_effectors", ())
            ),
            cameras=dict(payload.get("cameras") or {}),
            depths=dict(payload.get("depths") or {}),
            camera_tiles={
                name: _tile_from_dict(spec)
                for name, spec in (payload.get("camera_tiles") or {}).items()
            },
            message_definitions=dict(payload.get("message_definitions") or {}),
            anchor_camera=payload.get("anchor_camera"),
        )
    except KeyError as exc:
        raise ProfileError(f"profile is missing required key: {exc}") from exc


def document_kind(payload: dict[str, Any]) -> str:
    """``"profile"``, ``"recipe"`` or ``"unknown"`` for a parsed JSON document.

    Profiles and recipes are both JSON, both carry a ``name``, and both are
    accepted as a bare path, so ``--recipe some-profile.json`` is an easy slip.
    Telling the two apart at load time turns an unknown-keys dump into a message
    that names the flag the file actually belongs to.  A recipe *references* a
    profile by name; only a profile *describes* one, which is what ``arm`` and
    ``end_effectors`` mark.
    """
    if "arm" in payload or "end_effectors" in payload:
        return "profile"
    if "profile" in payload or "storage" in payload or "alignment" in payload:
        return "recipe"
    return "unknown"


def builtin_profile_names() -> list[str]:
    """Names of the profiles shipped in ``profiles/builtin``."""
    return sorted(path.stem for path in BUILTIN_DIR.glob("*.json"))


@lru_cache(maxsize=None)
def _load_builtin(name: str) -> RobotProfile:
    path = BUILTIN_DIR / f"{name}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("_comment", None)
    declared = str(payload.get("name", name))
    if declared != name:
        # A built-in is selected by filename but reports itself by this field --
        # in the run banner, in `rdp profiles`, and in the dataset manifest.  A
        # profile copied from another one and not renamed would otherwise have
        # every converted dataset claim it came from the file it was copied from.
        raise ProfileError(
            f"built-in profile {path.name} declares name {declared!r}; "
            f"the name field must match the filename ({name!r}). "
            "This usually means the file was copied from another profile."
        )
    return profile_from_dict(payload)


def load_profile(name_or_path: str) -> RobotProfile:
    """Resolve a built-in profile name or a path to a JSON profile file."""
    if name_or_path in set(builtin_profile_names()):
        return _load_builtin(name_or_path)
    path = Path(name_or_path).expanduser()
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.pop("_comment", None)
        if document_kind(payload) == "recipe":
            raise ProfileError(
                f"{path} is a conversion recipe, not a robot profile. "
                "Pass it with --recipe instead of --profile."
            )
        return profile_from_dict(payload)
    raise ProfileError(
        f"unknown profile {name_or_path!r}; built-ins are {builtin_profile_names()} "
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
