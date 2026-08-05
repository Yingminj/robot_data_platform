"""Alignment configuration and the aligned-episode value type.

The invariant the whole pipeline is built around is one LeRobot-style control
row::

    observation at tick k -> teleop action produced immediately after it

Raw sensor capture time (ROS ``header.stamp``) is used in ``capture`` mode.
``lerobot-loop`` mode instead uses rosbag record time as the availability time,
which more closely models LeRobot's asynchronous "latest camera frame" reads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

ALIGNMENT_MODES = ("capture", "lerobot-loop")
INVALID_FRAME_POLICIES = ("fail", "drop")
ACTION_GAP_POLICIES = ("fail", "hold-last-command", "joint-state-fill")
MISSING_TOPIC_POLICIES = ("fail", "fill")
GRID_ANCHORS = ("anchor-camera", "anchor-camera-ticks", "first-command")


@dataclass(frozen=True)
class AlignmentConfig:
    fps: int = 30
    mode: str = "lerobot-loop"
    image_tolerance_ms: float | None = None
    state_tolerance_ms: float | None = None
    # Multiple of the measured joint_states period used when
    # ``state_tolerance_ms`` is left unset.  A relative bound is what the db3
    # recipes need: those recordings drop joint_states, so the age of the newest
    # sample before a tick spans several periods, but the periods themselves
    # differ per batch and an absolute millisecond figure would not transfer.
    state_tolerance_periods: float = 1.5
    action_tolerance_ms: float | None = None
    action_pair_tolerance_ms: float = 5.0
    end_effector_tolerance_ms: float = 100.0
    image_height: int = 0
    image_width: int = 0
    invalid_frame_policy: str = "fail"
    include_depth: bool = False
    max_decode_errors: int = 0
    action_gap_policy: str = "hold-last-command"
    missing_topic_policy: str = "fail"
    grid_anchor: str = "anchor-camera-ticks"
    max_hold_fraction: float | None = None
    max_hold_run_s: float | None = None
    max_tick_rate_deviation: float = 0.1

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.max_tick_rate_deviation < 0:
            raise ValueError("max_tick_rate_deviation must be non-negative")
        if self.mode not in set(ALIGNMENT_MODES):
            raise ValueError("mode must be capture or lerobot-loop")
        if self.invalid_frame_policy not in set(INVALID_FRAME_POLICIES):
            raise ValueError("invalid_frame_policy must be fail or drop")
        if self.action_gap_policy not in set(ACTION_GAP_POLICIES):
            raise ValueError(
                "action_gap_policy must be fail, hold-last-command or joint-state-fill"
            )
        if self.missing_topic_policy not in set(MISSING_TOPIC_POLICIES):
            raise ValueError("missing_topic_policy must be fail or fill")
        if self.missing_topic_policy == "fill" and self.action_gap_policy != "joint-state-fill":
            raise ValueError(
                "missing_topic_policy fill reconstructs a silent arm from its own measured "
                "joints, which is what action_gap_policy joint-state-fill does for gaps; "
                "set action_gap_policy to joint-state-fill"
            )
        if self.grid_anchor not in set(GRID_ANCHORS):
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
        if self.state_tolerance_periods <= 0:
            raise ValueError("state_tolerance_periods must be positive")
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
        return (
            self.action_tolerance_ms if self.action_tolerance_ms is not None else 1000.0 / self.fps
        )

    def resolve_state_tolerance_ms(self, source_period_ms: float) -> float:
        """Bound the age of the newest state sample preceding each tick.

        With "latest sample before tick" semantics that age is naturally spread
        over one source period, so a fixed default that happens to equal the
        publishing period rejects rows for ordinary jitter.  Default to
        ``state_tolerance_periods`` source periods unless the caller pins an
        absolute value.
        """
        if self.state_tolerance_ms is not None:
            return self.state_tolerance_ms
        return max(self.state_tolerance_periods * source_period_ms, 1.0)


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
