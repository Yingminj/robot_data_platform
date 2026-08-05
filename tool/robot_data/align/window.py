"""Choosing which slice of a recording becomes an episode.

Episode windowing follows teleoperation activity rather than the bag extent:
the usable range runs from the first arm command to the last arm command, and
the grid is anchored to the newest anchor-camera frame at or before that first
command, so row 0 starts on a fresh image at the moment teleoperation begins.
Everything outside that range is discarded.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robot_data.align.bag_io import RawBag
from robot_data.align.config import AlignmentConfig
from robot_data.errors import ConversionError


@dataclass(frozen=True)
class Window:
    start_ns: int
    end_ns: int
    command_start_ns: int
    command_end_ns: int
    anchor_ns: int
    bag_start_ns: int
    bag_end_ns: int


def compute_window(raw: RawBag, cfg: AlignmentConfig) -> Window:
    """Teleoperation-activity window: first arm command to last arm command."""
    profile = raw.profile
    # An arm whose command topic never spoke has no span to contribute; it is
    # reconstructed from its own measured joints for the whole episode, exactly
    # as a silent stretch inside a bag would be.
    live_command_topics = [t for t in profile.arm.command_topics if raw.has_arm_command(t)]
    firsts = [int(raw.cmd_t[topic][0]) for topic in live_command_topics]
    lasts = [int(raw.cmd_t[topic][-1]) for topic in live_command_topics]
    if cfg.action_gap_policy == "joint-state-fill":
        # A silent arm is filled from its own measured joints, which exist for
        # the whole bag, so every row is defined as soon as *any* arm is being
        # driven: take the union of the arms' command spans.  Sequential teleop
        # -- drive one arm, then the other -- leaves the intersection empty even
        # though the episode is perfectly usable.
        command_start, command_end = min(firsts), max(lasts)
    else:
        # Holding the last command needs one from every arm, so the window can
        # only start once each arm has spoken and must end while all are still
        # live: the intersection of the command spans.
        command_start, command_end = max(firsts), min(lasts)
    if command_end <= command_start:
        if cfg.action_gap_policy == "joint-state-fill":
            raise ConversionError("Arm command topics carry no usable time span")
        raise ConversionError(
            "Arm command topics do not overlap in time; for sequential or "
            "alternating single-arm teleop use --action-gap-policy joint-state-fill"
        )

    observation_times: list[np.ndarray] = [
        raw.arm_t,
        *raw.image_t.values(),
        *raw.depth_t.values(),
    ]
    for effector in profile.end_effectors:
        # Whichever stream actually supplies this effector's observation is the
        # one the window has to stay inside: a measurement when the recording
        # has one, the command echo otherwise.
        if raw.has_effector_state(effector):
            observation_times.append(raw.ee_state_t[effector.name])
        elif raw.has_effector_command(effector):
            observation_times.append(raw.ee_cmd_t[effector.name])

    command_times = [raw.cmd_t[topic] for topic in live_command_topics]
    bag_start = min(int(values[0]) for values in [*observation_times, *command_times])
    bag_end = max(int(values[-1]) for values in [*observation_times, *command_times])

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
    return Window(
        start_ns=start,
        end_ns=end,
        command_start_ns=command_start,
        command_end_ns=command_end,
        anchor_ns=anchor,
        bag_start_ns=bag_start,
        bag_end_ns=bag_end,
    )
