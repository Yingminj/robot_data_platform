"""Turning a scanned bag into strict, fixed-rate LeRobot-style control rows.

Gaps inside the teleoperation window are filled by holding the last issued
command (zero-order hold), which is what a real controller does when commands
stop arriving; held rows are counted and can be rejected via
``max_hold_fraction`` / ``max_hold_run_s``.

State and action dimensions come from the robot profile, not from constants, so
grippers and multi-DoF dexterous hands share one code path.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from robot_data.align.bag_io import RawBag, scan_bag
from robot_data.align.config import AlignedEpisode, AlignmentConfig
from robot_data.align.indexing import (
    backward_difference,
    interpolate,
    nearest_indices,
    next_indices,
    previous_indices,
    runs,
    stats_ms,
)
from robot_data.align.media import decode_selected_media
from robot_data.align.window import compute_window
from robot_data.errors import ConversionError
from robot_data.profiles.schema import EndEffectorSpec, RobotProfile

# Where an end effector's observation or action actually came from.
COMMAND_ECHO = "command_echo"
MEASURED = "measured"
STATE_FILL = "state_fill"


def _select_command(
    times: np.ndarray, values: np.ndarray, grid_ns: np.ndarray, action_tol: int, policy: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pick the command for each tick, holding the last one across gaps.

    Returns (indices, lead_ns, real_mask, valid_mask).
    """
    next_idx, lead, next_ok = next_indices(times, grid_ns)
    real = next_ok & (lead <= action_tol)
    if policy == "fail":
        return next_idx, lead, real, real

    previous_idx, _, previous_ok = previous_indices(times, grid_ns)
    # Inside a gap hold the last issued command; before the first command
    # (possible when the grid is anchored to a camera frame) use the first one.
    indices = np.where(real, next_idx, np.where(previous_ok, previous_idx, 0))
    return indices, lead, real, np.ones_like(real, dtype=bool)


def _fill_source(topic: str, profile: RobotProfile) -> str:
    """What stood in for ``topic``, for the per-episode audit."""
    if topic in profile.arm.command_topics:
        return f"{profile.arm.joint_states_topic} (that arm's measured joints)"
    for effector in profile.end_effectors:
        if topic == effector.command_topic:
            return f"{effector.state_topic} (measured position)"
        if topic == effector.state_topic:
            return f"{effector.command_topic} (command echo)"
    return "unknown"


def _effector_observation(
    effector: EndEffectorSpec,
    raw: RawBag,
    grid_ns: np.ndarray,
    tolerance_ns: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Observed end-effector state; echoes the command when unmeasured.

    ``has_measured_state`` is what the profile declares; ``has_effector_state``
    is what this particular recording delivered.  Generations differ, so a
    declared-but-absent feedback topic degrades to the command echo rather than
    rejecting the bag.
    """
    if raw.has_effector_state(effector):
        times = raw.ee_state_t[effector.name]
        values = raw.ee_state_v[effector.name]
        source = MEASURED
    elif raw.has_effector_command(effector):
        times = raw.ee_cmd_t[effector.name]
        values = raw.ee_cmd_v[effector.name]
        source = COMMAND_ECHO
    else:
        raise ConversionError(
            f"end effector {effector.name}: neither {effector.command_topic} nor a measured "
            "state topic carries data, so its observation cannot be reconstructed"
        )
    indices, age, valid = previous_indices(times, grid_ns)
    valid &= age <= tolerance_ns
    return values[indices], times[indices], valid, source


def align_rosbag(bag_dir: Path, profile: RobotProfile, cfg: AlignmentConfig) -> AlignedEpisode:
    """Convert one rosbag into strict, fixed-rate LeRobot-style control rows."""
    raw = scan_bag(bag_dir.expanduser().resolve(), profile, cfg)
    profile = raw.profile
    window = compute_window(raw, cfg)

    if cfg.grid_anchor == "anchor-camera-ticks":
        # One row per anchor-camera frame: the observation image is then always
        # exactly fresh instead of up to a full camera period stale, which is
        # what "latest frame before tick" costs when the camera rate and the
        # requested fps are nearly equal.
        anchor_times = raw.image_t[profile.resolved_anchor_camera]
        grid_ns = anchor_times[
            (anchor_times >= window.start_ns) & (anchor_times <= window.end_ns)
        ].astype(np.int64)
        frame_count = int(grid_ns.size)
    else:
        frame_count = int(math.floor((window.end_ns - window.start_ns) * cfg.fps / 1e9)) + 1
        grid_ns = window.start_ns + np.rint(np.arange(frame_count) * 1e9 / cfg.fps).astype(
            np.int64
        )
    if frame_count < 2:
        raise ConversionError(f"Teleoperation window produces only {frame_count} frame(s)")

    # With anchor-camera ticks the row spacing is the camera's real period, but
    # LeRobot derives its timestamp column as frame_index / fps.  If the camera
    # does not actually run at --fps the dataset clock silently drifts, so make
    # the mismatch an error rather than a property of the output.
    tick_period_ms = float(np.median(np.diff(grid_ns))) / 1e6 if grid_ns.size > 1 else 0.0
    measured_fps = 1000.0 / tick_period_ms if tick_period_ms > 0 else 0.0
    if cfg.grid_anchor == "anchor-camera-ticks" and measured_fps > 0:
        deviation = abs(measured_fps - cfg.fps) / cfg.fps
        if deviation > cfg.max_tick_rate_deviation:
            raise ConversionError(
                f"anchor camera {profile.resolved_anchor_camera!r} ticks at "
                f"{measured_fps:.2f} Hz but --fps is {cfg.fps} "
                f"({100 * deviation:.1f}% off): the dataset timestamp column would not match "
                f"the real row spacing. Set --fps {round(measured_fps)}, pick another "
                f"--anchor-camera, or use --grid-anchor anchor-camera for a true fixed-rate grid "
                f"(raise --max-tick-rate-deviation to accept the mismatch)."
            )

    arm_period_ms = (
        float(np.median(np.diff(raw.arm_t))) / 1e6 if raw.arm_t.size > 1 else 1000.0 / cfg.fps
    )
    state_tolerance_ms = cfg.resolve_state_tolerance_ms(arm_period_ms)

    image_tol = round(cfg.resolved_image_tolerance_ms * 1e6)
    state_tol = round(state_tolerance_ms * 1e6)
    action_tol = round(cfg.resolved_action_tolerance_ms * 1e6)
    pair_tol = round(cfg.action_pair_tolerance_ms * 1e6)
    effector_tol = round(cfg.end_effector_tolerance_ms * 1e6)

    valid_parts: dict[str, np.ndarray] = {}

    # -- imagery ---------------------------------------------------------
    image_indices: dict[str, np.ndarray] = {}
    image_offsets: dict[str, np.ndarray] = {}
    for name, source_t in raw.image_t.items():
        if cfg.mode == "capture":
            indices, offsets = nearest_indices(source_t, grid_ns)
            valid = offsets <= image_tol
        else:
            indices, offsets, valid = previous_indices(source_t, grid_ns)
            valid &= offsets <= image_tol
        image_indices[name] = indices
        image_offsets[name] = offsets
        valid_parts[f"image:{name}"] = valid

    depth_indices: dict[str, np.ndarray] = {}
    depth_offsets: dict[str, np.ndarray] = {}
    for name, source_t in raw.depth_t.items():
        if cfg.mode == "capture":
            indices, offsets = nearest_indices(source_t, grid_ns)
            valid = offsets <= image_tol
        else:
            indices, offsets, valid = previous_indices(source_t, grid_ns)
            valid &= offsets <= image_tol
        depth_indices[name] = indices
        depth_offsets[name] = offsets
        valid_parts[f"depth:{name}"] = valid

    # -- arm state -------------------------------------------------------
    if cfg.mode == "capture":
        arm_qpos, arm_state_t, arm_valid = interpolate(raw.arm_t, raw.arm_pos, grid_ns, state_tol)
        if raw.arm_vel.size and np.all(np.isfinite(raw.arm_vel)):
            arm_qvel, _, velocity_valid = interpolate(raw.arm_t, raw.arm_vel, grid_ns, state_tol)
            arm_valid &= velocity_valid
        else:
            arm_qvel = np.full_like(arm_qpos, np.nan)
    else:
        arm_idx, arm_age, arm_valid = previous_indices(raw.arm_t, grid_ns)
        arm_valid &= arm_age <= state_tol
        arm_qpos = raw.arm_pos[arm_idx]
        arm_state_t = raw.arm_t[arm_idx]
        arm_qvel = raw.arm_vel[arm_idx] if raw.arm_vel.size else np.full_like(arm_qpos, np.nan)
    valid_parts["arm_state"] = arm_valid

    # -- end-effector observation ---------------------------------------
    effector_state: dict[str, np.ndarray] = {}
    effector_state_t: dict[str, np.ndarray] = {}
    effector_state_source: dict[str, str] = {}
    for effector in profile.end_effectors:
        values, times, valid, source = _effector_observation(effector, raw, grid_ns, effector_tol)
        effector_state[effector.name] = values
        effector_state_t[effector.name] = times
        effector_state_source[effector.name] = source
        valid_parts[f"state:{effector.name}"] = valid

    # -- arm action ------------------------------------------------------
    live_command_topics = [t for t in profile.arm.command_topics if raw.has_arm_command(t)]
    cmd_indices: dict[str, np.ndarray] = {}
    cmd_leads: dict[str, np.ndarray] = {}
    real_masks: dict[str, np.ndarray] = {}
    for topic in profile.arm.command_topics:
        if not raw.has_arm_command(topic):
            # Nothing was ever published: every row is a fill row, which the
            # joint-state-fill assembly below turns into that arm's measured
            # joints.  No tick can violate a limit it has no sample for, so the
            # topic contributes no validity mask.
            cmd_indices[topic] = np.zeros(frame_count, dtype=np.intp)
            cmd_leads[topic] = np.zeros(frame_count, dtype=np.int64)
            real_masks[topic] = np.zeros(frame_count, dtype=bool)
            continue
        indices, lead, real, valid = _select_command(
            raw.cmd_t[topic], raw.cmd_v[topic], grid_ns, action_tol, cfg.action_gap_policy
        )
        cmd_indices[topic] = indices
        cmd_leads[topic] = lead
        real_masks[topic] = real
        valid_parts[f"command:{topic}"] = valid

    selected_cmd_times = np.stack(
        [raw.cmd_t[topic][cmd_indices[topic]] for topic in live_command_topics]
    )
    command_skew = selected_cmd_times.max(axis=0) - selected_cmd_times.min(axis=0)
    all_real = np.logical_and.reduce([real_masks[t] for t in profile.arm.command_topics])
    any_real = np.logical_or.reduce([real_masks[t] for t in profile.arm.command_topics])
    # Skew only means anything when every arm has a genuine command this tick.
    pair_valid = ~all_real | (command_skew <= pair_tol)
    if cfg.action_gap_policy == "fail":
        valid_parts["command_pair_skew"] = pair_valid
    if cfg.action_gap_policy == "joint-state-fill":
        # Arms are teleoperated independently here, so a row still carries real
        # intent when any arm was commanded; requiring all of them would mask
        # out every row of a single-arm episode.  Per-arm detail stays in
        # audit.hold.joint_state_fill_rows.
        hold_mask = ~any_real
    else:
        hold_mask = ~(all_real & pair_valid)

    if cfg.action_gap_policy == "joint-state-fill":
        # A silent arm is filled from its own measured state, so an episode is
        # usable as long as *some* arm was genuinely teleoperated.
        if not any_real.any():
            raise ConversionError("Episode contains no genuine teleop command rows on any arm")
    elif cfg.action_gap_policy != "fail" and not all_real.any():
        raise ConversionError("Episode contains no genuine teleop command rows")

    # -- end-effector action --------------------------------------------
    reference_time = selected_cmd_times.max(axis=0)
    effector_action: dict[str, np.ndarray] = {}
    effector_action_t: dict[str, np.ndarray] = {}
    effector_action_age: dict[str, np.ndarray] = {}
    effector_action_source: dict[str, str] = {}
    for effector in profile.end_effectors:
        if not raw.has_effector_command(effector):
            if not raw.has_effector_state(effector):
                raise ConversionError(
                    f"end effector {effector.name}: neither {effector.command_topic} nor a "
                    "measured state topic carries data, so its action cannot be reconstructed"
                )
            # Same contract as a silent arm: the measured position is the only
            # honest statement of what the operator asked for.  Identity copy of
            # the observation, flagged in the audit.
            effector_action[effector.name] = effector_state[effector.name]
            effector_action_t[effector.name] = effector_state_t[effector.name]
            effector_action_age[effector.name] = np.zeros(frame_count, dtype=np.int64)
            effector_action_source[effector.name] = STATE_FILL
            continue
        times = raw.ee_cmd_t[effector.name]
        values = raw.ee_cmd_v[effector.name]
        indices, age, valid = previous_indices(times, reference_time)
        valid &= age <= effector_tol
        effector_action[effector.name] = values[indices]
        effector_action_t[effector.name] = times[indices]
        effector_action_age[effector.name] = age
        effector_action_source[effector.name] = "command"
        valid_parts[f"action:{effector.name}"] = valid

    # -- validity --------------------------------------------------------
    valid_mask = np.ones(frame_count, dtype=bool)
    for part in valid_parts.values():
        valid_mask &= part
    invalid_counts = {
        name: int((~part).sum()) for name, part in valid_parts.items() if not part.all()
    }
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
    arm_state_t = keep(arm_state_t)
    command_skew = keep(command_skew)
    hold_mask = keep(hold_mask)
    all_real = keep(all_real)
    any_real = keep(any_real)
    image_indices = {name: keep(indices) for name, indices in image_indices.items()}
    depth_indices = {name: keep(indices) for name, indices in depth_indices.items()}

    # -- assemble state / action ----------------------------------------
    # Each command topic drives a fixed slice of the arm, in declaration order:
    # topic i owns joint_names[i * command_dim : (i + 1) * command_dim].  That
    # mapping is enforced by ArmSpec, so a silent arm can be filled from its own
    # measured joints without guessing which columns belong to it.
    arm_blocks: list[np.ndarray] = []
    joint_state_fill_rows: dict[str, int] = {}
    for position, topic in enumerate(profile.arm.command_topics):
        if raw.has_arm_command(topic):
            block = raw.cmd_v[topic][keep(cmd_indices[topic])]
        else:
            block = np.zeros(
                (int(arm_qpos.shape[0]), profile.arm.command_dim), dtype=np.float64
            )
        if cfg.action_gap_policy == "joint-state-fill":
            filled = ~keep(real_masks[topic])
            if filled.any():
                start = position * profile.arm.command_dim
                stop = start + profile.arm.command_dim
                block = block.copy()
                # Older recordings drop joint_cmd while the arm is still held in
                # place by the controller; its measured position is then the
                # only honest statement of the commanded pose.
                block[filled] = arm_qpos[filled, start:stop]
            joint_state_fill_rows[topic] = int(filled.sum())
        arm_blocks.append(block)
    qpos_parts: list[np.ndarray] = [arm_qpos]
    action_parts: list[np.ndarray] = [np.concatenate(arm_blocks, axis=1)]
    for effector in profile.end_effectors:
        qpos_parts.append(keep(effector_state[effector.name]))
        action_parts.append(keep(effector_action[effector.name]))

    qpos = np.concatenate(qpos_parts, axis=1).astype(np.float32)
    action = np.concatenate(action_parts, axis=1).astype(np.float32)
    if qpos.shape[1] != profile.state_dim or action.shape[1] != profile.action_dim:
        raise ConversionError(
            f"Assembled width mismatch: state {qpos.shape[1]} / action {action.shape[1]} "
            f"vs profile {profile.state_dim} / {profile.action_dim}"
        )
    if np.array_equal(qpos, action):
        raise ConversionError(
            "Entire action array equals observation state; refusing unsafe dataset"
        )

    # -- velocity (causal) ----------------------------------------------
    if not np.all(np.isfinite(arm_qvel)):
        arm_qvel = backward_difference(arm_qpos, cfg.fps)
    velocity_parts = [arm_qvel]
    for effector in profile.end_effectors:
        velocity_parts.append(backward_difference(keep(effector_state[effector.name]), cfg.fps))
    qvel = np.concatenate(velocity_parts, axis=1).astype(np.float32)

    images, depths = decode_selected_media(raw, cfg, image_indices, depth_indices)

    # -- hold accounting -------------------------------------------------
    hold_runs = runs(hold_mask)
    hold_segments = [
        {
            "start_time_s": float(grid_ns[start] - grid_ns[0]) / 1e9,
            "end_time_s": float(grid_ns[end] - grid_ns[0]) / 1e9,
            "duration_s": float(grid_ns[end] - grid_ns[start]) / 1e9,
            "rows": end - start + 1,
        }
        for start, end in hold_runs
    ]
    max_hold_run_s = max((item["duration_s"] for item in hold_segments), default=0.0)
    hold_fraction = float(hold_mask.mean()) if hold_mask.size else 0.0
    if cfg.max_hold_fraction is not None and hold_fraction > cfg.max_hold_fraction:
        raise ConversionError(
            f"held-action fraction {hold_fraction:.3f} exceeds limit {cfg.max_hold_fraction:.3f}"
        )
    if cfg.max_hold_run_s is not None and max_hold_run_s > cfg.max_hold_run_s:
        raise ConversionError(
            f"longest held-action run {max_hold_run_s:.2f}s exceeds limit {cfg.max_hold_run_s:.2f}s"
        )

    timestamps: dict[str, np.ndarray] = {
        "grid_ns": grid_ns,
        "arm_state_ns": arm_state_t,
        "action_hold_mask": hold_mask,
    }
    for topic in live_command_topics:
        key = topic.strip("/").replace("/", "_")
        timestamps[f"command_{key}_ns"] = raw.cmd_t[topic][keep(cmd_indices[topic])]
    for effector in profile.end_effectors:
        timestamps[f"state_{effector.name}_ns"] = keep(effector_state_t[effector.name])
        timestamps[f"action_{effector.name}_ns"] = keep(effector_action_t[effector.name])
    for name, indices in image_indices.items():
        timestamps[f"image_{name}_ns"] = raw.image_t[name][indices]
    for name, indices in depth_indices.items():
        timestamps[f"depth_{name}_ns"] = raw.depth_t[name][indices]

    def unique_ratio(indices: np.ndarray) -> float:
        return float(len(np.unique(indices)) / len(indices)) if len(indices) else 0.0

    audit: dict[str, Any] = {
        "schema_version": "aligned-control-rows-v2",
        "source_bag": str(raw.bag_dir),
        "profile": profile.to_dict(),
        "alignment_mode": cfg.mode,
        "grid_anchor": cfg.grid_anchor,
        "action_gap_policy": cfg.action_gap_policy,
        "fps": cfg.fps,
        # Actual row spacing: equals fps for fixed-rate grids, and the anchor
        # camera's real rate for anchor-camera-ticks.
        "measured_tick_hz": measured_fps,
        "candidate_frames": frame_count,
        "output_frames": int(qpos.shape[0]),
        "dropped_frames": int(frame_count - qpos.shape[0]),
        "invalid_counts": invalid_counts,
        "window": {
            "bag_duration_s": float(window.bag_end_ns - window.bag_start_ns) / 1e9,
            "command_span_s": float(window.command_end_ns - window.command_start_ns) / 1e9,
            "converted_span_s": float(window.end_ns - window.start_ns) / 1e9,
            "command_coverage": float(window.command_end_ns - window.command_start_ns)
            / float(window.bag_end_ns - window.bag_start_ns),
            "teleop_start_offset_s": float(window.command_start_ns - window.bag_start_ns) / 1e9,
            "anchor_lead_ms": float(window.command_start_ns - window.anchor_ns) / 1e6,
            "anchor_camera": profile.resolved_anchor_camera,
        },
        "hold": {
            "rows": int(hold_mask.sum()),
            "fraction": hold_fraction,
            "max_run_s": max_hold_run_s,
            "segments": hold_segments,
            "real_command_rows": int(all_real.sum()),
            "any_arm_real_command_rows": int(any_real.sum()),
            # Per arm: rows whose action came from measured joints rather than a
            # command.  Those columns are an identity copy of the observation,
            # so training should treat them via action_hold_mask.
            "joint_state_fill_rows": joint_state_fill_rows,
        },
        "unique_ratio": {
            **{f"image_{name}": unique_ratio(indices) for name, indices in image_indices.items()},
            **{
                f"command_{topic.strip('/').replace('/', '_')}": unique_ratio(
                    keep(cmd_indices[topic])
                )
                for topic in live_command_topics
            },
        },
        # Profile topics this recording did not deliver, and what replaced them.
        # Reconstructed columns are an identity copy of the corresponding
        # observation, so treat them as held rather than as demonstrated intent.
        "missing_topics": {
            topic: {
                "status": status,
                "filled_from": _fill_source(topic, profile),
            }
            for topic, status in sorted(raw.missing_topics.items())
        },
        "end_effector_state_source": effector_state_source,
        "end_effector_action_source": effector_action_source,
        "image_formats": {
            name: raw.image_kinds.get(topic, "unknown") for name, topic in profile.cameras.items()
        },
        "tolerant_cdr_parses": raw.tolerant_parses,
        "topic_counts": raw.topic_counts,
        "source_period_ms": {"arm_state": arm_period_ms},
        "limits_ms": {
            "image": cfg.resolved_image_tolerance_ms,
            "state": state_tolerance_ms,
            "action": cfg.resolved_action_tolerance_ms,
            "action_pair": cfg.action_pair_tolerance_ms,
            "end_effector": cfg.end_effector_tolerance_ms,
        },
        "metrics_ms": {
            # Lead is only meaningful where a genuine command exists; on held
            # rows it measures the distance to the far side of the gap.
            **{
                f"command_{topic.strip('/').replace('/', '_')}_lead": stats_ms(
                    keep(cmd_leads[topic])[all_real]
                )
                for topic in live_command_topics
            },
            "command_pair_skew": stats_ms(command_skew),
            **{
                f"action_{effector.name}_age": stats_ms(keep(effector_action_age[effector.name]))
                for effector in profile.end_effectors
            },
            **{
                f"image_{name}_offset": stats_ms(keep(offsets))
                for name, offsets in image_offsets.items()
            },
            **{
                f"depth_{name}_offset": stats_ms(keep(offsets))
                for name, offsets in depth_offsets.items()
            },
        },
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
        state_names=tuple(profile.state_names),
        robot_type=profile.robot_type,
    )
