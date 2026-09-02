#!/usr/bin/env python3
"""Replace joint state/action with EEF state/action in a LeRobot v3 dataset.

The input joint positions are read from ``observation.state``.  Forward
kinematics are evaluated from the same M6-696 MJCF and tool frames used by
``Apex_Deploy``.  With the default Euler representation, the new
``observation.state`` is::

    [eef_l(6), eef_r(6), gripper_L, gripper_R]

and therefore has 14 dimensions.  Each EEF pose is
``[x, y, z, roll, pitch, yaw]`` in the ``base_link`` frame.  Euler angles are
in radians and use
``R = Rz(yaw) Ry(pitch) Rx(roll)``.  The action is converted in exactly the
same way from its joint targets, yielding another 14-D vector.  Quaternion
output remains available with ``--rotation-repr quaternion`` and produces
16-D state/action vectors.  No separate EEF columns are added.

The source dataset is never modified.  A sibling dataset is created using hard
links where possible (so videos do not consume another full copy), while data
parquets and metadata are atomically replaced in the new dataset.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_DATASET = Path(
    "/home/snorlax/work/yaocao/recorded_data/tidy_up_stationery_le/"
    "batch_success_505"
)
DEFAULT_MJCF = Path(__file__).resolve().parent / (
    "Apex_Deploy/robot_node/marvin_description/mjcf/matrix/m6_696.xml"
)
POSE_NAMES = {
    "euler": ["x", "y", "z", "roll", "pitch", "yaw"],
    "quaternion": ["x", "y", "z", "qx", "qy", "qz", "qw"],
}
STAT_NAMES = ["min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99"]


@dataclass(frozen=True)
class FixedTransform:
    matrix: np.ndarray


@dataclass(frozen=True)
class HingeTransform:
    name: str
    axis: np.ndarray
    position: np.ndarray


def _numbers(value: str | None, default: tuple[float, ...]) -> np.ndarray:
    if value is None:
        return np.asarray(default, dtype=np.float64)
    return np.asarray([float(item) for item in value.split()], dtype=np.float64)


def _quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat)
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError(f"invalid MJCF quaternion: {quat.tolist()}")
    w, x, y, z = quat / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _fixed_body_transform(body: ET.Element) -> np.ndarray:
    unsupported = [
        name for name in ("euler", "axisangle", "xyaxes", "zaxis") if body.get(name)
    ]
    if unsupported:
        raise ValueError(
            f"body {body.get('name')!r} uses unsupported orientation attribute(s): {unsupported}"
        )
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = _quat_wxyz_to_matrix(
        _numbers(body.get("quat"), (1.0, 0.0, 0.0, 0.0))
    )
    matrix[:3, 3] = _numbers(body.get("pos"), (0.0, 0.0, 0.0))
    return matrix


def _find_body_chain(
    mjcf_path: Path, target_name: str
) -> list[FixedTransform | HingeTransform]:
    root = ET.parse(mjcf_path).getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"MJCF has no worldbody: {mjcf_path}")

    def visit(body: ET.Element, parent_ops: list[FixedTransform | HingeTransform]):
        ops = [*parent_ops, FixedTransform(_fixed_body_transform(body))]
        for joint in body.findall("joint"):
            joint_type = joint.get("type", "hinge")
            if joint_type != "hinge":
                raise ValueError(
                    f"joint {joint.get('name')!r} has unsupported type {joint_type!r}; only hinge is supported"
                )
            name = joint.get("name")
            if not name:
                raise ValueError("all joints on an EEF chain must have names")
            axis = _numbers(joint.get("axis"), (0.0, 0.0, 1.0))
            axis_norm = np.linalg.norm(axis)
            if not np.isfinite(axis_norm) or axis_norm == 0.0:
                raise ValueError(f"joint {name!r} has an invalid axis")
            ops.append(
                HingeTransform(
                    name=name,
                    axis=axis / axis_norm,
                    position=_numbers(joint.get("pos"), (0.0, 0.0, 0.0)),
                )
            )

        if body.get("name") == target_name:
            return ops
        for child in body.findall("body"):
            result = visit(child, ops)
            if result is not None:
                return result
        return None

    for body in worldbody.findall("body"):
        result = visit(body, [])
        if result is not None:
            return result
    raise ValueError(f"frame/body {target_name!r} was not found in {mjcf_path}")


def _axis_angle_transforms(
    axis: np.ndarray, position: np.ndarray, angles: np.ndarray
) -> np.ndarray:
    """Return batched T(position) R(axis, angle) T(-position)."""
    x, y, z = axis
    skew = np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)
    skew2 = skew @ skew
    sin_angle = np.sin(angles)[:, None, None]
    one_minus_cos = (1.0 - np.cos(angles))[:, None, None]
    rotations = (
        np.eye(3, dtype=np.float64)[None, :, :]
        + sin_angle * skew
        + one_minus_cos * skew2
    )

    transforms = np.zeros((len(angles), 4, 4), dtype=np.float64)
    transforms[:, :3, :3] = rotations
    transforms[:, :3, 3] = position - np.einsum("nij,j->ni", rotations, position)
    transforms[:, 3, 3] = 1.0
    return transforms


def _rotation_matrices_to_xyzw(rotations: np.ndarray) -> np.ndarray:
    """Convert rotation matrices to normalized quaternions with non-negative w."""
    count = rotations.shape[0]
    quats = np.empty((count, 4), dtype=np.float64)
    for index, rotation in enumerate(rotations):
        trace = float(np.trace(rotation))
        if trace > 0.0:
            scale = np.sqrt(trace + 1.0) * 2.0
            qw = 0.25 * scale
            qx = (rotation[2, 1] - rotation[1, 2]) / scale
            qy = (rotation[0, 2] - rotation[2, 0]) / scale
            qz = (rotation[1, 0] - rotation[0, 1]) / scale
        elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
            scale = (
                np.sqrt(
                    max(0.0, 1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])
                )
                * 2.0
            )
            qw = (rotation[2, 1] - rotation[1, 2]) / scale
            qx = 0.25 * scale
            qy = (rotation[0, 1] + rotation[1, 0]) / scale
            qz = (rotation[0, 2] + rotation[2, 0]) / scale
        elif rotation[1, 1] > rotation[2, 2]:
            scale = (
                np.sqrt(
                    max(0.0, 1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2])
                )
                * 2.0
            )
            qw = (rotation[0, 2] - rotation[2, 0]) / scale
            qx = (rotation[0, 1] + rotation[1, 0]) / scale
            qy = 0.25 * scale
            qz = (rotation[1, 2] + rotation[2, 1]) / scale
        else:
            scale = (
                np.sqrt(
                    max(0.0, 1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1])
                )
                * 2.0
            )
            qw = (rotation[1, 0] - rotation[0, 1]) / scale
            qx = (rotation[0, 2] + rotation[2, 0]) / scale
            qy = (rotation[1, 2] + rotation[2, 1]) / scale
            qz = 0.25 * scale
        quat = np.asarray([qx, qy, qz, qw], dtype=np.float64)
        quat /= np.linalg.norm(quat)
        if quat[3] < 0.0:
            quat = -quat
        quats[index] = quat
    return quats


def _rotation_matrices_to_rpy(rotations: np.ndarray) -> np.ndarray:
    """Convert matrices to roll/pitch/yaw using R = Rz(yaw) Ry(pitch) Rx(roll).

    At the pitch = +/- pi/2 singularity, yaw is set to zero and roll absorbs
    the remaining observable rotation.  Returned angles are in radians.
    """
    rotations = np.asarray(rotations, dtype=np.float64)
    pitch = np.arctan2(
        -rotations[:, 2, 0],
        np.hypot(rotations[:, 0, 0], rotations[:, 1, 0]),
    )
    roll = np.arctan2(rotations[:, 2, 1], rotations[:, 2, 2])
    yaw = np.arctan2(rotations[:, 1, 0], rotations[:, 0, 0])

    singular = np.hypot(rotations[:, 0, 0], rotations[:, 1, 0]) < 1e-8
    if np.any(singular):
        roll[singular] = np.arctan2(
            -rotations[singular, 1, 2], rotations[singular, 1, 1]
        )
        yaw[singular] = 0.0
    return np.stack((roll, pitch, yaw), axis=1)


class MjcfForwardKinematics:
    def __init__(
        self,
        mjcf_path: Path,
        joint_names: list[str],
        left_frame: str,
        right_frame: str,
        rotation_repr: str = "euler",
    ):
        if rotation_repr not in POSE_NAMES:
            raise ValueError(f"unsupported rotation representation: {rotation_repr!r}")
        self.mjcf_path = mjcf_path
        self.rotation_repr = rotation_repr
        self.joint_index = {name: index for index, name in enumerate(joint_names)}
        if len(self.joint_index) != len(joint_names):
            raise ValueError("joint names in the state feature must be unique")
        self.left_ops = _find_body_chain(mjcf_path, left_frame)
        self.right_ops = _find_body_chain(mjcf_path, right_frame)
        required = {
            op.name
            for ops in (self.left_ops, self.right_ops)
            for op in ops
            if isinstance(op, HingeTransform)
        }
        missing = sorted(required - self.joint_index.keys())
        if missing:
            raise ValueError(f"state feature is missing MJCF joints: {missing}")

    def _evaluate_chain(
        self, joint_positions: np.ndarray, ops: list[FixedTransform | HingeTransform]
    ) -> np.ndarray:
        count = joint_positions.shape[0]
        transforms = np.broadcast_to(np.eye(4, dtype=np.float64), (count, 4, 4)).copy()
        for op in ops:
            if isinstance(op, FixedTransform):
                transforms = transforms @ op.matrix
            else:
                angles = joint_positions[:, self.joint_index[op.name]]
                transforms = transforms @ _axis_angle_transforms(
                    op.axis, op.position, angles
                )
        positions = transforms[:, :3, 3]
        if self.rotation_repr == "euler":
            rotations = _rotation_matrices_to_rpy(transforms[:, :3, :3])
        else:
            rotations = _rotation_matrices_to_xyzw(transforms[:, :3, :3])
        return np.concatenate((positions, rotations), axis=1).astype(np.float32)

    def evaluate(self, joint_positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        joint_positions = np.asarray(joint_positions, dtype=np.float64)
        if joint_positions.ndim != 2:
            raise ValueError(
                f"joint positions must be a 2-D array, got {joint_positions.shape}"
            )
        return (
            self._evaluate_chain(joint_positions, self.left_ops),
            self._evaluate_chain(joint_positions, self.right_ops),
        )


def _json_value(value: Any) -> Any:
    array = np.asarray(value)
    return array.item() if array.ndim == 0 else array.tolist()


def _link_or_copy(source: str, destination: str) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=4)
        handle.write("\n")
    os.replace(temporary, path)


def _replace_or_append_column(table, name: str, values):
    index = table.schema.get_field_index(name)
    if index >= 0:
        return table.set_column(index, name, values)
    return table.append_column(name, values)


def _load_dependencies():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        from lerobot.datasets.compute_stats import (
            aggregate_stats,
            compute_episode_stats,
        )
    except ImportError as error:
        raise RuntimeError(
            "This script requires pyarrow and lerobot. Run it with the project environment, for example:\n"
            "  conda run -n lerobot python add_eef_to_lerobot.py ..."
        ) from error
    return pa, pq, compute_episode_stats, aggregate_stats


def _read_dataset_info(dataset_root: Path) -> dict[str, Any]:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"missing LeRobot metadata: {info_path}")
    with info_path.open(encoding="utf-8") as handle:
        info = json.load(handle)
    if info.get("codebase_version") != "v3.0":
        raise ValueError(
            f"expected a LeRobot v3.0 dataset, got {info.get('codebase_version')!r}"
        )
    return info


def _state_joint_names(info: dict[str, Any], state_key: str) -> list[str]:
    feature = info.get("features", {}).get(state_key)
    if not feature:
        raise ValueError(f"info.json has no feature {state_key!r}")
    names = feature.get("names")
    if not isinstance(names, list):
        raise ValueError(f"feature {state_key!r} has no joint names")
    return [str(name) for name in names]


def _gripper_indices(joint_names: list[str], gripper_names: list[str]) -> list[int]:
    missing = [name for name in gripper_names if name not in joint_names]
    if missing:
        raise ValueError(f"state feature is missing gripper fields: {missing}")
    return [joint_names.index(name) for name in gripper_names]


def _combined_state_names(rotation_repr: str, gripper_names: list[str]) -> list[str]:
    pose_names = POSE_NAMES[rotation_repr]
    return [
        *(f"eef_l_{name}" for name in pose_names),
        *(f"eef_r_{name}" for name in pose_names),
        *gripper_names,
    ]


def validate_first_frame(
    dataset_root: Path,
    mjcf_path: Path,
    state_key: str,
    action_key: str,
    left_frame: str,
    right_frame: str,
    rotation_repr: str,
    gripper_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    _, pq, _, _ = _load_dependencies()
    info = _read_dataset_info(dataset_root)
    state_names = _state_joint_names(info, state_key)
    action_names = _state_joint_names(info, action_key)
    data_files = sorted((dataset_root / "data").glob("chunk-*/file-*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"no data parquet files under {dataset_root / 'data'}")
    table = pq.ParquetFile(data_files[0]).read_row_group(
        0, columns=[state_key, action_key]
    )
    states = np.asarray(table[state_key].slice(0, 1).to_pylist(), dtype=np.float64)
    actions = np.asarray(table[action_key].slice(0, 1).to_pylist(), dtype=np.float64)
    state_fk = MjcfForwardKinematics(
        mjcf_path, state_names, left_frame, right_frame, rotation_repr
    )
    action_fk = MjcfForwardKinematics(
        mjcf_path, action_names, left_frame, right_frame, rotation_repr
    )
    state_left, state_right = state_fk.evaluate(states)
    action_left, action_right = action_fk.evaluate(actions)
    converted_state = np.concatenate(
        (
            state_left,
            state_right,
            states[:, _gripper_indices(state_names, gripper_names)],
        ),
        axis=1,
    ).astype(np.float32)
    converted_action = np.concatenate(
        (
            action_left,
            action_right,
            actions[:, _gripper_indices(action_names, gripper_names)],
        ),
        axis=1,
    ).astype(np.float32)
    return converted_state, converted_action


def _convert_state_and_action_in_data_files(
    dataset_root: Path,
    info: dict[str, Any],
    mjcf_path: Path,
    state_key: str,
    action_key: str,
    gripper_names: list[str],
    left_frame: str,
    right_frame: str,
    rotation_repr: str,
):
    pa, pq, compute_episode_stats, aggregate_stats = _load_dependencies()
    state_names = _state_joint_names(info, state_key)
    action_names = _state_joint_names(info, action_key)
    state_fk = MjcfForwardKinematics(
        mjcf_path, state_names, left_frame, right_frame, rotation_repr
    )
    action_fk = MjcfForwardKinematics(
        mjcf_path, action_names, left_frame, right_frame, rotation_repr
    )
    state_gripper_indices = _gripper_indices(state_names, gripper_names)
    action_gripper_indices = _gripper_indices(action_names, gripper_names)
    output_dimension = 2 * len(POSE_NAMES[rotation_repr]) + len(gripper_names)
    data_files = sorted((dataset_root / "data").glob("chunk-*/file-*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"no data parquet files under {dataset_root / 'data'}")

    per_episode_values: dict[int, dict[str, list[np.ndarray]]] = {}
    processed_rows = 0
    expected_rows = int(info.get("total_frames", -1))

    for file_number, data_path in enumerate(data_files, start=1):
        parquet_file = pq.ParquetFile(data_path)
        for feature_key in (state_key, action_key):
            if feature_key not in parquet_file.schema_arrow.names:
                raise ValueError(f"{data_path} has no column {feature_key!r}")

        temporary = data_path.with_name(f".{data_path.name}.tmp-{uuid.uuid4().hex}")
        writer = None
        try:
            for row_group in range(parquet_file.metadata.num_row_groups):
                table = parquet_file.read_row_group(row_group)
                states = np.asarray(table[state_key].to_pylist(), dtype=np.float64)
                actions = np.asarray(table[action_key].to_pylist(), dtype=np.float64)
                if states.ndim != 2 or states.shape[1] != len(state_names):
                    raise ValueError(
                        f"{data_path}, row group {row_group}: {state_key} has shape {states.shape}, "
                        f"expected (*, {len(state_names)})"
                    )
                if actions.ndim != 2 or actions.shape[1] != len(action_names):
                    raise ValueError(
                        f"{data_path}, row group {row_group}: {action_key} has shape {actions.shape}, "
                        f"expected (*, {len(action_names)})"
                    )
                if not np.isfinite(states).all() or not np.isfinite(actions).all():
                    raise ValueError(
                        f"{data_path}, row group {row_group}: joint state/action contains NaN/Inf"
                    )

                state_left, state_right = state_fk.evaluate(states)
                action_left, action_right = action_fk.evaluate(actions)
                if (
                    not np.isfinite(state_left).all()
                    or not np.isfinite(state_right).all()
                    or not np.isfinite(action_left).all()
                    or not np.isfinite(action_right).all()
                ):
                    raise ValueError(
                        f"{data_path}, row group {row_group}: FK produced NaN/Inf"
                    )

                converted_state = np.concatenate(
                    (
                        state_left,
                        state_right,
                        states[:, state_gripper_indices],
                    ),
                    axis=1,
                )
                converted_action = np.concatenate(
                    (
                        action_left,
                        action_right,
                        actions[:, action_gripper_indices],
                    ),
                    axis=1,
                )
                if (
                    converted_state.shape[1] != output_dimension
                    or converted_action.shape[1] != output_dimension
                ):
                    raise AssertionError(
                        f"converted state/action dimensions are {converted_state.shape[1]}/"
                        f"{converted_action.shape[1]}, expected {output_dimension}"
                    )
                state_array = pa.array(
                    converted_state.astype(np.float32).tolist(),
                    type=pa.list_(pa.float32()),
                )
                action_array = pa.array(
                    converted_action.astype(np.float32).tolist(),
                    type=pa.list_(pa.float32()),
                )
                table = _replace_or_append_column(table, state_key, state_array)
                table = _replace_or_append_column(table, action_key, action_array)
                if writer is None:
                    writer = pq.ParquetWriter(
                        temporary, table.schema, compression="snappy"
                    )
                writer.write_table(table, row_group_size=len(table))

                episode_indices = np.asarray(
                    table["episode_index"].to_numpy(), dtype=np.int64
                )
                for episode_index in np.unique(episode_indices):
                    mask = episode_indices == episode_index
                    values = per_episode_values.setdefault(
                        int(episode_index), {state_key: [], action_key: []}
                    )
                    values[state_key].append(converted_state[mask])
                    values[action_key].append(converted_action[mask])
                processed_rows += len(table)
        except Exception:
            if writer is not None:
                writer.close()
            temporary.unlink(missing_ok=True)
            raise
        else:
            if writer is None:
                raise ValueError(f"empty parquet file: {data_path}")
            writer.close()
            os.replace(temporary, data_path)
        print(
            f"[{file_number}/{len(data_files)}] converted {data_path.relative_to(dataset_root)} "
            f"({parquet_file.metadata.num_rows} rows)",
            flush=True,
        )

    if expected_rows >= 0 and processed_rows != expected_rows:
        raise ValueError(
            f"processed {processed_rows} rows, but info.json says {expected_rows}"
        )

    feature_spec = {
        state_key: {"dtype": "float32", "shape": [output_dimension]},
        action_key: {"dtype": "float32", "shape": [output_dimension]},
    }
    per_episode_stats: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    for episode_index, feature_values in sorted(per_episode_values.items()):
        episode_data = {
            key: np.concatenate(chunks, axis=0)
            for key, chunks in feature_values.items()
        }
        per_episode_stats[episode_index] = compute_episode_stats(
            episode_data, feature_spec
        )
    global_stats = aggregate_stats(list(per_episode_stats.values()))
    return processed_rows, per_episode_stats, global_stats


def _update_episode_metadata(
    dataset_root: Path,
    per_episode_stats: dict[int, dict[str, dict[str, np.ndarray]]],
    state_key: str,
    action_key: str,
) -> None:
    pa, pq, _, _ = _load_dependencies()
    episode_files = sorted(
        (dataset_root / "meta" / "episodes").glob("chunk-*/file-*.parquet")
    )
    if not episode_files:
        raise FileNotFoundError("no meta/episodes parquet files found")

    seen: set[int] = set()
    for episode_path in episode_files:
        table = pq.read_table(episode_path)
        episode_indices = np.asarray(table["episode_index"].to_numpy(), dtype=np.int64)
        for feature_key in (state_key, action_key):
            for stat_name in STAT_NAMES:
                values = []
                for episode_index in episode_indices:
                    index = int(episode_index)
                    if index not in per_episode_stats:
                        raise ValueError(
                            f"episode {index} is in metadata but not in data parquets"
                        )
                    values.append(
                        _json_value(per_episode_stats[index][feature_key][stat_name])
                    )
                    seen.add(index)
                value_type = (
                    pa.list_(pa.int64())
                    if stat_name == "count"
                    else pa.list_(pa.float64())
                )
                column = pa.array(values, type=value_type)
                table = _replace_or_append_column(
                    table, f"stats/{feature_key}/{stat_name}", column
                )

        temporary = episode_path.with_name(
            f".{episode_path.name}.tmp-{uuid.uuid4().hex}"
        )
        pq.write_table(table, temporary, compression="snappy")
        os.replace(temporary, episode_path)

    missing = sorted(per_episode_stats.keys() - seen)
    if missing:
        raise ValueError(
            f"episodes are in data parquets but missing from metadata: {missing[:10]}"
        )


def convert_dataset(args: argparse.Namespace) -> Path:
    source = args.dataset.resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"dataset directory does not exist: {source}")
    mjcf_path = args.mjcf.resolve()
    if not mjcf_path.is_file():
        raise FileNotFoundError(f"MJCF file does not exist: {mjcf_path}")

    output = (
        args.output.resolve() if args.output else source.with_name(f"{source.name}_eef")
    )
    if output == source:
        raise ValueError("output must differ from the source dataset")
    if source in output.parents:
        raise ValueError("output must not be placed inside the source dataset")
    if args.state_key == args.action_key:
        raise ValueError("state and action feature keys must differ")
    if len(set(args.gripper_names)) != len(args.gripper_names):
        raise ValueError("left and right gripper feature names must differ")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    info = _read_dataset_info(source)
    staging = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    try:
        shutil.copytree(source, staging, copy_function=_link_or_copy)
        rows, per_episode_stats, global_stats = _convert_state_and_action_in_data_files(
            staging,
            info,
            mjcf_path,
            args.state_key,
            args.action_key,
            args.gripper_names,
            args.left_frame,
            args.right_frame,
            args.rotation_repr,
        )
        _update_episode_metadata(
            staging, per_episode_stats, args.state_key, args.action_key
        )

        output_info = dict(info)
        output_info["features"] = dict(info["features"])
        state_names = _combined_state_names(args.rotation_repr, args.gripper_names)
        output_info["features"][args.state_key] = {
            "dtype": "float32",
            "shape": [len(state_names)],
            "names": state_names,
        }
        output_info["features"][args.action_key] = {
            "dtype": "float32",
            "shape": [len(state_names)],
            "names": state_names,
        }
        _atomic_write_json(staging / "meta" / "info.json", output_info)

        stats_path = staging / "meta" / "stats.json"
        with stats_path.open(encoding="utf-8") as handle:
            stats = json.load(handle)
        stats[args.state_key] = {
            name: _json_value(value)
            for name, value in global_stats[args.state_key].items()
        }
        stats[args.action_key] = {
            name: _json_value(value)
            for name, value in global_stats[args.action_key].items()
        }
        _atomic_write_json(stats_path, stats)

        os.rename(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    print(f"Done: {rows} frames, {len(per_episode_stats)} episodes")
    print(f"Output: {output}")
    print(
        f"Feature: {args.state_key} = "
        f"[{','.join(_combined_state_names(args.rotation_repr, args.gripper_names))}]"
    )
    print(
        f"Feature: {args.action_key} = same "
        f"{len(_combined_state_names(args.rotation_repr, args.gripper_names))}-D "
        "EEF + gripper representation"
    )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replace joint observation.state/action with M6-696 EEF representations."
    )
    parser.add_argument("dataset", nargs="?", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--output", type=Path, help="output dataset (default: <dataset>_eef)"
    )
    parser.add_argument("--mjcf", type=Path, default=DEFAULT_MJCF)
    parser.add_argument("--state-key", default="observation.state")
    parser.add_argument("--action-key", default="action")
    parser.add_argument(
        "--gripper-names",
        nargs=2,
        default=["gripper_L", "gripper_R"],
        metavar=("LEFT", "RIGHT"),
        help="state feature names to retain after the two EEF poses",
    )
    parser.add_argument("--left-frame", default="left_tool")
    parser.add_argument("--right-frame", default="right_tool")
    parser.add_argument(
        "--rotation-repr",
        choices=sorted(POSE_NAMES),
        default="euler",
        help="rotation representation (default: euler, radians)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="compute and print the first converted state without writing a dataset",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.validate_only:
            state, action = validate_first_frame(
                args.dataset.resolve(),
                args.mjcf.resolve(),
                args.state_key,
                args.action_key,
                args.left_frame,
                args.right_frame,
                args.rotation_repr,
                args.gripper_names,
            )
            print(
                f"observation.state names: {_combined_state_names(args.rotation_repr, args.gripper_names)}"
            )
            print(f"observation.state: {state[0].tolist()}")
            print(
                f"action names: {_combined_state_names(args.rotation_repr, args.gripper_names)}"
            )
            print(f"action: {action[0].tolist()}")
            return 0
        convert_dataset(args)
        return 0
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())