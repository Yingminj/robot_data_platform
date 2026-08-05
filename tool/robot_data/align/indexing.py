"""Vectorised nearest/previous/next sample lookups on sorted timestamp arrays.

Every alignment decision -- which camera frame, which joint sample, which
command belongs to a tick -- reduces to one of these, so they live apart from
the policy that chooses between them.
"""

from __future__ import annotations

import numpy as np


def nearest_indices(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    right = np.searchsorted(source, target, side="left")
    right = np.clip(right, 0, len(source) - 1)
    left = np.clip(right - 1, 0, len(source) - 1)
    use_right = np.abs(source[right] - target) < np.abs(target - source[left])
    indices = np.where(use_right, right, left)
    return indices, np.abs(source[indices] - target)


def previous_indices(
    source: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.searchsorted(source, target, side="right") - 1
    valid = indices >= 0
    safe = np.clip(indices, 0, len(source) - 1)
    age = target - source[safe]
    return safe, age, valid


def next_indices(
    source: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.searchsorted(source, target, side="left")
    valid = indices < len(source)
    safe = np.clip(indices, 0, len(source) - 1)
    lead = source[safe] - target
    return safe, lead, valid


def interpolate(
    source_t: np.ndarray, source_v: np.ndarray, target: np.ndarray, tolerance_ns: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    right = np.searchsorted(source_t, target, side="left")
    right_safe = np.clip(right, 0, len(source_t) - 1)
    exact = (right < len(source_t)) & (source_t[right_safe] == target)
    left = right - 1
    valid = exact | ((left >= 0) & (right < len(source_t)))
    left_safe = np.clip(left, 0, len(source_t) - 1)
    left_age = target - source_t[left_safe]
    right_lead = source_t[right_safe] - target
    valid &= exact | ((left_age <= tolerance_ns) & (right_lead <= tolerance_ns))
    denom = (source_t[right_safe] - source_t[left_safe]).astype(np.float64)
    alpha = np.divide(
        target - source_t[left_safe], denom, out=np.zeros_like(denom), where=denom != 0
    )
    output = source_v[left_safe] + alpha[:, None] * (source_v[right_safe] - source_v[left_safe])
    output[exact] = source_v[right_safe[exact]]
    selected_time = np.where(exact, source_t[right_safe], target)
    return output, selected_time, valid


def backward_difference(values: np.ndarray, fps: int) -> np.ndarray:
    """Causal derivative: row k depends only on rows <= k."""
    output = np.zeros_like(values, dtype=np.float64)
    if values.shape[0] > 1:
        output[1:] = (values[1:] - values[:-1]) * float(fps)
        output[0] = output[1]
    return output


def stats_ms(values_ns: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values_ns, dtype=np.float64) / 1e6
    if not values.size:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "max": None}
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous ``True`` runs as inclusive (start, end) index pairs."""
    if not mask.any():
        return []
    padded = np.concatenate(([False], mask, [False]))
    edges = np.diff(padded.astype(np.int8))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0] - 1
    return list(zip(starts.tolist(), ends.tolist(), strict=True))
