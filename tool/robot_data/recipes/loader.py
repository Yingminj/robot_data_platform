"""Named conversion recipes: one data format spelled out once.

A recipe bundles everything that is a property of *how a batch was recorded*
rather than of *this particular run*: which profile applies, which camera topics
that generation used, how lenient the alignment has to be, and the video encoder
settings the datasets are built with.  Without them the same handful of flags
gets retyped per batch, and the camera topic ends up being hand-edited into the
profile -- which is how a profile and a recording drift apart.

Precedence, lowest to highest:

1. dataclass defaults in :class:`~robot_data.align.config.AlignmentConfig` and
   :class:`~robot_data.writers.lerobot_v3.RGBVideoConfig`
2. the recipe's ``alignment`` / ``video`` blocks
3. flags given explicitly on the command line

Step 3 is why the CLI parsers default every recipe-controlled flag to ``None``:
an explicit ``--crf 28`` has to be distinguishable from "not mentioned", which a
non-None argparse default cannot express.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from robot_data.errors import RecipeError
from robot_data.profiles.schema import RobotProfile, apply_topic_overrides, load_profile

RECIPE_DIR = Path(__file__).resolve().parent

# Keys accepted inside each block, mirroring the dataclass fields they fill.
ALIGNMENT_KEYS = {
    "fps",
    "mode",
    "image_tolerance_ms",
    "state_tolerance_ms",
    "state_tolerance_periods",
    "action_tolerance_ms",
    "action_pair_tolerance_ms",
    "end_effector_tolerance_ms",
    "image_height",
    "image_width",
    "invalid_frame_policy",
    "include_depth",
    "max_decode_errors",
    "action_gap_policy",
    "missing_topic_policy",
    "grid_anchor",
    "max_hold_fraction",
    "max_hold_run_s",
    "max_tick_rate_deviation",
}
VIDEO_KEYS = {"codec", "pixel_format", "crf", "gop", "preset", "fast_decode", "encoder_threads"}
TOP_LEVEL_KEYS = {
    "name",
    "description",
    "storage",
    "profile",
    "cameras",
    "depths",
    "anchor_camera",
    "robot_type",
    "alignment",
    "video",
    "notes",
    "_comment",
}


@dataclass(frozen=True)
class Recipe:
    """A resolved recipe: the profile plus the defaults it implies."""

    name: str
    description: str
    storage: str
    profile_ref: str
    profile: RobotProfile
    alignment: dict[str, Any] = field(default_factory=dict)
    video: dict[str, Any] = field(default_factory=dict)
    robot_type: str | None = None
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """What gets recorded in the dataset manifest."""
        return {
            "name": self.name,
            "description": self.description,
            "storage": self.storage,
            "profile": self.profile_ref,
            "alignment": dict(self.alignment),
            "video": dict(self.video),
            "robot_type": self.robot_type,
        }


def recipe_names() -> list[str]:
    return sorted(path.stem for path in RECIPE_DIR.glob("*.json"))


def _validate(payload: dict[str, Any], source: str) -> None:
    unknown = set(payload) - TOP_LEVEL_KEYS
    if unknown:
        raise RecipeError(f"{source}: unknown recipe keys {sorted(unknown)}")
    for block, allowed in (("alignment", ALIGNMENT_KEYS), ("video", VIDEO_KEYS)):
        extra = set(payload.get(block) or {}) - allowed
        if extra:
            raise RecipeError(f"{source}: unknown {block} keys {sorted(extra)}")
    if "profile" not in payload:
        raise RecipeError(f"{source}: recipe must name a profile")


@lru_cache(maxsize=None)
def load_recipe(name_or_path: str) -> Recipe:
    """Resolve a shipped recipe name or a path to a recipe JSON file."""
    if name_or_path in set(recipe_names()):
        path = RECIPE_DIR / f"{name_or_path}.json"
    else:
        path = Path(name_or_path).expanduser()
        if not path.is_file():
            raise RecipeError(
                f"unknown recipe {name_or_path!r}; shipped recipes are {recipe_names()} "
                "or pass a path to a recipe JSON"
            )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("_comment", None)
    _validate(payload, str(path))

    profile_ref = str(payload["profile"])
    # A bare name resolves against the built-ins; a relative path resolves
    # against the recipe file, so a recipe kept beside a custom profile keeps
    # working wherever the repository is checked out.
    candidate = (path.parent / profile_ref).resolve()
    profile = load_profile(str(candidate) if candidate.is_file() else profile_ref)
    profile = apply_topic_overrides(
        profile,
        cameras=payload.get("cameras"),
        depths=payload.get("depths"),
        anchor_camera=payload.get("anchor_camera"),
    )
    return Recipe(
        name=str(payload.get("name", path.stem)),
        description=str(payload.get("description", "")),
        storage=str(payload.get("storage", "any")),
        profile_ref=profile_ref,
        profile=profile,
        alignment=dict(payload.get("alignment") or {}),
        video=dict(payload.get("video") or {}),
        robot_type=payload.get("robot_type"),
        notes=tuple(payload.get("notes") or ()),
    )


def describe_recipes() -> list[dict[str, Any]]:
    """One summary row per shipped recipe, for ``rdp recipes``."""
    rows = []
    for name in recipe_names():
        recipe = load_recipe(name)
        rows.append(
            {
                "name": name,
                "storage": recipe.storage,
                "description": recipe.description,
                "profile": recipe.profile.name,
                "state_dim": recipe.profile.state_dim,
                "cameras": {
                    camera: recipe.profile.cameras[camera]
                    for camera in sorted(recipe.profile.cameras)
                },
                "anchor_camera": recipe.profile.resolved_anchor_camera,
                "alignment": recipe.alignment,
                "video": recipe.video,
                "notes": list(recipe.notes),
            }
        )
    return rows
