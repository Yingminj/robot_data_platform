"""Robot profiles: what a recording setup contains, as data rather than code."""

from robot_data.profiles.schema import (  # noqa: F401
    DEFAULT_PROFILE,
    ArmSpec,
    CameraTile,
    EndEffectorSpec,
    RobotProfile,
    apply_topic_overrides,
    builtin_profile_names,
    load_profile,
    parse_name_topic,
    profile_from_dict,
)
