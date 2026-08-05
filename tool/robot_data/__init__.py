"""Profile-driven conversion of robot recordings into training datasets.

Layering, lowest first -- nothing imports upward:

``errors`` / ``progress`` / ``discovery``
    Leaf utilities with no pipeline knowledge.
``profiles``
    What the robot is: topics, end effectors, cameras, derived state layout.
``ros``
    CDR and image decoding for the message types a recording carries.
``align``
    Reading a bag into arrays, choosing the episode window, and producing
    aligned control rows.
``writers``
    Serialising an aligned episode as ACT HDF5 or a LeRobot v3 dataset.
``qc``
    Read-only inspection of bags and HDF5 files.
``recipes``
    Named bundles of profile + alignment + video settings, so a data format is
    named once rather than respelled as flags on every run.
``cli``
    Argument parsing and the ``rdp`` entry points.
"""

__all__ = ["__version__"]

__version__ = "2.0.0"
