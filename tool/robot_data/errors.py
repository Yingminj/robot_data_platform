"""Exception types shared across the conversion pipeline.

Collected in one leaf module so that low-level decoders, the profile schema and
the alignment core can all raise and catch each other's errors without importing
each other.
"""

from __future__ import annotations


class ProfileError(ValueError):
    """A robot profile is internally inconsistent or unusable."""


class RecipeError(ValueError):
    """A conversion recipe is unknown or internally inconsistent."""


class MessageDecodeError(ValueError):
    """A ROS message could not be decoded."""


class ConversionError(RuntimeError):
    """A source episode cannot safely be converted."""
