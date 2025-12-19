"""diffai-python: AI/ML model diff tool

Compare PyTorch, Safetensors, NumPy, and MATLAB files with automatic
ML-aware analysis.
"""

from __future__ import annotations

from typing import Any

try:
    from diffai_python.diffai_python import __version__, diff, diff_paths, format_output
except ImportError:
    from diffai_python import __version__, diff, diff_paths, format_output


class DiffaiError(Exception):
    """Exception raised when a diffai operation fails."""

    pass


def diff_from_files(
    old_path: str, new_path: str, **kwargs: Any
) -> list[dict[str, Any]]:
    """Compare two files and return differences.

    This is a convenience function that wraps diff_paths().

    Args:
        old_path: Path to the old file
        new_path: Path to the new file
        **kwargs: Additional options passed to diff_paths

    Returns:
        List of difference dictionaries
    """
    return diff_paths(old_path, new_path, **kwargs)


__all__ = [
    "__version__",
    "diff",
    "diff_paths",
    "diff_from_files",
    "format_output",
    "DiffaiError",
]
