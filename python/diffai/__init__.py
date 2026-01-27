"""
diffai - AI/ML model diff tool for PyTorch, Safetensors, NumPy, and MATLAB tensor comparison

Example:
    import diffai

    old = {"layers": [{"weight": [1.0, 2.0, 3.0]}]}
    new = {"layers": [{"weight": [1.0, 2.0, 4.0]}]}

    results = diffai.diff(old, new)
    for change in results:
        print(f"{change['type']}: {change['path']}")
"""

from ._diffai import (
    diff,
    diff_paths,
    format_output,
)

__all__ = [
    "diff",
    "diff_paths",
    "format_output",
]
__version__ = "0.5.0"
