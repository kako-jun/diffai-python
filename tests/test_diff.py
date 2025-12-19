"""Tests for diffai-python."""

import pytest


def test_import():
    """Test that the module can be imported."""
    import diffai_python

    assert hasattr(diffai_python, "diff")
    assert hasattr(diffai_python, "diff_paths")
    assert hasattr(diffai_python, "format_output")
    assert hasattr(diffai_python, "__version__")


def test_diff_identical():
    """Test diffing identical objects."""
    import diffai_python

    obj = {"a": 1, "b": 2}
    results = diffai_python.diff(obj, obj)
    assert isinstance(results, list)
    assert len(results) == 0


def test_diff_added():
    """Test detecting added keys."""
    import diffai_python

    old = {"a": 1}
    new = {"a": 1, "b": 2}
    results = diffai_python.diff(old, new)
    assert len(results) >= 1
    added = [r for r in results if r["type"] == "Added" and r["path"] == "b"]
    assert len(added) == 1


def test_diff_removed():
    """Test detecting removed keys."""
    import diffai_python

    old = {"a": 1, "b": 2}
    new = {"a": 1}
    results = diffai_python.diff(old, new)
    assert len(results) >= 1
    removed = [r for r in results if r["type"] == "Removed" and r["path"] == "b"]
    assert len(removed) == 1


def test_diff_modified():
    """Test detecting modified values."""
    import diffai_python

    old = {"a": 1}
    new = {"a": 2}
    results = diffai_python.diff(old, new)
    assert len(results) >= 1
    modified = [r for r in results if r["type"] == "Modified" and r["path"] == "a"]
    assert len(modified) == 1
    assert modified[0]["old_value"] == 1
    assert modified[0]["new_value"] == 2


def test_diff_nested():
    """Test diffing nested objects."""
    import diffai_python

    old = {"nested": {"deep": {"value": 1}}}
    new = {"nested": {"deep": {"value": 2}}}
    results = diffai_python.diff(old, new)
    assert len(results) >= 1


def test_diff_with_epsilon():
    """Test epsilon option for numerical tolerance."""
    import diffai_python

    old = {"value": 1.0}
    new = {"value": 1.0001}

    results_without = diffai_python.diff(old, new)
    results_with = diffai_python.diff(old, new, epsilon=0.001)

    assert len(results_without) >= 1
    assert len(results_with) == 0


def test_diff_tensor_like():
    """Test handling tensor-like data."""
    import diffai_python

    old = {"layers": [{"weight": [1.0, 2.0, 3.0]}]}
    new = {"layers": [{"weight": [1.0, 2.0, 4.0]}]}
    results = diffai_python.diff(old, new)
    assert isinstance(results, list)


def test_format_output_json():
    """Test formatting results as JSON."""
    import diffai_python
    import json

    old = {"a": 1}
    new = {"a": 2}
    results = diffai_python.diff(old, new)
    formatted = diffai_python.format_output(results, "json")
    assert isinstance(formatted, str)
    parsed = json.loads(formatted)
    assert isinstance(parsed, list)


def test_format_output_diffai():
    """Test formatting results as diffai format."""
    import diffai_python

    old = {"a": 1}
    new = {"a": 2}
    results = diffai_python.diff(old, new)
    formatted = diffai_python.format_output(results, "diffai")
    assert isinstance(formatted, str)


def test_empty_objects():
    """Test diffing empty objects."""
    import diffai_python

    results = diffai_python.diff({}, {})
    assert isinstance(results, list)
    assert len(results) == 0
