# diffai

Python bindings for diffai - AI/ML model diff tool for PyTorch, Safetensors, NumPy, and MATLAB tensor comparison.

## Installation

```bash
pip install diffai
```

## Usage

```python
import diffai_python

# Compare two model configurations
old = {"layers": [{"weight": [1.0, 2.0, 3.0]}]}
new = {"layers": [{"weight": [1.0, 2.0, 4.0]}]}
results = diffai_python.diff(old, new)

# Compare files
results = diffai_python.diff_paths("model_v1.pt", "model_v2.pt")
```

## License

MIT
