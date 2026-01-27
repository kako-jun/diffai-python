# diffai-python の思想（Philosophy）

diffai-python は diffai-core の Python バインディングです。
AI/MLモデルファイル（PyTorch, Safetensors, NumPy, MATLAB）の比較機能を提供します。

## 構造

```
diffai-python/
├── src/lib.rs              # Rust PyO3 バインディング
├── python/diffai/          # Pythonモジュール
│   └── __init__.py         # re-export from ._diffai
├── Cargo.toml              # Rust 依存関係
├── pyproject.toml          # Python パッケージ設定
└── tests/                  # pytest テスト
```

## 開発

```bash
uv sync
maturin develop
pytest
```

## API

- `diff(old, new, **kwargs)` - 2つのオブジェクトを比較
- `diff_paths(old_path, new_path, **kwargs)` - 2つのファイル/ディレクトリを比較
- `format_output(results, format)` - 結果をフォーマット

## 重要

- diffai-core は crates.io から参照
- バージョンは diffai-core と同期
