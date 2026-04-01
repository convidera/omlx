# SPDX-License-Identifier: Apache-2.0
"""
Root conftest.py — mock Apple-Silicon-only dependencies so that unit tests
for pure-Python modules (e.g. omlx.mcp.*) can run on Linux CI runners.
"""

import sys
import types
from unittest.mock import MagicMock


def _mock_mlx() -> None:
    """Insert MagicMock stubs for mlx and mlx_lm before any imports occur."""
    if "mlx.core" in sys.modules:
        # Already importable (macOS with MLX installed) — nothing to do.
        try:
            import mlx.core  # noqa: F401
            return
        except ImportError:
            pass

    # Build a minimal package tree that satisfies all sub-module imports
    # encountered in omlx/scheduler.py and friends.
    for pkg_name in ("mlx", "mlx_lm", "mlx_embeddings", "mlx_vlm"):
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = []  # mark as package
        sys.modules[pkg_name] = pkg

    # Attach commonly referenced sub-modules
    _submodules = [
        "mlx.core",
        "mlx.nn",
        "mlx.nn.layers",
        "mlx.optimizers",
        "mlx.utils",
        "mlx_lm.generate",
        "mlx_lm.utils",
        "mlx_lm.models",
        "mlx_lm.models.base",
        "mlx_lm.models.cache",
        "mlx_lm.sample_utils",
        "mlx_lm.tokenizer_utils",
        "mlx_embeddings.core",
        "mlx_vlm.utils",
    ]
    for name in _submodules:
        if name not in sys.modules:
            sys.modules[name] = MagicMock()

    # Make mlx.core attributes look real enough for isinstance checks
    mx = sys.modules["mlx.core"]
    mx.array = MagicMock  # type: ignore[attr-defined]


_mock_mlx()
