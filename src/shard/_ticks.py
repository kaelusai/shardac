from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np

from ._protocol import INT_RANGES, SEGMENT, Protocol

F32_MAX = float(np.finfo(np.float32).max)


def _check(name: str, a: np.ndarray, kind: str | None) -> None:
    if a.dtype.kind not in "biuf":
        raise ValueError(f"column {name} must be numeric, got dtype {a.dtype}")
    if a.dtype.kind == "f" and not np.all(np.isfinite(a)):
        raise ValueError(f"column {name} contains NaN or infinite values")
    if kind is None:
        return
    if kind == "f32" and np.any(np.abs(a) > F32_MAX):
        raise ValueError(f"column {name} contains values outside the float32 range")
    if kind == "bool" and not np.isin(a, (0, 1)).all():
        raise ValueError(f"column {name} must hold only 0 and 1")
    if kind in INT_RANGES:
        lo, hi = INT_RANGES[kind]
        if np.any(a != np.round(a)) or np.any(a < lo) or np.any(a > hi):
            raise ValueError(f"column {name} must hold whole numbers in [{lo}, {hi}] ({kind})")


def columns(ticks, protocol: Protocol, required: Iterable[str], optional: Iterable[str]) -> dict[str, np.ndarray]:
    if not isinstance(ticks, Mapping) and not hasattr(ticks, "columns"):
        raise TypeError(
            f"ticks must be a pandas DataFrame or a mapping of column name to array, got {type(ticks).__name__}"
        )
    names = set(ticks.columns) if hasattr(ticks, "columns") else set(ticks.keys())
    required = tuple(required)
    missing = [c for c in required if c not in names]
    if missing:
        raise ValueError(f"ticks is missing column(s) {', '.join(missing)}, required: {', '.join(required)}")
    out = {}
    for c in (*required, *optional):
        if c not in names or c in out:
            continue
        a = np.asarray(ticks[c])
        if a.ndim != 1:
            raise ValueError(f"column {c} must be one-dimensional, got shape {a.shape}")
        _check(c, a, protocol.columns.get(c, "i32" if c == SEGMENT else None))
        out[c] = a
    lengths = {c: len(a) for c, a in out.items()}
    if len(set(lengths.values())) > 1:
        raise ValueError(f"columns have different lengths: {lengths}")
    return out
