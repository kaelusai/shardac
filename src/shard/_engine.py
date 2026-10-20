from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise

import numpy as np

from ._protocol import SEGMENT, SEQUENCE, TELEPORT, TICK
from ._spec import Feature


def _diff(x: np.ndarray) -> np.ndarray:
    out = np.zeros_like(x)
    out[1:] = x[1:] - x[:-1]
    return out


def derive(features: tuple[Feature, ...], cols: dict[str, np.ndarray], lo: int, hi: int) -> np.ndarray:
    teleport = cols.get(TELEPORT)
    teleported = np.zeros(hi - lo, dtype=bool) if teleport is None else teleport[lo:hi] != 0
    after = teleported.copy()
    after[1:] |= teleported[:-1]
    out = np.empty((hi - lo, len(features)), dtype=np.float32)
    for i, f in enumerate(features):
        x = np.asarray(cols[f.column][lo:hi], dtype=np.float32)
        if f.op == "value":
            out[:, i] = x
            continue
        delta = _diff(x)
        delta[teleported] = 0.0
        if f.op == "delta":
            out[:, i] = delta
            continue
        accel = _diff(delta)
        accel[after] = 0.0
        out[:, i] = accel
    return out


def segments(cols: dict[str, np.ndarray], n: int) -> list[tuple[int, int]]:
    if n == 0:
        return []
    change = np.zeros(n - 1, dtype=bool)
    for key in (SEQUENCE, SEGMENT):
        if key in cols:
            change |= cols[key][1:] != cols[key][:-1]
    bounds = [0, *(np.flatnonzero(change) + 1).tolist(), n]
    spans = list(pairwise(bounds))
    tick = cols.get(TICK)
    if tick is not None:
        for lo, hi in spans:
            step = np.diff(tick[lo:hi])
            if np.any(step <= 0):
                i = lo + int(np.argmax(step <= 0)) + 1
                raise ValueError(
                    f"{TICK} must be strictly increasing within a segment, got {tick[i - 1]} then {tick[i]} at row {i}"
                )
    return spans


def window_starts(n: int, window: int, stride: int) -> range:
    return range(0, n - window + 1, stride)


def slide_starts(events: np.ndarray, n: int, window: int, stride: int) -> list[int]:
    ends = set()
    for a in events.tolist():
        e = -(-a // stride) * stride
        while e <= a + window - 1:
            if window - 1 <= e < n:
                ends.add(e)
            e += stride
    return [e - window + 1 for e in sorted(ends)]


def attack_starts(events: np.ndarray, n: int, window: int, pre: int, step: int) -> list[int]:
    starts, taken = [], None
    for a in events.tolist():
        if taken is not None and a - taken < step:
            continue
        taken = a
        if a - pre >= 0 and a - pre + window <= n:
            starts.append(a - pre)
    return starts


def stack(features: np.ndarray, starts: Sequence[int], window: int) -> np.ndarray:
    if not len(starts):
        return np.zeros((0, window, features.shape[1]), dtype=np.float32)
    return np.stack([features[s : s + window] for s in starts]).astype(np.float32, copy=False)
