from __future__ import annotations

import numpy as np

from ._engine import attack_starts, derive, segments, slide_starts, stack, window_starts
from ._protocol import SEGMENT, SEQUENCE, TELEPORT, TICK
from ._result import Windows
from ._runtime import ModelFiles, Session, verify
from ._ticks import columns


class Model:
    def __init__(self, files: ModelFiles, providers=None):
        spec = verify(files)
        self._spec = spec
        self.source = files.source
        self.name = spec.name
        self.version = spec.version
        self.protocol = spec.protocol.name
        self.window = spec.window
        self.stride = spec.stride
        self.threshold = spec.threshold
        self.feature_names = tuple(f.name for f in spec.features)
        self.columns = spec.columns
        self._session = Session(files, spec, providers)

    def __repr__(self) -> str:
        return f"<shard.Model {self.name} {self.version} from {self.source}>"

    def predict(self, ticks) -> Windows:
        cols, spans = self._read(ticks)
        n = len(cols[self.columns[0]])
        ticks_of = cols[TICK].astype(np.int64) if TICK in cols else np.arange(n, dtype=np.int64)
        parts, starts = [], []
        event = self._spec.event
        for lo, hi in spans:
            if event is None:
                offsets = window_starts(hi - lo, self.window, self.stride)
            elif self._spec.schedule == "attack":
                hits = np.flatnonzero(cols[event][lo:hi] != 0)
                offsets = attack_starts(hits, hi - lo, self.window, self._spec.pre, self.stride)
            else:
                hits = np.flatnonzero(cols[event][lo:hi] != 0)
                offsets = slide_starts(hits, hi - lo, self.window, self.stride)
            if len(offsets):
                parts.append(stack(derive(self._spec.features, cols, lo, hi), offsets, self.window))
                starts.append(lo + np.asarray(offsets, dtype=np.int64))
        stacked = np.concatenate(parts) if parts else np.zeros((0, self.window, len(self.feature_names)), np.float32)
        start_rows = np.concatenate(starts) if starts else np.zeros(0, dtype=np.int64)
        return Windows(
            prob=self(stacked),
            start_tick=ticks_of[start_rows],
            end_tick=ticks_of[start_rows + self.window - 1],
            threshold=self.threshold,
        )

    def features(self, ticks) -> np.ndarray:
        cols, spans = self._read(ticks)
        if not spans:
            return np.zeros((0, len(self.feature_names)), dtype=np.float32)
        return np.concatenate([derive(self._spec.features, cols, lo, hi) for lo, hi in spans])

    def windows(self, features: np.ndarray, *, starts=None) -> np.ndarray:
        features = np.asarray(features, dtype=np.float32)
        if starts is None:
            return stack(features, window_starts(len(features), self.window, self.stride), self.window)
        rows = np.asarray(starts)
        if rows.ndim != 1 or (rows.size and rows.dtype.kind not in "iu"):
            raise ValueError("starts must be a one-dimensional sequence of row indices")
        last = len(features) - self.window
        if np.any(rows < 0) or np.any(rows > last):
            raise ValueError(f"starts must be in [0, {last}] so that every window has {self.window} rows")
        return stack(features, rows.tolist(), self.window)

    def __call__(self, windows: np.ndarray) -> np.ndarray:
        windows = np.asarray(windows, dtype=np.float32)
        expected = (self.window, len(self.feature_names))
        if windows.ndim != 3 or windows.shape[1:] != expected:
            raise ValueError(f"windows must have shape [n, {expected[0]}, {expected[1]}], got {list(windows.shape)}")
        return self._session.run(windows)

    def _read(self, ticks) -> tuple[dict[str, np.ndarray], list[tuple[int, int]]]:
        cols = columns(ticks, self._spec.protocol, self.columns, (TICK, SEQUENCE, SEGMENT, TELEPORT))
        n = len(cols[self.columns[0]])
        return cols, segments(cols, n)
