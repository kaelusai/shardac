from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from ._protocol import INT_RANGES, SEGMENT, SEQUENCE, SHARD_2, TICK, Protocol


def _windows(directory: Path) -> list[Path]:
    found = [(int(m.group(1)), p) for p in directory.glob("w*.csv") if (m := re.fullmatch(r"w(\d+)\.csv", p.name))]
    return [p for _, p in sorted(found)]


def _read(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open(encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        data = np.loadtxt(f, delimiter=",", ndmin=2, dtype=np.float64)
    if data.size and data.shape[1] != len(header):
        raise ValueError(f"{path}: {data.shape[1]} values per row, {len(header)} columns in the header")
    return header, data.reshape(-1, len(header))


def _segments(keys: np.ndarray, adjacent: set[tuple[int, int, int, int]]) -> np.ndarray:
    segment = np.zeros(len(keys), dtype=np.int64)
    for i in range(1, len(keys)):
        pair = (int(keys[i - 1, 0]), int(keys[i - 1, 1]), int(keys[i, 0]), int(keys[i, 1]))
        segment[i] = segment[i - 1] + (pair not in adjacent)
    return segment


def read_collect(path, *, protocol: Protocol = SHARD_2) -> dict[str, np.ndarray]:
    directory = Path(path).expanduser()
    files = _windows(directory)
    if not files:
        raise FileNotFoundError(f"{directory}: no w*.csv windows found")
    tables = [_read(p) for p in files]
    header = tables[0][0]
    if any(h != header for h, _ in tables[1:]):
        raise ValueError(f"{directory}: windows have different columns")
    if SEQUENCE not in header or TICK not in header:
        raise ValueError(f"{directory}: windows lack {SEQUENCE} or {TICK}")
    seq_i, tick_i = header.index(SEQUENCE), header.index(TICK)
    blocks = [data for _, data in tables if len(data)]
    rows = np.concatenate(blocks) if blocks else np.zeros((0, len(header)))

    if any(len(b) > 1 and np.all(b[:, tick_i] == b[0, tick_i]) for b in blocks):
        raise ValueError(f"{directory}: {TICK} does not advance within a window, the recording format is not supported")
    keys = rows[:, [seq_i, tick_i]].astype(np.int64)
    adjacent = set()
    start = 0
    for block in blocks:
        k = keys[start : start + len(block)]
        adjacent.update(zip(k[:-1, 0].tolist(), k[:-1, 1].tolist(), k[1:, 0].tolist(), k[1:, 1].tolist(), strict=True))
        start += len(block)
    order = np.lexsort((np.arange(len(rows)), keys[:, 1], keys[:, 0]))
    keys, rows = keys[order], rows[order]
    first = np.ones(len(rows), dtype=bool)
    first[1:] = np.any(keys[1:] != keys[:-1], axis=1)
    keys, rows = keys[first], rows[first]
    segment = _segments(keys, adjacent)

    out = {}
    for i, name in enumerate(header):
        if name in protocol.window_columns:
            continue
        col = rows[:, i]
        kind = protocol.columns.get(name)
        out[name] = np.round(col).astype(np.int64) if kind == "bool" or kind in INT_RANGES else col
    out[SEGMENT] = segment
    return out
