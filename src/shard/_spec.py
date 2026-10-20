from __future__ import annotations

import math
from dataclasses import dataclass

from ._protocol import PROTOCOLS, Protocol

FORMAT = 1
OPS = ("value", "delta", "accel")
MAX_WINDOW = 4096
MAX_FEATURES = 256
FEATURE_KEYS = {"name", "op", "column"}


@dataclass(frozen=True)
class Feature:
    name: str
    op: str
    column: str


@dataclass(frozen=True)
class Spec:
    name: str
    version: str
    protocol: Protocol
    window: int
    stride: int
    threshold: float
    features: tuple[Feature, ...]
    event: str | None
    schedule: str | None
    pre: int
    input_name: str
    output_name: str
    sha256: str

    @property
    def columns(self) -> tuple[str, ...]:
        names = [f.column for f in self.features] + ([self.event] if self.event else [])
        return tuple(dict.fromkeys(names))


def _int(meta: dict, key: str, lo: int, hi: int, source: str) -> int:
    value = meta.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or not lo <= value <= hi:
        raise ValueError(f"{source}: {key} must be an integer in [{lo}, {hi}], got {value!r}")
    return value


def _number(value, what: str, source: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"{source}: {what} must be a finite number, got {value!r}")
    return float(value)


def _text(meta: dict, key: str, source: str) -> str:
    value = meta.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{source}: {key} must be a non-empty string")
    return value


def parse(meta: dict, source: str) -> Spec:
    if not isinstance(meta, dict):
        raise ValueError(f"{source}: metadata.json must hold an object")
    fmt = meta.get("format")
    if fmt != FORMAT:
        raise ValueError(f"{source}: metadata format {fmt!r} is not supported, this shardac reads format {FORMAT}")
    protocol = PROTOCOLS.get(meta.get("protocol"))
    if protocol is None:
        raise ValueError(
            f"{source}: protocol {meta.get('protocol')!r} is not supported by this shardac, "
            f"known: {', '.join(PROTOCOLS)}. Upgrade shardac."
        )
    window = _int(meta, "window", 1, MAX_WINDOW, source)
    stride = _int(meta, "stride", 1, window, source)
    threshold = _number(meta.get("threshold"), "threshold", source)
    if not 0 < threshold < 1:
        raise ValueError(f"{source}: threshold must be in (0, 1), got {threshold}")

    raw = meta.get("features")
    if not isinstance(raw, list) or not 1 <= len(raw) <= MAX_FEATURES:
        raise ValueError(f"{source}: features must be a list of 1 to {MAX_FEATURES} entries")
    features = []
    for i, f in enumerate(raw):
        if not isinstance(f, dict) or set(f) != FEATURE_KEYS:
            raise ValueError(f"{source}: feature {i} must have exactly the keys {sorted(FEATURE_KEYS)}")
        if f["op"] not in OPS:
            raise ValueError(f"{source}: feature {f['name']!r} uses op {f['op']!r}, which needs a newer shardac")
        if f["column"] not in protocol.columns:
            raise ValueError(f"{source}: feature {f['name']!r} reads unknown column {f['column']!r} of {protocol.name}")
        features.append(Feature(str(f["name"]), f["op"], f["column"]))
    if len({f.name for f in features}) != len(features):
        raise ValueError(f"{source}: feature names must be unique")

    event, mode, pre = None, None, 0
    schedule = meta.get("schedule")
    if schedule is not None:
        shapes = {"slide": {"type", "event"}, "attack": {"type", "event", "pre"}}
        if not isinstance(schedule, dict) or shapes.get(schedule.get("type")) != set(schedule):
            raise ValueError(
                f'{source}: schedule must be {{"type": "slide", "event": <column>}} '
                f'or {{"type": "attack", "event": <column>, "pre": <rows>}}'
            )
        if schedule["event"] not in protocol.columns:
            raise ValueError(f"{source}: schedule event {schedule['event']!r} is not a column of {protocol.name}")
        event, mode = schedule["event"], schedule["type"]
        if mode == "attack":
            pre = _int(schedule, "pre", 0, window - 1, source)

    io = meta.get("io")
    try:
        input_name, output_name = io["input"]["name"], io["output"]["name"]
    except (TypeError, KeyError):
        raise ValueError(f"{source}: io must name the input and output tensors") from None

    return Spec(
        name=_text(meta, "name", source),
        version=_text(meta, "version", source),
        protocol=protocol,
        window=window,
        stride=stride,
        threshold=threshold,
        features=tuple(features),
        event=event,
        schedule=mode,
        pre=pre,
        input_name=str(input_name),
        output_name=str(output_name),
        sha256=_text(meta, "sha256", source),
    )
