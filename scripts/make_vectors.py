import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import shard  # noqa: E402

MODEL = "shard-2-nano"
OUT = ROOT / "tests" / "vectors" / f"{MODEL}.json"


def synthetic(seed: int, n: int, sequence_break: int, teleport_at: int, gap_at: int):
    rng = np.random.default_rng(seed)
    yaw = np.cumsum(rng.normal(0, 4, n)) + 90.0
    pitch = np.clip(np.cumsum(rng.normal(0, 1.5, n)), -90, 90)
    attack = (rng.random(n) < 0.15).astype(int)
    since = np.empty(n, dtype=int)
    last = None
    for i in range(n):
        if attack[i]:
            last = i
        since[i] = 255 if last is None else min(255, i - last)
    teleport = np.zeros(n, dtype=int)
    teleport[teleport_at] = 1
    sequence = np.where(np.arange(n) < sequence_break, 1, 2)
    tick = np.arange(1000, 1000 + n)
    tick[gap_at:] += 1
    return {
        "sequence_id": sequence,
        "tick_index": tick,
        "yaw": np.round(yaw, 3),
        "pitch": np.round(pitch, 3),
        "attack_this_tick": attack,
        "ticks_since_attack": since,
        "teleport_tick": teleport,
    }


def main():
    model = shard.load(MODEL)
    cases = []
    for seed in (1, 2, 3):
        ticks = synthetic(seed, 160, sequence_break=90, teleport_at=40, gap_at=60)
        windows = model.predict(ticks)
        cases.append(
            {
                "input": {k: v.tolist() for k, v in ticks.items()},
                "features": model.features(ticks).tolist(),
                "windows": {
                    "start_tick": windows.start_tick.tolist(),
                    "end_tick": windows.end_tick.tolist(),
                    "prob": [float(p) for p in windows.prob],
                },
            }
        )
    doc = {
        "model": model.name,
        "version": model.version,
        "protocol": model.protocol,
        "tolerance": {"features": 1e-6, "prob": 1e-5},
        "cases": cases,
    }
    OUT.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    print(f"{OUT}: {len(cases)} cases, {sum(len(c['windows']['prob']) for c in cases)} windows")


if __name__ == "__main__":
    main()
