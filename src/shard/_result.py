from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True, eq=False)
class Windows:
    prob: np.ndarray = field(repr=False)
    start_tick: np.ndarray = field(repr=False)
    end_tick: np.ndarray = field(repr=False)
    threshold: float

    def __len__(self) -> int:
        return len(self.prob)

    def __repr__(self) -> str:
        return f"Windows({len(self)}, suspicious={int(self.suspicious.sum())})"

    @property
    def suspicious(self) -> np.ndarray:
        return self.prob > self.threshold

    def to_pandas(self):
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError('to_pandas() needs pandas: pip install "shardac[pandas]"') from e
        return pd.DataFrame(
            {
                "start_tick": self.start_tick,
                "end_tick": self.end_tick,
                "prob": self.prob,
                "suspicious": self.suspicious,
            }
        )
