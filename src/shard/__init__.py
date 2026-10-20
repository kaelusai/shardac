from __future__ import annotations

import json
from importlib.resources import files as _package_files
from pathlib import Path

from . import _hub
from ._collect import read_collect
from ._model import Model
from ._result import Windows
from ._runtime import METADATA_FILE, MODEL_FILE, ModelFiles, read_dir

__version__ = "1.0.0"
__all__ = ["Model", "Windows", "load", "models", "read_collect"]

_WEIGHTS = _package_files("shard._weights")


def models() -> tuple[str, ...]:
    return tuple(sorted(d.name for d in _WEIGHTS.iterdir() if d.is_dir() and d.joinpath(METADATA_FILE).is_file()))


def _bundled(name: str) -> ModelFiles:
    d = _WEIGHTS.joinpath(name)
    return ModelFiles(
        d.joinpath(MODEL_FILE).read_bytes(),
        json.loads(d.joinpath(METADATA_FILE).read_text(encoding="utf-8")),
        f"shardac:{name}",
    )


def load(source: str | Path, *, revision: str | None = None, providers=None) -> Model:
    """Accepts a bundled model name, a local directory or a Hugging Face repository id."""
    if isinstance(source, str) and source in models():
        if revision is not None:
            raise ValueError("revision applies only to a Hugging Face repository id")
        return Model(_bundled(source), providers)
    path = Path(source).expanduser()
    if path.is_dir():
        if revision is not None:
            raise ValueError("revision applies only to a Hugging Face repository id")
        return Model(read_dir(path, str(path)), providers)
    text = str(source)
    if text.count("/") == 1 and not text.startswith((".", "/", "~")):
        return Model(_hub.download(text, revision), providers)
    raise FileNotFoundError(
        f"{source} is not a bundled model ({', '.join(models())}), a model directory or a Hugging Face repository id"
    )
