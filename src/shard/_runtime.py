from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ._spec import Spec, parse

MODEL_FILE = "model.onnx"
METADATA_FILE = "metadata.json"
MAX_MODEL_BYTES = 256 * 2**20
MAX_METADATA_BYTES = 2**20
_BATCH = 4096


@dataclass(frozen=True)
class ModelFiles:
    model: bytes
    metadata: dict
    source: str


def check_size(name: str, size: int, source: str) -> None:
    limit = MAX_MODEL_BYTES if name == MODEL_FILE else MAX_METADATA_BYTES
    if size > limit:
        raise ValueError(f"{source}: {name} is {size} bytes, the limit is {limit}")


def read_dir(directory: Path, source: str) -> ModelFiles:
    data = {}
    for name in (MODEL_FILE, METADATA_FILE):
        p = directory / name
        if not p.is_file():
            raise FileNotFoundError(f"{source}: {name} not found in {directory}")
        check_size(name, p.stat().st_size, source)
        data[name] = p.read_bytes()
    return ModelFiles(data[MODEL_FILE], json.loads(data[METADATA_FILE].decode("utf-8")), source)


def verify(files: ModelFiles) -> Spec:
    spec = parse(files.metadata, files.source)
    digest = hashlib.sha256(files.model).hexdigest()
    if spec.sha256 != digest:
        raise ValueError(f"{files.source}: {MODEL_FILE} sha256 {digest} does not match metadata {spec.sha256}")
    return spec


class Session:
    def __init__(self, files: ModelFiles, spec: Spec, providers=None):
        import onnxruntime as ort

        self.session = ort.InferenceSession(files.model, providers=providers or ["CPUExecutionProvider"])
        self.input, self.output = spec.input_name, spec.output_name
        inputs = {i.name: i for i in self.session.get_inputs()}
        outputs = {o.name for o in self.session.get_outputs()}
        if set(inputs) != {self.input} or self.output not in outputs:
            raise ValueError(
                f"{files.source}: graph inputs {sorted(inputs)} and outputs {sorted(outputs)} do not match io"
            )
        shape = inputs[self.input].shape
        expected = [spec.window, len(spec.features)]
        if inputs[self.input].type != "tensor(float)" or len(shape) != 3 or list(shape[1:]) != expected:
            raise ValueError(f"{files.source}: graph input must be float32 [batch, {expected[0]}, {expected[1]}]")

    def run(self, windows: np.ndarray) -> np.ndarray:
        if len(windows) == 0:
            return np.zeros(0, dtype=np.float32)
        out = [
            self.session.run([self.output], {self.input: windows[i : i + _BATCH]})[0].reshape(-1)
            for i in range(0, len(windows), _BATCH)
        ]
        return np.concatenate(out)
