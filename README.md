# shardac

Open-weight models from Shard for detecting cheats in Minecraft: Java Edition from player data received by the server.

The package ships Shard 2 Nano, the smallest model of the Shard 2 family. The model card, evaluation and limitations are published on [Hugging Face](https://huggingface.co/kaelusai/shard-2-nano).

## Installation

```bash
pip install shardac
```

The bundled model weights are included in the package. Loading models from Hugging Face requires the `hub` extra:

```bash
pip install "shardac[hub]"
```

The distribution is named `shardac` and is imported as `shard`. It cannot be installed in the same environment as the unrelated `shard` package from PyPI, which uses the same import name.

## Quick start

```python
import shard

model = shard.load("shard-2-nano")
windows = model.predict(ticks)

for start, end, prob in zip(windows.start_tick, windows.end_tick, windows.prob):
    print(f"ticks {start}-{end}: {prob:.3f}")
```

`ticks` holds the data of one player with one row per client tick, oldest first, as a pandas DataFrame or a mapping of column name to array. Column names follow the tick schema of the Shard 2 plugin. Shard 2 Nano reads:

| Column | Required | Definition |
|---|---|---|
| `yaw` | yes | View yaw in degrees as sent by the client, not normalised to [-180, 180). |
| `pitch` | yes | View pitch in degrees. |
| `attack_this_tick` | yes | 1 if the player attacked an entity during the tick, otherwise 0. |
| `ticks_since_attack` | yes | Ticks elapsed since the most recent attack, 0 on the attack tick, 255 before the first attack, saturating at 255. |
| `teleport_tick` | no | 1 on the first tick after the client applied a server teleport or a server-set rotation, otherwise 0. |
| `tick_index` | no | Tick counter, reported with windows. |
| `sequence_id`, `segment` | no | A change of either value starts a new segment, and windows never span segments. |

A session directory written by the plugin's collect mode is read directly:

```python
ticks = shard.read_collect("path/to/collect/<session>")
```

## Results

`model.predict` cuts windows by the schedule in the model's metadata. For Shard 2 Nano these are windows of 32 ticks around attacks, with the attack on the 17th tick and at least 16 ticks between the attacks of consecutive windows. It returns `Windows`:

- `windows.prob`: the probability of each window.
- `windows.start_tick`, `windows.end_tick`: the first and last tick each window covers.
- `windows.suspicious`: windows with a probability above the 0.9 threshold.
- `windows.to_pandas()`: one row per window, requires pandas.

A single window above the threshold is not sufficient grounds for punishing a player. How window probabilities are aggregated over a session is described in the [Shard 2 methodology](https://shard.ac/methodology/shard-2).

## Lower-level API

```python
features = model.features(ticks)
batch = model.windows(features, starts=[100, 340, 512])  # windows starting at given rows
batch = model.windows(features)  # a window every stride rows
prob = model(batch)
```

Deltas and accelerations of a row depend on the two rows before it, so windows are cut from features computed over the whole recording. Features computed over exactly one window's rows differ on its first two rows.

`model.features` computes the features of every row, `model.windows` treats its input as one contiguous segment, and calling the model on a batch of shape `[n, window, features]` returns the probability of each window. `model.feature_names`, `model.columns`, `model.window`, `model.stride` and `model.threshold` describe the model.

## Loading

```python
shard.models()  # names of the bundled models
shard.load("shard-2-nano")  # bundled model
shard.load("./models/shard-2-nano")  # local directory with model.onnx and metadata.json
shard.load("kaelusai/shard-2-nano", revision="v1.0.0")
shard.load("shard-2-nano", providers=["CUDAExecutionProvider"])
```

A model is an ONNX graph and a `metadata.json` that declares its features from the plugin's tick schema, so models of other authors in this format load the same way. The loader verifies the SHA-256 of `model.onnx` against `metadata.json`, which guards against corrupted files but not against a malicious repository. For repositories of other authors, pin `revision` to a commit. A string of the form `owner/name` that is neither a bundled model nor an existing directory is treated as a Hugging Face repository id, so relative paths should start with `./`.

GPU execution requires `onnxruntime-gpu` in place of `onnxruntime`. The two packages cannot be installed in the same environment.

## Ports

[CONTRACT.md](CONTRACT.md) specifies the model format, the protocol columns, the feature operations, windowing and the reading of collect recordings for implementations in other languages. Reference vectors are in [tests/vectors](tests/vectors).

## License

The code, the model weights and the metadata are released under the Apache License 2.0.
