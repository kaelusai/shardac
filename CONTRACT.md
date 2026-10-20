# Model format 1

This document specifies how a model packaged for shardac is run, so that it can be reimplemented in other languages. An implementation that follows it reproduces the reference vectors in `tests/vectors/` within the tolerances stated there.

## Model files

A model is a directory, a Hugging Face repository or a package resource with two files:

| File | Content |
|---|---|
| `model.onnx` | ONNX graph with one float32 input of shape `[batch, window, features]` and one output of shape `[batch, 1]` holding the probability of each window. Clipping, normalisation and the output sigmoid are part of the graph. The graph requires `batch >= 1`, so it is not run when there are no windows. |
| `metadata.json` | The description below. |

## metadata.json

| Key | Content |
|---|---|
| `format` | `1`, the version of this document. |
| `protocol` | The input protocol, `shard-2`. |
| `name`, `version` | Model identifier and version. |
| `io` | `{"input": {"name": ...}, "output": {"name": ...}}`, the tensor names in the graph. |
| `window`, `stride` | Window length in rows and the step between window starts. |
| `threshold` | Decision threshold in (0, 1). A window is suspicious when `prob > threshold`. |
| `features` | Ordered list of `{"name", "op", "column"}`. The order is the order of the last input dimension. |
| `schedule` | Optional. `{"type": "slide", "event": <column>}` selects the windows described below. Without it, windows start every `stride` rows. |
| `sha256` | SHA-256 of `model.onnx`. |

Unknown top-level keys are ignored. Unknown keys inside a feature, an unknown `op` and an unknown `protocol` are errors, and a model that uses them needs a newer implementation.

## Protocol `shard-2`

The input is the per-tick data recorded by the Shard 2 plugin for one player, one row per client tick, oldest first. A row corresponds to one movement packet received by the server, nominally 20 per second. Column names and wire types are those of the plugin's tick schema. The package lists them in `src/shard/_protocol.py`. Values of `bool` columns are 0 or 1, values of integer columns are whole numbers within the range of their type, and values of `f32` columns are finite and within the float32 range.

Columns with a special role:

| Column | Role |
|---|---|
| `sequence_id` | Optional. Rows with different values belong to different segments. |
| `segment` | Optional, not part of the plugin schema. Rows with different values belong to different segments. |
| `tick_index` | Optional. Counter of the recording, strictly increasing within a segment, used for the ticks reported with windows. It may skip values, for example on rows the plugin does not record, and a skip does not start a segment. Without it, rows are numbered `0, 1, ..., n - 1`. |
| `teleport_tick` | Optional. 1 on the first recorded row after the client applied a server teleport or a server-set rotation, otherwise 0. That row already carries the position and rotation set by the server. |

Columns used by Shard 2 Nano:

| Column | Definition |
|---|---|
| `yaw` | Yaw in degrees as sent by the client, not normalised to [-180, 180). |
| `pitch` | Pitch in degrees as sent by the client. |
| `attack_this_tick` | 1 if the player attacked an entity in this row, otherwise 0. |
| `ticks_since_attack` | Rows since the most recent attack: 0 on the attack row, 255 before the first attack, saturating at 255. |

## Segments

A segment is a maximal run of consecutive rows with equal `sequence_id` and equal `segment`. Features and windows are computed for each segment independently.

## Features

Each column is converted to float32 before any arithmetic, and all arithmetic is in float32. For row `t` of a segment, with `t = 0` its first row:

| `op` | Value |
|---|---|
| `value` | `x[t]` |
| `delta` | `d[t] = x[t] - x[t-1]`, 0 for `t = 0` and on rows with `teleport_tick = 1` |
| `accel` | `d[t] - d[t-1]` with `d` as above, 0 for `t = 0`, on rows with `teleport_tick = 1` and on the row after each of them |

## Windows

Rows are numbered `0 .. n - 1` within a segment of `n` rows. With a `slide` schedule, a window ends at every row `e` such that `e` is a multiple of `stride`, `window - 1 <= e < n`, and the rows `e - window + 1 .. e` contain at least one row where the `event` column is not 0. This is the sliding schedule of the Shard service. The service aligns `e` to its own row counter, so a reimplementation reproduces the set of windows up to that phase. Without a schedule, windows start at rows `0, stride, 2 * stride, ...` while `start + window <= n`. A window holds `window` consecutive rows of the features in the order of `features`. Windows of all segments are ordered by their start. A window reports the `tick_index` of its first and last row.

## Collect recordings

A recording of the Shard 2 plugin's collect mode is a directory of window files `w0.csv`, `w1.csv`, ... with one header line of column names. Windows overlap. They are merged by the key (`sequence_id`, `tick_index`), keeping the first copy in window order, and sorted by that key. The columns `ticks_to_attack` and `window_start` describe a window rather than a tick and are dropped. A new `segment` starts where two consecutive merged rows are not consecutive rows of any window file, which marks a hole in the recording.

## Reference vectors

`tests/vectors/<model>.json` gives, for each case, the per-tick input, the features of every row and, for every window, `start_tick`, `end_tick` and `prob`. Features must match within `tolerance.features` and probabilities within `tolerance.prob`, both absolute.
