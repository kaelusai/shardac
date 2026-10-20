import json
import shutil
from importlib.resources import files
from pathlib import Path

import numpy as np
import pytest

import shard

NAME = "shard-2-nano"
WEIGHTS = Path(str(files("shard._weights").joinpath(NAME)))
VECTORS = json.loads((Path(__file__).parent / "vectors" / f"{NAME}.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def model():
    return shard.load(NAME)


def ticks(n, start=0, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "tick_index": np.arange(start, start + n),
        "yaw": np.cumsum(rng.normal(0, 3, n)),
        "pitch": rng.normal(0, 5, n),
        "attack_this_tick": (np.arange(n) % 4 == 0).astype(int),
        "ticks_since_attack": np.arange(n) % 4,
    }


def copy_model(tmp_path, **changes):
    shutil.copy(WEIGHTS / "model.onnx", tmp_path / "model.onnx")
    meta = json.loads((WEIGHTS / "metadata.json").read_text(encoding="utf-8"))
    meta.update(changes)
    (tmp_path / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    return tmp_path


def test_attributes(model):
    assert model.name == NAME
    assert model.protocol == "shard-2"
    assert model.threshold == 0.9
    assert (model.window, model.stride) == (32, 16)
    assert model.feature_names == (
        "delta_yaw",
        "delta_pitch",
        "attack_this_tick",
        "ticks_since_attack",
        "accel_yaw",
        "accel_pitch",
    )
    assert model.columns == ("yaw", "pitch", "attack_this_tick", "ticks_since_attack")
    assert model.version == VECTORS["version"]
    assert shard.models() == (NAME,)


@pytest.mark.parametrize("case", range(len(VECTORS["cases"])))
def test_reference_vectors(model, case):
    c = VECTORS["cases"][case]
    tol = VECTORS["tolerance"]
    x = {k: np.asarray(v) for k, v in c["input"].items()}
    np.testing.assert_allclose(model.features(x), c["features"], atol=tol["features"], rtol=0)
    w = model.predict(x)
    np.testing.assert_array_equal(w.start_tick, c["windows"]["start_tick"])
    np.testing.assert_array_equal(w.end_tick, c["windows"]["end_tick"])
    np.testing.assert_allclose(w.prob, c["windows"]["prob"], atol=tol["prob"], rtol=0)


def test_predict_matches_windows(model):
    t = ticks(100)
    w = model.predict(t)
    batch = model.windows(model.features(t), starts=w.start_tick)
    np.testing.assert_allclose(model(batch), w.prob, atol=1e-7)
    np.testing.assert_array_equal(w.suspicious, w.prob > model.threshold)


def test_windows_do_not_cross_sequences(model):
    a, b = ticks(40, 0), ticks(40, 100, seed=1)
    joined = {k: np.concatenate([a[k], b[k]]) for k in a}
    joined["sequence_id"] = np.r_[np.ones(40, int), np.full(40, 2)]
    w = model.predict(joined)
    assert list(w.start_tick) == [0, 100]
    assert np.all(w.end_tick - w.start_tick == 31)


def test_without_tick_index_rows_are_numbered(model):
    t = ticks(48)
    del t["tick_index"]
    assert list(model.predict(t).start_tick) == [0, 16]


def test_attack_windows_center_the_attack_and_respect_the_step(model):
    n = 100
    t = ticks(n)
    t["attack_this_tick"] = np.zeros(n, dtype=int)
    t["attack_this_tick"][[10, 20, 50, 60, 70]] = 1
    w = model.predict(t)
    assert list(w.start_tick) == [34, 54]
    assert list(w.end_tick) == [65, 85]
    assert len(model.predict({**t, "attack_this_tick": np.zeros(n, dtype=int)})) == 0


def test_slide_schedule_still_loads(tmp_path):
    m = shard.load(copy_model(tmp_path, stride=8, schedule={"type": "slide", "event": "attack_this_tick"}))
    t = ticks(100)
    t["attack_this_tick"] = np.zeros(100, dtype=int)
    t["attack_this_tick"][[10, 50]] = 1
    assert list(m.predict(t).end_tick) == [32, 40, 56, 64, 72, 80]


def test_short_input_has_no_windows(model):
    assert len(model.predict(ticks(10))) == 0


def test_pandas_roundtrip(model):
    pd = pytest.importorskip("pandas")
    w = model.predict(pd.DataFrame(ticks(64)))
    out = w.to_pandas()
    assert list(out.columns) == ["start_tick", "end_tick", "prob", "suspicious"]
    assert len(out) == len(w)


def test_windows_compare_by_identity(model):
    t = ticks(64)
    a, b = model.predict(t), model.predict(t)
    assert a != b
    assert a == a
    assert len(repr(a)) < 100


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda t: t.pop("yaw"), "missing column"),
        (lambda t: t.__setitem__("pitch", t["pitch"][:-1]), "different lengths"),
        (lambda t: t["yaw"].__setitem__(3, np.nan), "NaN"),
        (lambda t: t["yaw"].__setitem__(slice(3, 5), 1e39), "float32"),
        (lambda t: t.__setitem__("yaw", t["yaw"].astype(complex)), "numeric"),
        (lambda t: t["attack_this_tick"].__setitem__(3, 2), "only 0 and 1"),
        (lambda t: t.__setitem__("teleport_tick", np.full(40, 0.5)), "only 0 and 1"),
        (lambda t: t["ticks_since_attack"].__setitem__(3, 300), "u8"),
        (lambda t: t.__setitem__("tick_index", t["tick_index"][::-1].copy()), "strictly increasing"),
    ],
)
def test_input_errors(model, change, message):
    t = ticks(40)
    change(t)
    with pytest.raises(ValueError, match=message):
        model.predict(t)


def test_call_shape_error(model):
    with pytest.raises(ValueError, match=r"\[n, 32, 6\]"):
        model(np.zeros((2, 64, 6)))


def test_load_directory(tmp_path):
    m = shard.load(copy_model(tmp_path))
    assert m.source == str(tmp_path)
    assert m.name == NAME


def test_load_rejects_tampered_model(tmp_path):
    copy_model(tmp_path)
    (tmp_path / "model.onnx").write_bytes((WEIGHTS / "model.onnx").read_bytes() + b"\0")
    with pytest.raises(ValueError, match="sha256"):
        shard.load(tmp_path)


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"format": 2}, "format"),
        ({"protocol": "shard-1"}, "protocol"),
        ({"window": 24}, "graph input"),
        ({"schedule": {"type": "attack", "event": "attack_this_tick", "pre": 40}}, "pre"),
        ({"schedule": {"type": "attack", "event": "attack_this_tick"}}, "schedule"),
        ({"window": 10**6}, "window"),
        ({"stride": 0}, "stride"),
        ({"threshold": 1.5}, "threshold"),
        ({"features": [{"name": "a", "op": "jerk", "column": "yaw"}]}, "newer shardac"),
        ({"features": [{"name": "a", "op": "value", "column": "nope"}]}, "unknown column"),
        ({"features": [{"name": "a", "op": "value", "column": "yaw", "scale": 2}]}, "exactly the keys"),
    ],
)
def test_load_rejects_bad_metadata(tmp_path, changes, message):
    with pytest.raises(ValueError, match=message):
        shard.load(copy_model(tmp_path, **changes))


def test_load_unknown_source():
    with pytest.raises(FileNotFoundError):
        shard.load("./no/such/dir")
