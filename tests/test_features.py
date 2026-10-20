import numpy as np
import pytest

import shard


@pytest.fixture(scope="module")
def model():
    return shard.load("shard-2-nano")


def ticks(yaw, pitch, attack=None, since=None, **extra):
    n = len(yaw)
    return {
        "yaw": np.asarray(yaw, dtype=float),
        "pitch": np.asarray(pitch, dtype=float),
        "attack_this_tick": np.zeros(n, dtype=int) if attack is None else np.asarray(attack),
        "ticks_since_attack": np.full(n, 255) if since is None else np.asarray(since),
        **{k: np.asarray(v) for k, v in extra.items()},
    }


def test_ops_shape_and_order(model):
    f = model.features(ticks([0, 10, 25, 30], [0, -1, -3, -3], [0, 1, 0, 0], [255, 0, 1, 2]))
    assert f.dtype == np.float32
    assert f.shape == (4, 6)
    np.testing.assert_array_equal(f[:, 0], [0, 10, 15, 5])
    np.testing.assert_array_equal(f[:, 1], [0, -1, -2, 0])
    np.testing.assert_array_equal(f[:, 2], [0, 1, 0, 0])
    np.testing.assert_array_equal(f[:, 3], [255, 0, 1, 2])
    np.testing.assert_array_equal(f[:, 4], [0, 10, 5, -10])
    np.testing.assert_array_equal(f[:, 5], [0, -1, -1, 2])


def test_teleport_zeroes_deltas_and_next_accel(model):
    f = model.features(ticks([0, 10, 200, 210, 215], [0] * 5, teleport_tick=[0, 0, 1, 0, 0]))
    np.testing.assert_array_equal(f[:, 0], [0, 10, 0, 10, 5])
    np.testing.assert_array_equal(f[:, 4], [0, 10, 0, 0, -5])


def test_sequence_and_segment_restart_deltas(model):
    f = model.features(ticks([0, 10, 30, 35, 50], [0] * 5, sequence_id=[1, 1, 2, 2, 2]))
    np.testing.assert_array_equal(f[:, 0], [0, 10, 0, 5, 15])
    f = model.features(ticks([0, 10, 30, 35], [0] * 4, segment=[0, 0, 1, 1]))
    np.testing.assert_array_equal(f[:, 0], [0, 10, 0, 5])


def test_tick_gaps_do_not_split(model):
    f = model.features(ticks([0, 10, 30], [0] * 3, tick_index=[5, 6, 9]))
    np.testing.assert_array_equal(f[:, 0], [0, 10, 20])


def test_windows_stride_and_short_input(model):
    f = np.arange(50 * 6, dtype=np.float32).reshape(50, 6)
    w = model.windows(f)
    assert w.shape == (2, model.window, 6)
    np.testing.assert_array_equal(w[1, 0], f[model.stride])
    assert model.windows(f[:31]).shape == (0, model.window, 6)


def test_windows_at_given_starts(model):
    f = np.arange(50 * 6, dtype=np.float32).reshape(50, 6)
    w = model.windows(f, starts=[18, 0, 5])
    assert w.shape == (3, model.window, 6)
    np.testing.assert_array_equal(w[0, 0], f[18])
    np.testing.assert_array_equal(w[2, -1], f[5 + model.window - 1])
    assert model.windows(f, starts=[]).shape == (0, model.window, 6)


@pytest.mark.parametrize("starts", [[-1], [19], [1.5], [[0, 1]]])
def test_windows_reject_bad_starts(model, starts):
    with pytest.raises(ValueError, match="starts"):
        model.windows(np.zeros((50, 6), dtype=np.float32), starts=starts)


@pytest.mark.parametrize("tick", [[1, 2, 2], [3, 2, 1]])
def test_tick_index_must_increase_within_segment(model, tick):
    with pytest.raises(ValueError, match="strictly increasing"):
        model.features(ticks([0, 1, 2], [0] * 3, tick_index=tick))
