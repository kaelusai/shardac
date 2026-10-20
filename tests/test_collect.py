import numpy as np
import pytest

import shard

HEADER = ["sequence_id", "tick_index", "yaw", "pitch", "ticks_to_attack", "attack_this_tick", "ticks_since_attack"]


def write_window(path, rows):
    lines = [",".join(HEADER)] + [",".join(str(v) for v in r) for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def rows(seq, ticks, ticks_to_attack=0):
    return [[seq, t, t * 0.5, 0.0, ticks_to_attack, int(t % 4 == 0), t % 4] for t in ticks]


def test_overlapping_windows_merge(tmp_path):
    write_window(tmp_path / "w0.csv", rows(1, range(0, 20), ticks_to_attack=5))
    write_window(tmp_path / "w1.csv", rows(1, range(10, 30), ticks_to_attack=-3))
    r = shard.read_collect(tmp_path)
    assert r["tick_index"].tolist() == list(range(30))
    assert set(np.unique(r["segment"])) == {0}
    assert "ticks_to_attack" not in r


def test_hole_and_sequence_start_segments(tmp_path):
    write_window(tmp_path / "w0.csv", rows(1, range(0, 10)))
    write_window(tmp_path / "w1.csv", rows(1, range(10, 20)))
    write_window(tmp_path / "w2.csv", rows(2, range(20, 30)))
    r = shard.read_collect(tmp_path)
    assert r["segment"][:10].tolist() == [0] * 10
    assert r["segment"][10:20].tolist() == [1] * 10
    assert r["segment"][20:].tolist() == [2] * 10


def test_gaps_inside_a_window_stay_in_segment(tmp_path):
    write_window(tmp_path / "w0.csv", rows(1, [0, 1, 3, 4, 7]))
    r = shard.read_collect(tmp_path)
    assert r["segment"].tolist() == [0] * 5


def test_window_order_is_numeric(tmp_path):
    write_window(tmp_path / "w10.csv", rows(1, range(100, 110)))
    write_window(tmp_path / "w2.csv", rows(1, range(0, 10)))
    r = shard.read_collect(tmp_path)
    assert r["tick_index"][0] == 0


def test_recording_scores(tmp_path):
    write_window(tmp_path / "w0.csv", rows(1, range(0, 64)))
    r = shard.read_collect(tmp_path)
    assert list(shard.load("shard-2-nano").predict(r).end_tick) == [31, 47, 63]


def test_rejects_non_advancing_tick_index(tmp_path):
    write_window(tmp_path / "w0.csv", rows(1, [4, 4, 4]))
    with pytest.raises(ValueError, match="does not advance"):
        shard.read_collect(tmp_path)


def test_rejects_empty_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        shard.read_collect(tmp_path)
