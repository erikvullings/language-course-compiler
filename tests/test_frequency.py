"""wordfreq cBpack reader behavior."""

from __future__ import annotations

import msgpack
import pytest

from course_compiler.frequency import load_frequencies


def _write_cbpack(path, buckets, header=None):
    payload = [header or {"format": "cB", "version": 1}, *buckets]
    path.write_bytes(msgpack.dumps(payload))
    return path


def test_ranks_are_assigned_most_frequent_first(tmp_path):
    # bucket index in data[1:] is the centibel; lower index = more frequent
    path = _write_cbpack(tmp_path / "f.msgpack", [[], ["de"], ["het", "huis"]])
    table = load_frequencies(path)

    assert table["de"].rank == 1
    assert table["het"].rank == 2
    assert table["huis"].rank == 3
    assert table["de"].source == "wordfreq"


def test_zipf_decreases_with_rarity(tmp_path):
    path = _write_cbpack(tmp_path / "f.msgpack", [[], ["de"], ["zeldzaam"]])
    table = load_frequencies(path)
    assert table["de"].zipf > table["zeldzaam"].zipf


def test_rejects_non_cbpack_file(tmp_path):
    path = tmp_path / "bad.msgpack"
    path.write_bytes(msgpack.dumps([{"format": "other"}, []]))
    with pytest.raises(ValueError):
        load_frequencies(path)
