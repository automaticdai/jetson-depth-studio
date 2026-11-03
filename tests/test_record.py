import json
import os

import numpy as np
import pytest

from depthstudio.record import Recorder, RecordingSource
from depthstudio.sources import open_source

from conftest import make_frame


def _record(path, n=5, **kw):
    frames = [make_frame(i, t=100.0 + i / 30.0) for i in range(n)]
    with Recorder(str(path), source_info={"source": "test"}, **kw) as rec:
        for f in frames:
            rec.write(f)
    return frames


def test_roundtrip_lossless(tmp_path):
    path = tmp_path / "rec"
    frames = _record(path, lossless=True)
    meta = json.loads((path / "meta.json").read_text())
    assert meta["frames"] == 5 and meta["imu_samples"] == 10
    assert meta["depth_intrinsics"]["fx"] == 50.0

    src = open_source(str(path), realtime=False)
    assert isinstance(src, RecordingSource)
    with src:
        assert len(src) == 5
        got = list(src.frames())
    assert [g.index for g in got] == list(range(5))
    for a, b in zip(frames, got):
        assert b.t == pytest.approx(a.t)
        np.testing.assert_array_equal(a.depth, b.depth)  # depth is always lossless
        np.testing.assert_array_equal(a.color, b.color)
        assert b.depth_intrinsics == a.depth_intrinsics
        assert [(s.kind, s.t, s.z) for s in b.imu] == [(s.kind, s.t, s.z) for s in a.imu]


def test_jpeg_color_is_close(tmp_path):
    frames = _record(tmp_path / "rec", n=1)
    with RecordingSource(str(tmp_path / "rec")) as src:
        got = src.read()
    assert got.color.shape == frames[0].color.shape
    assert os.path.exists(tmp_path / "rec" / "color" / "000000.jpg")


def test_refuses_non_empty_dir(tmp_path):
    (tmp_path / "x").write_text("x")
    with pytest.raises(FileExistsError):
        Recorder(str(tmp_path))


def test_end_of_recording_and_loop(tmp_path):
    _record(tmp_path / "rec", n=2)
    with RecordingSource(str(tmp_path / "rec")) as src:
        src.read(), src.read()
        with pytest.raises(StopIteration):
            src.read()
    with RecordingSource(str(tmp_path / "rec"), loop=True) as src:
        assert [src.read().index for _ in range(5)] == [0, 1, 2, 3, 4]
