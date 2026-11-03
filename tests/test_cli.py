import os
import socket
import threading
import time
import urllib.request

import pytest

from depthstudio import cli
from depthstudio.record import Recorder
from depthstudio.stream import MjpegServer

from conftest import make_frame


@pytest.fixture
def recording(tmp_path):
    path = tmp_path / "rec"
    with Recorder(str(path), source_info={"source": "test"}, lossless=True) as rec:
        for i in range(20):
            rec.write(make_frame(i, t=100.0 + i / 30.0))
    return str(path)


def test_info(recording, capsys):
    assert cli.main(["info", recording]) == 0
    out = capsys.readouterr().out
    assert "frames    : 20" in out and "IMU       : 40 samples" in out


def test_snapshot_from_recording(recording, tmp_path):
    out = tmp_path / "snap"
    assert cli.main(["snapshot", "-s", recording, "-o", str(out), "--warmup", "3"]) == 0
    assert {"color.png", "depth.png", "depth_color.jpg", "points.ply", "meta.json"} <= set(os.listdir(out))


def test_record_from_recording(recording, tmp_path):
    out = tmp_path / "copy"
    assert cli.main(["record", "-s", recording, "-o", str(out), "-n", "10"]) == 0
    assert cli.main(["info", str(out)]) == 0
    assert len(os.listdir(out / "depth")) == 10


def test_unknown_source_is_an_error(capsys):
    assert cli.main(["snapshot", "-s", "nope"]) == 1
    assert "unknown source" in capsys.readouterr().err


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_mjpeg_server_serves_page_snapshot_and_stream():
    server = MjpegServer("127.0.0.1", _free_port())
    server.start()
    try:
        def feed():
            for i in range(50):
                server.publish(b"\xff\xd8jpeg%d\xff\xd9" % i)
                time.sleep(0.02)
        threading.Thread(target=feed, daemon=True).start()
        base = "http://127.0.0.1:%d" % server.port
        assert b"/stream" in urllib.request.urlopen(base + "/", timeout=5).read()
        snap = urllib.request.urlopen(base + "/snapshot.jpg", timeout=5)
        assert snap.headers["Content-Type"] == "image/jpeg" and snap.read().startswith(b"\xff\xd8jpeg")
        stream = urllib.request.urlopen(base + "/stream", timeout=5)
        assert "multipart/x-mixed-replace" in stream.headers["Content-Type"]
        assert stream.read(200).count(b"--frame") >= 1
        stream.close()
    finally:
        server.close()
