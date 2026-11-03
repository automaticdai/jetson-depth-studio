import numpy as np

from depthstudio.export import write_ply
from depthstudio.viz import colorize_depth, compose


def test_colorize_depth_marks_invalid_black(frame):
    img = colorize_depth(frame.depth, frame.depth_scale)
    assert img.shape == frame.depth.shape + (3,) and img.dtype == np.uint8
    assert not img[:4, :4].any()
    assert img[10:, 10:].any()


def test_compose_side_by_side(frame):
    img = compose(frame, fps=29.9)
    assert img.shape == (frame.depth.shape[0], frame.depth.shape[1] * 2, 3)


def test_write_ply_binary_with_colour(tmp_path, frame):
    path = tmp_path / "p.ply"
    n = write_ply(str(path), frame)
    assert n == int((frame.depth > 0).sum())
    data = path.read_bytes()
    header, body = data.split(b"end_header\n", 1)
    assert b"element vertex %d" % n in header and b"property uchar red" in header
    assert len(body) == n * (3 * 4 + 3)
