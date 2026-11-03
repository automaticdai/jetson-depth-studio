import numpy as np
import pytest

from depthstudio import Frame, Intrinsics


def test_intrinsics_roundtrip_and_K():
    i = Intrinsics(640, 480, 600.0, 601.0, 320.5, 240.5, model="brown_conrady", coeffs=[0.1, 0, 0, 0, 0])
    assert Intrinsics.from_dict(i.to_dict()) == i
    np.testing.assert_allclose(i.K, [[600.0, 0, 320.5], [0, 601.0, 240.5], [0, 0, 1]])


def test_depth_m(frame):
    z = frame.depth_m()
    assert z.dtype == np.float32
    assert z[10, 10] == pytest.approx(frame.depth[10, 10] * 0.001)
    assert Frame(index=0, t=0.0).depth_m() is None


def test_point_cloud_backprojects_pinhole():
    depth = np.zeros((3, 3), np.uint16)
    depth[1, 1] = 2000  # principal point, 2 m
    depth[0, 2] = 1000  # u=2, v=0, 1 m
    intr = Intrinsics(3, 3, fx=2.0, fy=2.0, cx=1.0, cy=1.0)
    pts = Frame(index=0, t=0.0, depth=depth, depth_intrinsics=intr).point_cloud()
    np.testing.assert_allclose(sorted(pts.tolist(), key=lambda p: p[2]),
                               [[0.5, -0.5, 1.0], [0.0, 0.0, 2.0]], atol=1e-6)


def test_point_cloud_requires_intrinsics(frame):
    frame.depth_intrinsics = None
    with pytest.raises(ValueError):
        frame.point_cloud()
