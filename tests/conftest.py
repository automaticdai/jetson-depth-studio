import numpy as np
import pytest

from depthstudio import Frame, ImuSample, Intrinsics


def make_frame(index: int = 0, t: float = 100.0, w: int = 64, h: int = 48) -> Frame:
    rng = np.random.default_rng(index)
    depth = rng.integers(300, 4000, size=(h, w), dtype=np.uint16)
    depth[:4, :4] = 0  # some invalid pixels
    color = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    intr = Intrinsics(width=w, height=h, fx=50.0, fy=50.0, cx=w / 2, cy=h / 2)
    imu = [ImuSample(t - 0.004, "accel", 0.1, -0.2, 9.8), ImuSample(t - 0.002, "gyro", 0.01, 0.0, -0.01)]
    return Frame(index=index, t=t, color=color, depth=depth, depth_scale=0.001,
                 color_intrinsics=intr, depth_intrinsics=intr, imu=imu)


@pytest.fixture
def frame() -> Frame:
    return make_frame()
