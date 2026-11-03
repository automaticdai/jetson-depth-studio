"""Camera-independent frame types shared by every source, the recorder and the viewers."""

from dataclasses import asdict, dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class Intrinsics:
    """Pinhole intrinsics of one image stream (pixels), plus optional distortion coefficients."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    model: str = "none"
    coeffs: List[float] = field(default_factory=list)

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Intrinsics":
        return cls(**d)


@dataclass
class ImuSample:
    """One accelerometer (m/s^2) or gyroscope (rad/s) reading."""

    t: float  # seconds, same clock as Frame.t
    kind: str  # "accel" or "gyro"
    x: float
    y: float
    z: float


@dataclass
class Frame:
    """One capture from a source.

    `t` is in seconds on the source's clock; RealSense uses its global (host-synchronised)
    time domain and the other sources use time.time(), so all of them are comparable.
    """

    index: int
    t: float
    color: Optional[np.ndarray] = None  # H x W x 3, BGR, uint8
    depth: Optional[np.ndarray] = None  # H x W, uint16, multiply by depth_scale for metres
    depth_scale: float = 0.001
    color_intrinsics: Optional[Intrinsics] = None
    depth_intrinsics: Optional[Intrinsics] = None
    imu: List[ImuSample] = field(default_factory=list)  # samples received since the previous frame

    def depth_m(self) -> Optional[np.ndarray]:
        """Depth in metres as float32 (0 where invalid), or None if the frame has no depth."""
        if self.depth is None:
            return None
        return self.depth.astype(np.float32) * np.float32(self.depth_scale)

    def point_cloud(self) -> np.ndarray:
        """Back-project valid depth pixels to an N x 3 array of XYZ points in metres."""
        if self.depth is None or self.depth_intrinsics is None:
            raise ValueError("point_cloud() needs depth and depth_intrinsics")
        intr = self.depth_intrinsics
        z = self.depth_m()
        v, u = np.nonzero(z)
        z = z[v, u]
        x = (u - intr.cx) * z / intr.fx
        y = (v - intr.cy) * z / intr.fy
        return np.stack([x, y, z], axis=1)
