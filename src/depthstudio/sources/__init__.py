"""Frame sources and a factory that builds one from a short spec string.

Specs:
    realsense[:SERIAL]       Intel RealSense depth camera (depth + colour + IMU)
    csi[:SENSOR_ID]          MIPI CSI camera via nvarguscamerasrc (default sensor 0)
    usb[:DEVICE]             USB/UVC webcam, DEVICE is an index or /dev/videoN (default 0)
    PATH                     a directory made by `depthstudio record`
"""

import os

from .base import Source, SourceError


def open_source(spec: str, width: int = None, height: int = None, fps: int = None,
                imu: bool = True, realtime: bool = True) -> Source:
    kind, _, arg = spec.partition(":")
    size = {k: v for k, v in (("width", width), ("height", height), ("fps", fps)) if v}
    if kind == "realsense":
        from .realsense import RealSenseSource
        return RealSenseSource(serial=arg or None, imu=imu, **size)
    if kind == "csi":
        from .csi import CsiCameraSource
        return CsiCameraSource(sensor_id=int(arg or 0), **size)
    if kind == "usb":
        from .usb import UsbCameraSource
        return UsbCameraSource(device=int(arg) if arg.isdigit() else (arg or 0), **size)
    if os.path.isdir(spec):
        from ..record import RecordingSource
        return RecordingSource(spec, realtime=realtime)
    raise SourceError("unknown source %r (use realsense, csi[:N], usb[:N|/dev/videoN] or a recording dir)" % spec)


__all__ = ["Source", "SourceError", "open_source"]
