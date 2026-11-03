"""Intel RealSense (D435i) source: depth + colour + IMU through a single pipeline.

On the Jetson Nano with the RSUSB backend, starting the IMU and the video streams in separate
sessions can leave the second one silent until the camera is reset. Running everything in one
pipeline avoids that, and start() resets the camera and retries once if any requested stream
fails to deliver. See scripts/fix_imu.sh for the kernel side of the IMU fix.
"""

import logging
import time
from typing import List, Optional

from ..frames import Frame, ImuSample, Intrinsics
from .base import Source, SourceError

log = logging.getLogger(__name__)


def _rs():
    try:
        import pyrealsense2 as rs
    except ImportError as e:
        raise SourceError("pyrealsense2 is not available for this Python "
                          "(on the Jetson use python3.8; see scripts/install_librealsense.sh)") from e
    return rs


def list_devices() -> List[dict]:
    rs = _rs()
    out = []
    for dev in rs.context().query_devices():
        def info(key):
            return dev.get_info(key) if dev.supports(key) else None
        out.append({
            "name": info(rs.camera_info.name),
            "serial": info(rs.camera_info.serial_number),
            "firmware": info(rs.camera_info.firmware_version),
            "usb": info(rs.camera_info.usb_type_descriptor),
            "imu": any(s.is_motion_sensor() for s in dev.query_sensors()),
        })
    return out


def _intrinsics(profile) -> Intrinsics:
    i = profile.as_video_stream_profile().get_intrinsics()
    return Intrinsics(width=i.width, height=i.height, fx=i.fx, fy=i.fy, cx=i.ppx, cy=i.ppy,
                      model=str(i.model).split(".")[-1], coeffs=list(i.coeffs))


class RealSenseSource(Source):
    name = "realsense"

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30,
                 color: bool = True, depth: bool = True, imu: bool = True,
                 align_to_color: bool = True, serial: Optional[str] = None,
                 accel_hz: int = 200, gyro_hz: int = 400,
                 start_timeout: float = 5.0, reset_on_failure: bool = True):
        if not (color or depth or imu):
            raise ValueError("enable at least one of color, depth, imu")
        self.width, self.height, self.fps = width, height, fps
        self.color, self.depth, self.imu = color, depth, imu
        self.align_to_color = align_to_color and color and depth
        self.serial = serial
        self.accel_hz, self.gyro_hz = accel_hz, gyro_hz
        self.start_timeout = start_timeout
        self.reset_on_failure = reset_on_failure
        self._pipe = None
        self._queue = None
        self._align = None
        self._pending_imu: List[ImuSample] = []
        self._first: Optional[Frame] = None
        self._index = 0
        self._last_numbers = (None, None)  # (depth, color) frame numbers of the last emitted Frame
        self._depth_scale = 0.001
        self._color_intr = self._depth_intr = None
        self.device_info: dict = {}

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        try:
            self._start_and_check()
        except SourceError as e:
            if not self.reset_on_failure:
                raise
            log.warning("%s; resetting the camera and retrying", e)
            self.stop()
            self._hardware_reset()
            self._start_and_check()

    def stop(self) -> None:
        if self._pipe is not None:
            try:
                self._pipe.stop()
            except RuntimeError:
                pass
        self._pipe = self._queue = self._first = None
        self._pending_imu = []
        self._last_numbers = (None, None)

    def _start_and_check(self) -> None:
        rs = _rs()
        cfg = rs.config()
        if self.serial:
            cfg.enable_device(self.serial)
        if self.depth:
            cfg.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        if self.color:
            cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        if self.imu:
            cfg.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, self.accel_hz)
            cfg.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, self.gyro_hz)

        # A frame_queue (not a Python callback) keeps IMU samples and video framesets separate
        # without the GIL deadlock that sensor callbacks hit in this pybind11 build.
        self._queue = rs.frame_queue(2000, keep_frames=True)
        self._pipe = rs.pipeline()
        profile = self._pipe.start(cfg, self._queue)

        dev = profile.get_device()
        self.device_info = {
            "name": dev.get_info(rs.camera_info.name),
            "serial": dev.get_info(rs.camera_info.serial_number),
            "firmware": dev.get_info(rs.camera_info.firmware_version),
        }
        if self.depth:
            self._depth_scale = dev.first_depth_sensor().get_depth_scale()
            self._depth_intr = _intrinsics(profile.get_stream(rs.stream.depth))
        if self.color:
            self._color_intr = _intrinsics(profile.get_stream(rs.stream.color))
        if self.align_to_color:
            self._align = rs.align(rs.stream.color)
            self._depth_intr = self._color_intr

        # Every requested stream must actually deliver, otherwise report failure (start() resets).
        want = {k for k, on in (("depth", self.depth), ("color", self.color),
                                ("accel", self.imu), ("gyro", self.imu)) if on}
        seen = set()
        first = None
        deadline = time.monotonic() + self.start_timeout
        while want - seen and time.monotonic() < deadline:
            frame = self.read(timeout=0.5)
            if frame is None:
                continue
            if frame.depth is not None:
                seen.add("depth")
            if frame.color is not None:
                seen.add("color")
            seen.update(s.kind for s in frame.imu)
            if first is None and (frame.depth is not None or frame.color is not None):
                first = frame
        if want - seen:
            raise SourceError("RealSense streams did not start: missing %s" % sorted(want - seen))
        self._index = 0
        self._first = first

    def _hardware_reset(self, wait: float = 15.0) -> None:
        rs = _rs()
        devs = [d for d in rs.context().query_devices()
                if not self.serial or d.get_info(rs.camera_info.serial_number) == self.serial]
        if not devs:
            raise SourceError("no RealSense device found")
        serial = devs[0].get_info(rs.camera_info.serial_number)
        devs[0].hardware_reset()
        time.sleep(2.0)  # the device drops off the bus before it re-enumerates
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline:
            if any(d.get_info(rs.camera_info.serial_number) == serial for d in rs.context().query_devices()):
                time.sleep(1.0)
                return
            time.sleep(0.5)
        raise SourceError("RealSense %s did not come back after hardware reset" % serial)

    # ------------------------------------------------------------------ reading
    def read(self, timeout: float = 1.0) -> Optional[Frame]:
        if self._queue is None:
            raise SourceError("source not started")
        if self._first is not None:  # frame consumed by the start-up check
            frame, self._first = self._first, None
            frame.index = self._index
            self._index += 1
            frame.imu = frame.imu + self._take_imu()
            return frame

        rs = _rs()
        video = self.depth or self.color
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            ok, f = self._queue.try_wait_for_frame(max(1, int(remaining * 1000)))
            if not ok:
                break
            if f.is_frameset():
                frame = self._make_frame(f.as_frameset())
                if frame is not None:
                    return frame
                continue
            if f.is_motion_frame():
                self._add_imu(f.as_motion_frame(), rs)
                if not video and self._queue_empty():
                    break
        if not video and self._pending_imu:
            return self._make_imu_frame()
        return None

    def _queue_empty(self) -> bool:
        f = self._queue.poll_for_frame()
        if f:  # IMU-only mode, so anything queued is a motion frame
            self._add_imu(f.as_motion_frame(), _rs())
            return False
        return True

    def _add_imu(self, mf, rs) -> None:
        d = mf.get_motion_data()
        kind = "accel" if mf.get_profile().stream_type() == rs.stream.accel else "gyro"
        self._pending_imu.append(ImuSample(mf.get_timestamp() / 1000.0, kind, d.x, d.y, d.z))

    def _take_imu(self) -> List[ImuSample]:
        imu, self._pending_imu = self._pending_imu, []
        return imu

    def _make_imu_frame(self) -> Frame:
        imu = self._take_imu()
        frame = Frame(index=self._index, t=imu[-1].t, imu=imu)
        self._index += 1
        return frame

    def _make_frame(self, fs) -> Optional[Frame]:
        """Build a Frame, or None if the set repeats an image that was already emitted.

        The syncer emits partial updates (new depth + previous colour, then the same depth +
        new colour) while streams settle or the consumer lags; only pairs that are both new pass.
        """
        import numpy as np

        df, cf = fs.get_depth_frame(), fs.get_color_frame()
        numbers = (df.get_frame_number() if df else None, cf.get_frame_number() if cf else None)
        if (self.depth and (not df or numbers[0] == self._last_numbers[0])) or \
                (self.color and (not cf or numbers[1] == self._last_numbers[1])):
            return None
        self._last_numbers = numbers

        if self._align is not None:
            fs = self._align.process(fs)
        depth = color = None
        if self.depth:
            df = fs.get_depth_frame()
            if df:
                depth = np.asanyarray(df.get_data()).copy()
        if self.color:
            cf = fs.get_color_frame()
            if cf:
                color = np.asanyarray(cf.get_data()).copy()
        frame = Frame(index=self._index, t=fs.get_timestamp() / 1000.0, color=color, depth=depth,
                      depth_scale=self._depth_scale, color_intrinsics=self._color_intr,
                      depth_intrinsics=self._depth_intr if depth is not None else None,
                      imu=self._take_imu())
        self._index += 1
        return frame

    def describe(self) -> dict:
        return {"source": self.name, "device": self.device_info, "width": self.width,
                "height": self.height, "fps": self.fps, "aligned_to_color": self.align_to_color,
                "accel_hz": self.accel_hz if self.imu else None,
                "gyro_hz": self.gyro_hz if self.imu else None}
