"""MIPI CSI camera (e.g. the IMX219 on a Jetson Nano) through nvarguscamerasrc.

The pip OpenCV build has no GStreamer support, so frames come from a gst-launch-1.0 child
process writing raw BGR to stdout. A reader thread keeps only the newest frame, so read()
never returns stale images when the consumer is slower than the camera.
"""

import select
import subprocess
import threading
import time
from typing import List, Optional

import numpy as np

from ..frames import Frame
from .base import Source, SourceError


class CsiCameraSource(Source):
    name = "csi"

    def __init__(self, sensor_id: int = 0, width: int = 1280, height: int = 720, fps: int = 30,
                 flip_method: int = 0):
        self.sensor_id, self.width, self.height, self.fps = sensor_id, width, height, fps
        self.flip_method = flip_method
        self._proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._cond = threading.Condition()
        self._latest: Optional[np.ndarray] = None
        self._latest_t = 0.0
        self._seq = 0
        self._read_seq = 0
        self._index = 0
        self._running = False

    def pipeline_args(self) -> List[str]:
        out_w, out_h = (self.height, self.width) if self.flip_method in (1, 3, 4, 6) else (self.width, self.height)
        return ["gst-launch-1.0", "-q",
                "nvarguscamerasrc", "sensor-id=%d" % self.sensor_id, "!",
                "video/x-raw(memory:NVMM),width=%d,height=%d,framerate=%d/1" % (self.width, self.height, self.fps), "!",
                "nvvidconv", "flip-method=%d" % self.flip_method, "!",
                "video/x-raw,format=BGRx,width=%d,height=%d" % (out_w, out_h), "!",
                "videoconvert", "!", "video/x-raw,format=BGR", "!",
                "fdsink", "fd=1", "sync=false"]

    @property
    def _shape(self):
        if self.flip_method in (1, 3, 4, 6):
            return (self.width, self.height, 3)
        return (self.height, self.width, 3)

    def start(self) -> None:
        self._proc = subprocess.Popen(self.pipeline_args(), stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE, bufsize=0)
        self._running = True
        self._thread = threading.Thread(target=self._reader, name="csi-reader", daemon=True)
        self._thread.start()
        if self.read(timeout=10.0) is None:
            err = self._stderr_tail()
            self.stop()
            raise SourceError("CSI camera %d produced no frames. %s" % (self.sensor_id, err))
        self._index = 0

    def _reader(self) -> None:
        size = int(np.prod(self._shape))
        out = self._proc.stdout  # unbuffered FileIO (bufsize=0)
        while self._running:
            buf = bytearray(size)  # fresh buffer per frame: the previous one is still being consumed
            view = memoryview(buf)
            got = 0
            while got < size:
                ready, _, _ = select.select([out], [], [], 0.5)
                if not self._running:
                    return
                if not ready:
                    continue
                n = out.readinto(view[got:])
                if not n:  # EOF: gst-launch exited
                    self._running = False
                    with self._cond:
                        self._cond.notify_all()
                    return
                got += n
            frame = np.frombuffer(buf, np.uint8).reshape(self._shape)
            with self._cond:
                self._latest, self._latest_t = frame, time.time()
                self._seq += 1
                self._cond.notify_all()

    def read(self, timeout: float = 1.0) -> Optional[Frame]:
        with self._cond:
            if not self._cond.wait_for(lambda: self._seq > self._read_seq or not self._running, timeout):
                return None
            if self._seq == self._read_seq:  # stopped without a new frame
                return None
            self._read_seq = self._seq
            frame = Frame(index=self._index, t=self._latest_t, color=self._latest)
        self._index += 1
        return frame

    def _stderr_tail(self) -> str:
        if self._proc is None or self._proc.poll() is None:
            return "Is nvargus-daemon running and the camera connected?"
        return self._proc.stderr.read().decode(errors="replace").strip()[-500:]

    def stop(self) -> None:
        self._running = False
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
            for stream in (self._proc.stdout, self._proc.stderr):
                stream.close()
            self._proc = None
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

    def describe(self) -> dict:
        return {"source": self.name, "sensor_id": self.sensor_id, "width": self.width,
                "height": self.height, "fps": self.fps, "flip_method": self.flip_method}
