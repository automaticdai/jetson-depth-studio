"""USB (UVC) webcam through OpenCV's V4L2 backend."""

import time
from typing import Optional, Union

import cv2

from ..frames import Frame
from .base import Source, SourceError


class UsbCameraSource(Source):
    name = "usb"

    def __init__(self, device: Union[int, str] = 0, width: int = 640, height: int = 480, fps: int = 30):
        self.device, self.width, self.height, self.fps = device, width, height, fps
        self._cap = None
        self._index = 0

    def start(self) -> None:
        self._cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if not self._cap.isOpened():
            raise SourceError("cannot open USB camera %r" % (self.device,))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        self._index = 0

    def read(self, timeout: float = 1.0) -> Optional[Frame]:
        if self._cap is None:
            raise SourceError("source not started")
        ok, img = self._cap.read()  # blocks for one frame period; V4L2 has its own timeout
        if not ok:
            return None
        frame = Frame(index=self._index, t=time.time(), color=img)
        self._index += 1
        return frame

    def stop(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def describe(self) -> dict:
        return {"source": self.name, "device": self.device, "width": self.width,
                "height": self.height, "fps": self.fps}
