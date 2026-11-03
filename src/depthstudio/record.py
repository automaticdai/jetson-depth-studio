"""Record frames to a plain directory and play them back as a Source.

Layout (everything readable without this package):

    meta.json     source description, depth scale, intrinsics, frame count
    frames.csv    index,t,color,depth   (file names relative to the recording)
    imu.csv       frame_index,t,kind,x,y,z
    color/000000.jpg ...   BGR images (JPEG, or PNG with lossless=True)
    depth/000000.png ...   uint16 PNG, multiply by depth_scale for metres
"""

import csv
import datetime
import json
import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import cv2
import numpy as np

from .frames import Frame, ImuSample, Intrinsics
from .sources.base import Source, SourceError

FORMAT = "depthstudio-recording"
VERSION = 1


class Recorder:
    def __init__(self, path: str, source_info: Optional[dict] = None, lossless: bool = False,
                 jpeg_quality: int = 90, workers: int = 3, max_pending: int = 64):
        if os.path.exists(path) and os.listdir(path):
            raise FileExistsError("recording directory is not empty: %s" % path)
        self.path = path
        self.source_info = source_info or {}
        self.lossless = lossless
        self.jpeg_quality = jpeg_quality
        self.count = 0
        self.imu_count = 0
        self._meta_frame: Optional[Frame] = None
        os.makedirs(os.path.join(path, "color"), exist_ok=True)
        os.makedirs(os.path.join(path, "depth"), exist_ok=True)
        self._frames_f = open(os.path.join(path, "frames.csv"), "w", newline="")
        self._imu_f = open(os.path.join(path, "imu.csv"), "w", newline="")
        self._frames = csv.writer(self._frames_f)
        self._imu = csv.writer(self._imu_f)
        self._frames.writerow(["index", "t", "color", "depth"])
        self._imu.writerow(["frame_index", "t", "kind", "x", "y", "z"])
        self._started = datetime.datetime.now().isoformat(timespec="seconds")
        # Image encoding dominates (cv2 releases the GIL), so it runs on a small pool; the bounded
        # backlog applies back-pressure instead of buffering unbounded frames in memory.
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="recorder")
        self._pending = deque()
        self._max_pending = max_pending

    def _imwrite(self, name: str, img: np.ndarray, params: list) -> None:
        if not cv2.imwrite(os.path.join(self.path, name), img, params):
            raise IOError("failed to write " + name)

    def _submit(self, *args) -> None:
        while len(self._pending) >= self._max_pending:
            self._pending.popleft().result()
        while self._pending and self._pending[0].done():
            self._pending.popleft().result()  # surface write errors early
        self._pending.append(self._pool.submit(self._imwrite, *args))

    def write(self, frame: Frame) -> None:
        i = self.count
        color_name = depth_name = ""
        if frame.color is not None:
            if self.lossless:
                color_name = "color/%06d.png" % i
                params = [cv2.IMWRITE_PNG_COMPRESSION, 1]
            else:
                color_name = "color/%06d.jpg" % i
                params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
            self._submit(color_name, frame.color, params)
        if frame.depth is not None:
            depth_name = "depth/%06d.png" % i
            self._submit(depth_name, frame.depth.astype(np.uint16), [cv2.IMWRITE_PNG_COMPRESSION, 1])
        self._frames.writerow([i, repr(frame.t), color_name, depth_name])
        for s in frame.imu:
            self._imu.writerow([i, repr(s.t), s.kind, repr(s.x), repr(s.y), repr(s.z)])
        self.imu_count += len(frame.imu)
        if self._meta_frame is None and (frame.color is not None or frame.depth is not None):
            self._meta_frame = frame
            self._write_meta()
        self.count += 1

    def _write_meta(self) -> None:
        f = self._meta_frame
        meta = {
            "format": FORMAT, "version": VERSION, "created": self._started,
            "source": self.source_info, "frames": self.count, "imu_samples": self.imu_count,
            "depth_scale": f.depth_scale if f else None,
            "color_intrinsics": f.color_intrinsics.to_dict() if f and f.color_intrinsics else None,
            "depth_intrinsics": f.depth_intrinsics.to_dict() if f and f.depth_intrinsics else None,
        }
        tmp = os.path.join(self.path, "meta.json.tmp")
        with open(tmp, "w") as fh:
            json.dump(meta, fh, indent=2)
        os.replace(tmp, os.path.join(self.path, "meta.json"))

    def close(self) -> None:
        if self._frames_f.closed:
            return
        try:
            while self._pending:
                self._pending.popleft().result()
        finally:
            self._pool.shutdown(wait=True)
        self._frames_f.close()
        self._imu_f.close()
        self._write_meta()

    def __enter__(self) -> "Recorder":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def load_meta(path: str) -> dict:
    with open(os.path.join(path, "meta.json")) as fh:
        meta = json.load(fh)
    if meta.get("format") != FORMAT:
        raise SourceError("%s is not a depthstudio recording" % path)
    return meta


class RecordingSource(Source):
    """Plays a recording back. With realtime=True frames are paced by their timestamps."""

    name = "recording"

    def __init__(self, path: str, realtime: bool = False, loop: bool = False):
        self.path, self.realtime, self.loop = path, realtime, loop
        self.meta: dict = {}
        self._rows: List[dict] = []
        self._imu: dict = {}
        self._pos = 0
        self._t0_wall = self._t0_rec = None
        self._index = 0

    def start(self) -> None:
        self.meta = load_meta(self.path)
        with open(os.path.join(self.path, "frames.csv"), newline="") as fh:
            self._rows = list(csv.DictReader(fh))
        self._imu = {}
        imu_path = os.path.join(self.path, "imu.csv")
        if os.path.exists(imu_path):
            with open(imu_path, newline="") as fh:
                for r in csv.DictReader(fh):
                    self._imu.setdefault(int(r["frame_index"]), []).append(
                        ImuSample(float(r["t"]), r["kind"], float(r["x"]), float(r["y"]), float(r["z"])))
        self._ci = Intrinsics.from_dict(self.meta["color_intrinsics"]) if self.meta.get("color_intrinsics") else None
        self._di = Intrinsics.from_dict(self.meta["depth_intrinsics"]) if self.meta.get("depth_intrinsics") else None
        self._pos = 0
        self._index = 0
        self._t0_wall = self._t0_rec = None

    def stop(self) -> None:
        self._rows = []

    def __len__(self) -> int:
        return len(self._rows)

    @property
    def timestamps(self) -> List[float]:
        return [float(r["t"]) for r in self._rows]

    def read(self, timeout: float = 1.0) -> Optional[Frame]:
        if self._pos >= len(self._rows):
            if not (self.loop and self._rows):
                raise StopIteration
            self._pos = 0
            self._t0_wall = self._t0_rec = None
        row = self._rows[self._pos]
        rec_index = int(row["index"])
        self._pos += 1
        t = float(row["t"])
        if self.realtime:
            if self._t0_wall is None:
                self._t0_wall, self._t0_rec = time.monotonic(), t
            delay = (t - self._t0_rec) - (time.monotonic() - self._t0_wall)
            if delay > 0:
                time.sleep(delay)
        color = depth = None
        if row["color"]:
            color = cv2.imread(os.path.join(self.path, row["color"]), cv2.IMREAD_COLOR)
        if row["depth"]:
            depth = cv2.imread(os.path.join(self.path, row["depth"]), cv2.IMREAD_UNCHANGED)
        frame = Frame(index=self._index, t=t, color=color, depth=depth,
                      depth_scale=self.meta.get("depth_scale") or 0.001,
                      color_intrinsics=self._ci if color is not None else None,
                      depth_intrinsics=self._di if depth is not None else None,
                      imu=list(self._imu.get(rec_index, [])))
        self._index += 1
        return frame

    def describe(self) -> dict:
        return {"source": self.name, "path": os.path.abspath(self.path), "recorded": self.meta.get("source")}
