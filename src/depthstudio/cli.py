"""depthstudio command line: list devices, take snapshots, record, inspect recordings, live preview."""

import argparse
import datetime
import json
import logging
import os
import signal
import sys
import time

import cv2

from . import __version__
from .sources import SourceError, open_source


def _add_source_args(p: argparse.ArgumentParser, default: str = "realsense") -> None:
    p.add_argument("-s", "--source", default=default,
                   help="realsense[:SERIAL], csi[:ID], usb[:N|/dev/videoN] or a recording dir (default: %(default)s)")
    p.add_argument("--width", type=int, help="image width (source default if omitted)")
    p.add_argument("--height", type=int, help="image height")
    p.add_argument("--fps", type=int, help="frame rate")
    p.add_argument("--no-imu", action="store_true", help="RealSense: do not stream the IMU")


def _open(args, realtime: bool = True):
    return open_source(args.source, width=args.width, height=args.height, fps=args.fps,
                       imu=not args.no_imu, realtime=realtime)


def _stamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


class _StopFlag:
    """Turns Ctrl-C / SIGTERM into a flag so loops can finish their files cleanly."""

    def __init__(self):
        self.stop = False
        signal.signal(signal.SIGINT, self._set)
        signal.signal(signal.SIGTERM, self._set)

    def _set(self, *_):
        self.stop = True


# ---------------------------------------------------------------- commands
def cmd_devices(args) -> int:
    try:
        from .sources.realsense import list_devices
        devs = list_devices()
    except SourceError as e:
        devs = []
        print("RealSense: %s" % e)
    for d in devs:
        print("realsense:%s  %s  fw %s  USB %s  IMU %s" % (d["serial"], d["name"], d["firmware"], d["usb"],
                                                          "yes" if d["imu"] else "no"))
    if not devs:
        print("no RealSense devices")
    v4l = "/sys/class/video4linux"
    nodes = [n for n in os.listdir(v4l) if n.startswith("video")] if os.path.isdir(v4l) else []
    for node in sorted(nodes, key=lambda n: int(n[len("video"):])):
        with open("/sys/class/video4linux/%s/name" % node) as fh:
            name = fh.read().strip()
        if "RealSense" in name:
            continue  # use the realsense source instead
        kind = "csi" if "vi-output" in name else "usb"
        hint = "csi:%s" % node[len("video"):] if kind == "csi" else "usb:/dev/%s" % node
        print("%-22s %s" % (hint, name))
    return 0


def cmd_snapshot(args) -> int:
    from .export import write_ply
    from .viz import colorize_depth

    out = args.output or os.path.join("snapshots", _stamp())
    os.makedirs(out, exist_ok=True)
    with _open(args) as src:
        frame = None
        for frame in src.frames(count=args.warmup + 1):  # let auto-exposure settle
            pass
        info = src.describe()
    written = []
    if frame.color is not None:
        cv2.imwrite(os.path.join(out, "color.png"), frame.color)
        written.append("color.png")
    if frame.depth is not None:
        cv2.imwrite(os.path.join(out, "depth.png"), frame.depth)
        cv2.imwrite(os.path.join(out, "depth_color.jpg"), colorize_depth(frame.depth, frame.depth_scale, args.max_depth))
        written += ["depth.png", "depth_color.jpg"]
        if frame.depth_intrinsics is not None and not args.no_ply:
            n = write_ply(os.path.join(out, "points.ply"), frame)
            written.append("points.ply (%d points)" % n)
    meta = {"source": info, "t": frame.t, "depth_scale": frame.depth_scale,
            "color_intrinsics": frame.color_intrinsics.to_dict() if frame.color_intrinsics else None,
            "depth_intrinsics": frame.depth_intrinsics.to_dict() if frame.depth_intrinsics else None,
            "imu": [s.__dict__ for s in frame.imu]}
    with open(os.path.join(out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print("saved to %s: %s, meta.json" % (out, ", ".join(written)))
    return 0


def cmd_record(args) -> int:
    from .record import Recorder

    out = args.output or os.path.join("recordings", _stamp())
    stop = _StopFlag()
    with _open(args, realtime=False) as src:
        with Recorder(out, source_info=src.describe(), lossless=args.lossless) as rec:
            print("recording to %s (Ctrl-C to stop)" % out)
            t0 = last = time.monotonic()
            last_count = 0
            for frame in src.frames(duration=args.duration, count=args.frames):
                rec.write(frame)
                now = time.monotonic()
                if now - last >= 1.0:
                    print("  %5.1f s  %d frames  %.1f fps  %d IMU samples"
                          % (now - t0, rec.count, (rec.count - last_count) / (now - last), rec.imu_count))
                    last, last_count = now, rec.count
                if stop.stop:
                    break
    print("saved %d frames, %d IMU samples to %s" % (rec.count, rec.imu_count, out))
    return 0


def cmd_info(args) -> int:
    from .record import RecordingSource

    src = RecordingSource(args.recording)
    src.start()
    meta = src.meta
    ts = src.timestamps
    print("recording : %s" % os.path.abspath(args.recording))
    print("created   : %s" % meta.get("created"))
    print("source    : %s" % json.dumps(meta.get("source")))
    print("frames    : %d" % len(ts))
    if len(ts) > 1:
        dur = ts[-1] - ts[0]
        print("duration  : %.2f s (%.1f fps)" % (dur, (len(ts) - 1) / dur if dur > 0 else 0))
    print("IMU       : %d samples" % meta.get("imu_samples", 0))
    for key in ("color_intrinsics", "depth_intrinsics"):
        i = meta.get(key)
        if i:
            print("%-10s: %dx%d fx=%.1f fy=%.1f cx=%.1f cy=%.1f"
                  % (key.split("_")[0], i["width"], i["height"], i["fx"], i["fy"], i["cx"], i["cy"]))
    print("depth unit: %s m" % meta.get("depth_scale"))
    return 0


def cmd_stream(args) -> int:
    from .stream import MjpegServer
    from .viz import compose

    stop = _StopFlag()
    server = MjpegServer(args.host, args.port)
    server.start()
    print("streaming %s on http://%s:%d/  (Ctrl-C to stop)" % (args.source, args.host, server.port))
    fps = None
    try:
        with _open(args) as src:
            last = time.monotonic()
            while not stop.stop:
                try:
                    frame = src.read(timeout=5.0)
                except StopIteration:  # recording finished
                    break
                if frame is None:
                    raise SourceError("%s stopped delivering frames" % args.source)
                now = time.monotonic()
                fps = 1.0 / (now - last) if fps is None else 0.9 * fps + 0.1 / max(now - last, 1e-6)
                last = now
                ok, jpeg = cv2.imencode(".jpg", compose(frame, args.max_depth, fps),
                                        [cv2.IMWRITE_JPEG_QUALITY, args.quality])
                if ok:
                    server.publish(jpeg.tobytes())
    finally:
        server.close()
    return 0


# ---------------------------------------------------------------- entry point
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="depthstudio", description=__doc__)
    p.add_argument("--version", action="version", version="%(prog)s " + __version__)
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("devices", help="list cameras").set_defaults(func=cmd_devices)

    s = sub.add_parser("snapshot", help="save one frame (colour, depth, point cloud)")
    _add_source_args(s)
    s.add_argument("-o", "--output", help="output dir (default snapshots/<timestamp>)")
    s.add_argument("--warmup", type=int, default=15, help="frames to skip for auto-exposure (default %(default)s)")
    s.add_argument("--max-depth", type=float, default=4.0, help="colour scale range in metres")
    s.add_argument("--no-ply", action="store_true", help="skip the point cloud")
    s.set_defaults(func=cmd_snapshot)

    r = sub.add_parser("record", help="record frames + IMU to a directory")
    _add_source_args(r)
    r.add_argument("-o", "--output", help="output dir (default recordings/<timestamp>)")
    r.add_argument("-d", "--duration", type=float, help="seconds to record (default: until Ctrl-C)")
    r.add_argument("-n", "--frames", type=int, help="number of frames to record")
    r.add_argument("--lossless", action="store_true", help="PNG colour instead of JPEG")
    r.set_defaults(func=cmd_record)

    i = sub.add_parser("info", help="summarise a recording")
    i.add_argument("recording")
    i.set_defaults(func=cmd_info)

    st = sub.add_parser("stream", help="live MJPEG preview over HTTP")
    _add_source_args(st)
    st.add_argument("--host", default="0.0.0.0")
    st.add_argument("--port", type=int, default=8080)
    st.add_argument("--quality", type=int, default=80, help="JPEG quality")
    st.add_argument("--max-depth", type=float, default=4.0, help="colour scale range in metres")
    st.set_defaults(func=cmd_stream)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        return args.func(args)
    except (SourceError, FileExistsError) as e:
        print("error: %s" % e, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
