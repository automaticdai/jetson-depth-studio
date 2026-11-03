"""Minimal MJPEG-over-HTTP preview, so a headless Jetson can be watched from a browser."""

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

PAGE = b"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>depthstudio</title>
<style>body{margin:0;background:#111;color:#ddd;font:14px sans-serif}
img{display:block;max-width:100%;margin:0 auto}p{text-align:center}</style></head>
<body><img src="/stream" alt="live stream"><p>depthstudio live preview &middot;
<a style="color:#8cf" href="/snapshot.jpg">snapshot</a></p></body></html>"""


class MjpegServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 8080):
        self._cond = threading.Condition()
        self._jpeg: Optional[bytes] = None
        self._seq = 0
        self._closed = False
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                if self.path == "/":
                    self._send(200, "text/html; charset=utf-8", PAGE)
                elif self.path == "/snapshot.jpg":
                    jpeg = server.wait_frame(0, timeout=5.0)[1]
                    if jpeg is None:
                        self._send(503, "text/plain", b"no frame yet")
                    else:
                        self._send(200, "image/jpeg", jpeg)
                elif self.path == "/stream":
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    seq = 0
                    try:
                        while True:
                            seq, jpeg = server.wait_frame(seq, timeout=10.0)
                            if jpeg is None:
                                return
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n"
                                             % len(jpeg))
                            self.wfile.write(jpeg + b"\r\n")
                    except (BrokenPipeError, ConnectionResetError):
                        return
                else:
                    self._send(404, "text/plain", b"not found")

            def _send(self, code, ctype, body):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self._httpd = ThreadingHTTPServer((host, port), Handler)
        self._httpd.daemon_threads = True
        self.port = self._httpd.server_address[1]
        self._thread = threading.Thread(target=self._httpd.serve_forever, name="mjpeg", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def publish(self, jpeg: bytes) -> None:
        with self._cond:
            self._jpeg = jpeg
            self._seq += 1
            self._cond.notify_all()

    def wait_frame(self, after_seq: int, timeout: float):
        """Block until a frame newer than `after_seq` exists; returns (seq, jpeg) or (seq, None)."""
        with self._cond:
            self._cond.wait_for(lambda: self._seq > after_seq or self._closed, timeout)
            if self._closed or self._seq <= after_seq:
                return after_seq, None
            return self._seq, self._jpeg

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()
        self._httpd.shutdown()
        self._httpd.server_close()
