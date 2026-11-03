"""Turn frames into images for previews and snapshots."""

from typing import Optional

import cv2
import numpy as np

from .frames import Frame


def colorize_depth(depth: np.ndarray, depth_scale: float = 0.001, max_m: float = 4.0,
                   min_m: float = 0.0) -> np.ndarray:
    """Map uint16 depth to a BGR turbo image (near = red, far = blue); invalid pixels are black."""
    z = depth.astype(np.float32) * np.float32(depth_scale)
    norm = np.clip((z - min_m) / max(max_m - min_m, 1e-6), 0.0, 1.0)
    img = cv2.applyColorMap((255 * (1.0 - norm)).astype(np.uint8), cv2.COLORMAP_TURBO)
    img[depth == 0] = 0
    return img


def compose(frame: Frame, max_m: float = 4.0, fps: Optional[float] = None) -> np.ndarray:
    """Colour and colourised depth side by side, with a small status line."""
    panels = []
    if frame.color is not None:
        panels.append(frame.color)
    if frame.depth is not None:
        panels.append(colorize_depth(frame.depth, frame.depth_scale, max_m))
    if not panels:
        panels.append(np.zeros((240, 320, 3), np.uint8))
    h = min(p.shape[0] for p in panels)
    panels = [p if p.shape[0] == h else cv2.resize(p, (int(p.shape[1] * h / p.shape[0]), h)) for p in panels]
    img = np.hstack(panels)

    parts = ["#%d" % frame.index]
    if fps:
        parts.append("%.1f fps" % fps)
    if frame.depth is not None:
        d = frame.depth[frame.depth.shape[0] // 2, frame.depth.shape[1] // 2] * frame.depth_scale
        parts.append("centre %.2f m" % d if d else "centre --")
    accel = [s for s in frame.imu if s.kind == "accel"]
    if accel:
        a = accel[-1]
        parts.append("accel %.2f %.2f %.2f" % (a.x, a.y, a.z))
    text = "   ".join(parts)
    bar = img[:30]
    bar[:] = (bar * 0.35).astype(np.uint8)  # darken the strip so the text stays readable on any scene
    cv2.putText(img, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img
