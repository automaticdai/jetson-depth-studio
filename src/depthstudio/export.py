"""Export frames to common 3D formats."""

import numpy as np

from .frames import Frame


def write_ply(path: str, frame: Frame) -> int:
    """Write the frame's point cloud as binary PLY (coloured if colour is aligned to depth).

    Returns the number of points written.
    """
    points = frame.point_cloud().astype(np.float32)
    colors = None
    if frame.color is not None and frame.color.shape[:2] == frame.depth.shape:
        v, u = np.nonzero(frame.depth)  # same order as point_cloud()
        colors = frame.color[v, u][:, ::-1]  # BGR -> RGB

    fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    if colors is not None:
        fields += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
    data = np.empty(len(points), dtype=fields)
    data["x"], data["y"], data["z"] = points[:, 0], points[:, 1], points[:, 2]
    if colors is not None:
        data["red"], data["green"], data["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]

    header = ["ply", "format binary_little_endian 1.0", "element vertex %d" % len(points),
              "property float x", "property float y", "property float z"]
    if colors is not None:
        header += ["property uchar red", "property uchar green", "property uchar blue"]
    header.append("end_header")
    with open(path, "wb") as fh:
        fh.write(("\n".join(header) + "\n").encode("ascii"))
        fh.write(data.tobytes())
    return len(points)
