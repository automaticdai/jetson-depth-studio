# Depth + colour + IMU in one pipeline, the normal way to run a D435i.
import time, numpy as np, pyrealsense2 as rs
p = rs.pipeline(); c = rs.config()
c.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
c.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
c.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 200)
c.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 400)
q = rs.frame_queue(5000, keep_frames=True)
p.start(c, q)
counts = {}; acc = []
t = time.time()
while time.time() - t < 5:
    ok, f = q.try_wait_for_frame(1000)
    if not ok:
        continue
    for x in (f.as_frameset() if f.is_frameset() else [f]):
        s = x.get_profile().stream_name(); counts[s] = counts.get(s, 0) + 1
        if x.get_profile().stream_type() == rs.stream.accel:
            m = x.as_motion_frame().get_motion_data(); acc.append((m.x, m.y, m.z))
p.stop()
print("frames in 5 s:", counts)
if acc:
    a = np.mean(acc, 0); print("mean accel", np.round(a, 2), "|g| = %.2f m/s^2" % np.linalg.norm(a))
