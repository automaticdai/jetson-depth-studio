#!/usr/bin/env bash
# Stop the kernel HID sensor drivers from resetting the RealSense D435i IMU (run with sudo).
set -euo pipefail
[[ $EUID -eq 0 ]] || { echo "run as root: sudo $0" >&2; exit 1; }
cat > /etc/modprobe.d/blacklist-realsense-imu.conf <<'CONF'
# These reset the RealSense IMU configuration on re-bind
blacklist hid_sensor_hub
blacklist hid_sensor_accel_3d
blacklist hid_sensor_gyro_3d
blacklist hid_sensor_custom
CONF
modprobe -r hid_sensor_accel_3d hid_sensor_gyro_3d hid_sensor_custom hid_sensor_trigger hid_sensor_iio_common hid_sensor_hub
lsmod | grep hid_sensor && echo "some hid_sensor modules still loaded; reboot" || echo "hid_sensor modules unloaded"
