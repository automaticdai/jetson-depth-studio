#!/usr/bin/env bash
# Build and install librealsense2 + pyrealsense2 on a Jetson Nano (JetPack 4.6 / L4T R32.7, Ubuntu 18.04).
#
# Uses the RSUSB (libusb) backend, so no kernel patching is needed and the IMU
# is read over USB directly (no /sys/bus/iio permissions required).
# Re-running it upgrades: any previous build in SRC_DIR is uninstalled first.
#
# Usage:  sudo ./install_librealsense.sh
# Options (environment variables):
#   LRS_VERSION=v2.57.7   librealsense tag (v2.57.x targets D400 firmware 5.17)
#   PYTHON=python3.8      Python for pyrealsense2; v2.56.4+ needs >= 3.7 (JetPack's python3 is 3.6)
#   WITH_PY_EXTRAS=1      pip-install numpy + opencv-python-headless for $PYTHON (binary wheels only)
#   WITH_VIEWER=1         build realsense-viewer and graphical examples (0 = faster build)
#   WITH_CUDA=1           CUDA-accelerated pointcloud / colour conversion
#   JOBS=4                parallel make jobs (drop to 2 if the build gets OOM-killed)
#   SRC_DIR=/usr/local/src/librealsense
set -euo pipefail

LRS_VERSION="${LRS_VERSION:-v2.57.7}"
PYTHON="${PYTHON:-python3.8}"
WITH_PY_EXTRAS="${WITH_PY_EXTRAS:-1}"
WITH_VIEWER="${WITH_VIEWER:-1}"
WITH_CUDA="${WITH_CUDA:-1}"
JOBS="${JOBS:-$(nproc)}"
SRC_DIR="${SRC_DIR:-/usr/local/src/librealsense}"
SWAPFILE=/var/swapfile-librealsense

log() { printf '\n\033[1;32m==> %s\033[0m\n' "$*"; }
die() { printf '\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

[[ $EUID -eq 0 ]] || die "run as root: sudo $0"
[[ -f /etc/nv_tegra_release ]] || die "this script is for NVIDIA Jetson (L4T)"
[[ "$(uname -m)" == aarch64 ]] || die "expected aarch64"

# ---------------------------------------------------------------- dependencies
log "Installing build dependencies"
apt-get update
apt-get install -y --no-install-recommends \
    git cmake build-essential pkg-config curl ca-certificates udev \
    libssl-dev libusb-1.0-0-dev libudev-dev \
    "$PYTHON" "$PYTHON-dev"
if [[ $WITH_VIEWER == 1 ]]; then
    apt-get install -y --no-install-recommends \
        libgtk-3-dev libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev
fi

PY_BIN=$(command -v "$PYTHON")
PY_VER=$("$PY_BIN" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
PY_DIR=/usr/local/lib/python$PY_VER/dist-packages/pyrealsense2  # on sys.path for Ubuntu's pythons
PY_LIB=/usr/lib/aarch64-linux-gnu/libpython${PY_VER}m.so
[[ -e $PY_LIB ]] || PY_LIB=/usr/lib/aarch64-linux-gnu/libpython$PY_VER.so
PY_INC=$("$PY_BIN" -c 'import sysconfig; print(sysconfig.get_paths()["include"])')

# ---------------------------------------------------------------- swap
# The Nano has 4 GB RAM + ~2 GB zram; compiling with -j4 needs more.
cleanup_swap() {
    if [[ -f $SWAPFILE ]]; then
        log "Removing temporary swap file"
        swapoff "$SWAPFILE" 2>/dev/null || true
        rm -f "$SWAPFILE"
    fi
}
swap_kb=$(awk '/SwapTotal/ {print $2}' /proc/meminfo)
if (( swap_kb < 6 * 1024 * 1024 )) && [[ ! -f $SWAPFILE ]]; then
    log "Adding temporary 4 GB swap file at $SWAPFILE"
    fallocate -l 4G "$SWAPFILE"
    chmod 600 "$SWAPFILE"
    mkswap "$SWAPFILE" >/dev/null
    swapon "$SWAPFILE"
    trap cleanup_swap EXIT
fi

# ---------------------------------------------------------------- remove previous install
if [[ -f $SRC_DIR/build/install_manifest.txt ]]; then
    log "Uninstalling previous build in $SRC_DIR/build"
    make -C "$SRC_DIR/build" uninstall || xargs -d '\n' rm -f < "$SRC_DIR/build/install_manifest.txt"
    # files copied outside the manifest by the v2.55.1 workaround
    rm -rf /usr/local/lib/python3.*/dist-packages/pyrealsense2
    rm -rf "$SRC_DIR/build"
    ldconfig
fi

# ---------------------------------------------------------------- source
log "Fetching librealsense $LRS_VERSION into $SRC_DIR"
if [[ -d $SRC_DIR/.git ]]; then
    git -C "$SRC_DIR" fetch --depth 1 origin tag "$LRS_VERSION"
    git -C "$SRC_DIR" checkout -f "$LRS_VERSION"
else
    mkdir -p "$(dirname "$SRC_DIR")"
    git clone --depth 1 -b "$LRS_VERSION" https://github.com/IntelRealSense/librealsense.git "$SRC_DIR"
fi

# ---------------------------------------------------------------- configure
cuda_args=(-DBUILD_WITH_CUDA=OFF)
if [[ $WITH_CUDA == 1 && -x /usr/local/cuda/bin/nvcc ]]; then
    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
    cuda_args=(-DBUILD_WITH_CUDA=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc)
fi

log "Configuring (CUDA: ${cuda_args[0]#*=}, viewer: $WITH_VIEWER, python: $PY_BIN)"
mkdir -p "$SRC_DIR/build"
cd "$SRC_DIR/build"
# CMake 3.10 uses FindPythonInterp/FindPythonLibs, which would otherwise pick up python3.6's headers
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DFORCE_RSUSB_BACKEND=ON \
    -DBUILD_PYTHON_BINDINGS=ON \
    -DPYTHON_EXECUTABLE="$PY_BIN" \
    -DPYTHON_INCLUDE_DIR="$PY_INC" \
    -DPYTHON_LIBRARY="$PY_LIB" \
    -DPYTHON_INSTALL_DIR="$PY_DIR" \
    -DBUILD_EXAMPLES=ON \
    -DBUILD_GRAPHICAL_EXAMPLES="$( [[ $WITH_VIEWER == 1 ]] && echo ON || echo OFF )" \
    -DBUILD_TOOLS=ON \
    -DBUILD_UNIT_TESTS=OFF \
    -DCHECK_FOR_UPDATES=OFF \
    -DBUILD_WITH_DDS=OFF \
    "${cuda_args[@]}"

# ---------------------------------------------------------------- build
log "Building with -j$JOBS (expect 1-2 hours on a Nano)"
make -j"$JOBS"

log "Installing"
make install
# v2.55.1 only installs the module inside its BUILD_LEGACY_PYBACKEND block; later tags install it themselves
if ! ls "$PY_DIR"/pyrealsense2.cpython-*.so >/dev/null 2>&1; then
    cp -P "$SRC_DIR"/build/Release/pyrealsense2.cpython-*.so* "$PY_DIR/"
fi
[[ -f $PY_DIR/__init__.py ]] || install -m 644 "$SRC_DIR/wrappers/python/pyrealsense2/__init__.py" "$PY_DIR/__init__.py"
ldconfig

# ---------------------------------------------------------------- python extras
if [[ $WITH_PY_EXTRAS == 1 ]]; then
    log "Installing numpy + OpenCV for $PYTHON"
    if ! "$PY_BIN" -m pip --version >/dev/null 2>&1; then
        curl -fsSL "https://bootstrap.pypa.io/pip/$PY_VER/get-pip.py" -o /tmp/get-pip.py
        "$PY_BIN" /tmp/get-pip.py
        rm -f /tmp/get-pip.py
    fi
    # --ignore-installed: apt's python3-numpy (built for 3.6) sits on 3.8's path and would satisfy "numpy<2"
    "$PY_BIN" -m pip install --only-binary :all: --ignore-installed "numpy<2" opencv-python-headless \
        || echo "WARNING: numpy/OpenCV install for $PYTHON failed; pyrealsense2 itself is unaffected"
fi

# ---------------------------------------------------------------- udev
log "Installing udev rules"
cp "$SRC_DIR/config/99-realsense-libusb.rules" /etc/udev/rules.d/
udevadm control --reload-rules
udevadm trigger

# ---------------------------------------------------------------- kernel HID sensor drivers
# librealsense releases the IMU interface between configuring and streaming; each release lets
# hid_sensor_hub + hid_sensor_{accel,gyro}_3d re-probe, which powers one sensor off and resets the
# report intervals, so only accel or only gyro arrives. Keep those drivers away from the camera.
log "Blacklisting kernel HID sensor drivers (they reset the D435i IMU)"
cat > /etc/modprobe.d/blacklist-realsense-imu.conf <<'CONF'
# Installed by install_librealsense.sh: these reset the RealSense IMU configuration on re-bind
blacklist hid_sensor_hub
blacklist hid_sensor_accel_3d
blacklist hid_sensor_gyro_3d
blacklist hid_sensor_custom
CONF
modprobe -r hid_sensor_accel_3d hid_sensor_gyro_3d hid_sensor_custom hid_sensor_trigger \
    hid_sensor_iio_common hid_sensor_hub 2>/dev/null || echo "WARNING: could not unload hid_sensor modules now; reboot to apply"

# ---------------------------------------------------------------- verify
log "Verifying"
"$PY_BIN" -c "import pyrealsense2 as rs; print('pyrealsense2', rs.pyrealsense2.__version__)" \
    || echo "WARNING: pyrealsense2 import failed for $PYTHON (installed to $PY_DIR)"
if rs-enumerate-devices -s; then
    echo
    echo "Done. Use '$PYTHON' (not python3) for pyrealsense2. Try: realsense-viewer, rs-depth, rs-motion (IMU)"
else
    echo "Library installed, but no camera enumerated. Unplug/replug the D435i (USB 3 port) and run: rs-enumerate-devices"
fi
