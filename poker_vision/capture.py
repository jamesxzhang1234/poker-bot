"""Frame sources for the live app: a camera/virtual-camera device, or a
fixed region of the screen. Both expose the same read()/release() interface
so the rest of the app doesn't care which one is in use.
"""
import re
import subprocess

import cv2
import numpy as np


def list_avfoundation_devices():
    """Return a list of (index, name) for available AVFoundation video
    capture devices (cameras, virtual cameras, 'Capture screen N')."""
    proc = subprocess.run(
        ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
        capture_output=True, text=True,
    )
    devices = []
    in_video_section = False
    for line in proc.stderr.splitlines():
        if "AVFoundation video devices" in line:
            in_video_section = True
            continue
        if "AVFoundation audio devices" in line:
            break
        if in_video_section:
            m = re.search(r"\[(\d+)\]\s+(.+)$", line)
            if m:
                devices.append((int(m.group(1)), m.group(2).strip()))
    return devices


class CameraSource:
    """Reads frames from a camera or virtual-camera device index."""

    def __init__(self, index):
        self.cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
        if not self.cap.isOpened():
            raise SystemExit(
                f"Could not open camera device {index}. Run "
                f"`python -m poker_vision.capture --list` to see available devices."
            )

    def read(self):
        ok, frame = self.cap.read()
        return frame if ok else None

    def release(self):
        self.cap.release()


class ScreenRegionSource:
    """Reads frames from a fixed rectangle of the screen via mss."""

    def __init__(self, x, y, w, h):
        import mss  # imported lazily so mss is only required for this path
        self._mss = mss.mss()
        self.monitor = {"left": x, "top": y, "width": w, "height": h}

    def read(self):
        shot = self._mss.grab(self.monitor)
        frame = np.array(shot)  # BGRA
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    def release(self):
        self._mss.close()


def open_source(spec):
    """spec: 'camera:<index>' or 'region:x,y,w,h'.

    Note: there is no reliable 'obs' auto-detect. OpenCV's AVFoundation
    device numbering does NOT match `ffmpeg -f avfoundation -list_devices`
    (OpenCV's backend silently excludes some device types, e.g. "Desk View
    Camera" and "Capture screen N"), so a device's ffmpeg index cannot be
    used to open it through OpenCV. Use `tools/probe_cameras.py` to find the
    right index empirically, then pass it as `camera:<index>`.
    """
    kind, _, rest = spec.partition(":")
    if kind == "camera":
        return CameraSource(int(rest))
    if kind == "region":
        x, y, w, h = (int(v) for v in rest.split(","))
        return ScreenRegionSource(x, y, w, h)
    raise SystemExit(f"Unknown source spec: {spec!r} (expected camera:N, obs, or region:x,y,w,h)")


if __name__ == "__main__":
    import sys
    if "--list" in sys.argv:
        for idx, name in list_avfoundation_devices():
            print(f"[{idx}] {name}")
