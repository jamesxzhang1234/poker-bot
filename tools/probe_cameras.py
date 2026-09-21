"""Try opening camera indices 0-8 directly through OpenCV's AVFoundation
backend (independent of ffmpeg's device numbering) and save one snapshot
from each index that actually opens, so you can look at the images and
tell which index is your phone.

Usage: python tools/probe_cameras.py
"""
import cv2
import os

OUT_DIR = "frames/crops/probe"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for idx in range(9):
        cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            print(f"[{idx}] could not open")
            cap.release()
            continue
        ok, frame = cap.read()
        if not ok or frame is None:
            print(f"[{idx}] opened but no frame")
            cap.release()
            continue
        path = f"{OUT_DIR}/index_{idx}.png"
        cv2.imwrite(path, frame)
        print(f"[{idx}] OK - {frame.shape} - saved {path}")
        cap.release()


if __name__ == "__main__":
    main()
