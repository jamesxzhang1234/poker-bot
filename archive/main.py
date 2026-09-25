"""CLI: parse table state from a video file or a single image.

Usage:
    python -m poker_vision.main --video path/to/gameplay.mp4 --fps 1 --out states.jsonl
    python -m poker_vision.main --image path/to/frame.png
"""
import argparse
import json
import sys

import cv2

from .parser import TableParser


def iter_video_frames(path, sample_fps):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {path}")
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(int(round(native_fps / sample_fps)), 1)
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            yield idx, idx / native_fps, frame
        idx += 1
    cap.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", help="Path to a gameplay video")
    ap.add_argument("--image", help="Path to a single frame/screenshot")
    ap.add_argument("--fps", type=float, default=1.0, help="Sampling rate for --video")
    ap.add_argument("--out", help="Write JSONL output here instead of stdout")
    args = ap.parse_args()

    if not args.video and not args.image:
        ap.error("pass --video or --image")

    parser = TableParser()
    out_fh = open(args.out, "w") if args.out else sys.stdout

    try:
        if args.image:
            frame = cv2.imread(args.image)
            if frame is None:
                raise SystemExit(f"Could not read image: {args.image}")
            state = parser.parse(frame)
            out_fh.write(json.dumps(state.to_dict()) + "\n")
        else:
            for idx, ts, frame in iter_video_frames(args.video, args.fps):
                state = parser.parse(frame, frame_index=idx, timestamp_sec=ts)
                out_fh.write(json.dumps(state.to_dict()) + "\n")
                out_fh.flush()
    finally:
        if args.out:
            out_fh.close()


if __name__ == "__main__":
    main()
