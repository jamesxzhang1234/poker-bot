"""Live viewer: capture a frame source, parse it periodically in the
background, and display an annotated view + parsed state panel.

Usage:
    python -m poker_vision.live_app --source camera:2
    python -m poker_vision.live_app --source region:100,100,600,1300 --parse-interval 1.5

Find the right camera index with tools/probe_cameras.py first -- OpenCV's
own device numbering does not match `ffmpeg -f avfoundation -list_devices`.

Press 'q' to quit, 'p' to force an immediate re-parse.
"""
import argparse
import threading
import time

import cv2

from . import capture, overlay
from .parser import TableParser


class LiveState:
    def __init__(self):
        self.lock = threading.Lock()
        self.frame = None
        self.state_dict = None
        self.last_parse_ts = None
        self.parsing = False
        self.force_reparse = False
        self.stop = False


def parse_worker(shared, interval):
    parser = TableParser()
    while not shared.stop:
        with shared.lock:
            frame = shared.frame.copy() if shared.frame is not None else None
            force = shared.force_reparse
            shared.force_reparse = False

        if frame is None:
            time.sleep(0.1)
            continue

        with shared.lock:
            shared.parsing = True
        try:
            table_state = parser.parse(frame)
            state_dict = table_state.to_dict()
        except Exception as exc:  # keep the live loop alive even if one parse fails
            print(f"[parse error] {exc}")
            state_dict = None

        with shared.lock:
            shared.parsing = False
            if state_dict is not None:
                shared.state_dict = state_dict
                shared.last_parse_ts = time.time()

        # Wait out the interval, but wake early if the user forces a re-parse.
        waited = 0.0
        while waited < interval and not shared.stop:
            if shared.force_reparse:
                break
            time.sleep(0.1)
            waited += 0.1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True,
                    help="camera:<index> | region:x,y,w,h (find index with tools/probe_cameras.py)")
    ap.add_argument("--parse-interval", type=float, default=1.5,
                    help="Seconds between full-table parses (parsing is OCR-heavy and slow)")
    ap.add_argument("--display-height", type=int, default=900)
    args = ap.parse_args()

    src = capture.open_source(args.source)
    shared = LiveState()

    worker = threading.Thread(target=parse_worker, args=(shared, args.parse_interval), daemon=True)
    worker.start()

    window = "Poker Vision - Live"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    try:
        while True:
            frame = src.read()
            if frame is None:
                print("No frame from source; retrying...")
                time.sleep(0.2)
                continue

            with shared.lock:
                shared.frame = frame
                state_dict = shared.state_dict
                last_parse_ts = shared.last_parse_ts
                parsing = shared.parsing

            canvas = overlay.compose(frame, state_dict, last_parse_ts, parsing,
                                      display_height=args.display_height)
            cv2.imshow(window, canvas)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("p"):
                with shared.lock:
                    shared.force_reparse = True
    finally:
        shared.stop = True
        src.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
