"""Flask server: receives screenshots pushed from the phone, queues them for
processing, and a background worker thread parses each one with
TableParser and prints a summary to stdout.

See docs/superpowers/specs/2026-09-21-phone-screenshot-capture-design.md.

Usage:
    python -m poker_vision.server
"""
import argparse
import json
import queue
import re
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
from flask import Flask, request

from .parser import TableParser
from poker_vision.table_state import is_plausible_table
from poker_vision.text_summary import format_summary

DEFAULT_INCOMING_DIR = Path("frames/incoming")


def create_app(incoming_dir, work_queue):
    incoming_dir = Path(incoming_dir)
    incoming_dir.mkdir(parents=True, exist_ok=True)
    app = Flask(__name__)

    @app.route("/upload", methods=["POST"])
    def upload():
        data = request.get_data()
        if not data:
            return ("empty body", 400)
        ts = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        path = incoming_dir / f"{ts}.jpg"
        path.write_bytes(data)
        work_queue.put(str(path))
        return ("", 200)

    return app


def worker_loop(work_queue, parser_factory=TableParser, frame_index_start=0):
    """Pulls image paths off work_queue and prints a parsed summary for
    each. Put None on the queue to stop the loop (used by tests and by a
    clean shutdown)."""
    parser = parser_factory()
    frame_index = frame_index_start
    while True:
        path = work_queue.get()
        if path is None:
            break
        frame = cv2.imread(path)
        if frame is None:
            print(f"[parse error] unreadable image: {path}")
            continue
        try:
            state = parser.parse(frame, frame_index=frame_index, timestamp_sec=time.time())
        except Exception as exc:
            print(f"[parse error] {exc}")
            continue
        state_dict = state.to_dict()
        if not is_plausible_table(state_dict):
            print(f"[skipped] not a PurePoker table screenshot: {path}")
            continue
        frame_index += 1
        print(format_summary(state_dict))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blinds", help="Blind sizes as SB/BB, e.g. 1/2")
    args = ap.parse_args()

    if args.blinds:
        try:
            small_blind, big_blind = parse_blinds(args.blinds)
        except ValueError as exc:
            raise SystemExit(str(exc))
        write_session_info(DEFAULT_SESSION_FILE, args.blinds, small_blind, big_blind)
        print(f"Blinds set to {args.blinds} (SB={small_blind}, BB={big_blind}) -- saved to {DEFAULT_SESSION_FILE}")

    work_queue = queue.Queue()
    app = create_app(DEFAULT_INCOMING_DIR, work_queue)

    worker = threading.Thread(target=worker_loop, args=(work_queue,), daemon=True)
    worker.start()

    app.run(host="0.0.0.0", port=8765)


if __name__ == "__main__":
    main()
