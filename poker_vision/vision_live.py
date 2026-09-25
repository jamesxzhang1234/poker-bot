"""Live capture runner for the Vision path.

Spawns tools/ipad_capture/capture, reads its JSON lines, and prints a parsed
summary per frame -- the same output the push-based server produces, from a
different source.

There is deliberately no work queue here. The helper spends ~90ms per frame in
Vision while this side spends ~0.15ms turning the payload into a TableState,
so the consumer cannot fall behind the producer and the pipe is buffer enough.
Frames are dropped where dropping belongs: in the helper, which sets
alwaysDiscardsLateVideoFrames and skips frames whose content hasn't changed.

Usage:
    python -m poker_vision.vision_live
    python -m poker_vision.vision_live --fast --min-interval 0.5
    tools/ipad_capture/capture --dir frames/ipad | python -m poker_vision.vision_live --stdin
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from collections import namedtuple

from . import regions, seat_anchors, vision_reader
from .table_state import is_plausible_table
from .text_summary import format_summary

HELPER = Path(__file__).resolve().parent.parent / "tools" / "ipad_capture" / "capture"

Layout = namedtuple("Layout", "anchors board_slots rank_offset")

# Frame size -> that device's layout. Keyed on the captured resolution because
# that is what identifies the device; a frame from anything not listed here is
# skipped rather than run against the nearest layout, since guessing would
# produce a plausible-looking but wrong table.
_LAYOUT_BY_SIZE = {
    seat_anchors.IPAD_PORTRAIT_SIZE: Layout(
        seat_anchors.IPAD_PORTRAIT,
        seat_anchors.IPAD_BOARD_SLOTS,
        seat_anchors.IPAD_RANK_OFFSET),
    # The iPhone's board geometry already lives in regions.py, measured by
    # hand for the fixed-region parser; reuse it rather than keeping a second
    # copy that can drift. Its slots are at the reference resolution, which is
    # this frame size exactly, so no rescaling is needed here.
    seat_anchors.IPHONE_PORTRAIT_SIZE: Layout(
        seat_anchors.IPHONE_PORTRAIT,
        list(regions.BOARD_SLOTS),
        regions.RANK_OFFSET),
}


def layout_for(width, height):
    """The layout for a captured frame size, or None if unrecognised."""
    return _LAYOUT_BY_SIZE.get((width, height))


def anchors_for(width, height):
    """The seat anchors for a captured frame size, or None if unrecognised."""
    layout = layout_for(width, height)
    return layout.anchors if layout else None


def iter_states(lines, anchors=None, on_error=None):
    """Yield a TableState for each usable payload line.

    A bad line is reported and skipped, never raised: a truncated or
    interleaved write on stdout must not end a capture session mid-hand.
    """
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            frame = vision_reader.parse_payload(line)
        except (ValueError, KeyError, TypeError) as exc:
            if on_error:
                on_error(f"unreadable payload: {exc}")
            continue
        layout = layout_for(frame.width, frame.height)
        chosen = anchors or (layout.anchors if layout else None)
        if chosen is None:
            if on_error:
                on_error(f"no layout for {frame.width}x{frame.height}")
            continue
        yield vision_reader.build_table_state(
            line, chosen,
            board_slots=layout.board_slots if layout else None,
            rank_offset=layout.rank_offset if layout else None)


def run(lines, anchors=None, out=print, require_plausible=True):
    """Print a summary per frame. Returns how many frames were summarised."""
    shown = 0
    for state in iter_states(lines, anchors=anchors,
                             on_error=lambda m: print(f"[skipped] {m}", file=sys.stderr)):
        state_dict = state.to_dict()
        if require_plausible and not is_plausible_table(state_dict):
            print("[skipped] not a plausible table", file=sys.stderr)
            continue
        out(format_summary(state_dict))
        shown += 1
    return shown


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stdin", action="store_true",
                    help="read payloads from stdin instead of spawning the helper")
    ap.add_argument("--fast", action="store_true",
                    help="use Vision's .fast recognition (~14ms, fewer detections)")
    ap.add_argument("--min-interval", type=float, default=0.0,
                    help="seconds to wait between captured frames")
    ap.add_argument("--all-frames", action="store_true",
                    help="summarise every frame, even ones that don't look like a table")
    args = ap.parse_args()

    # Python block-buffers stdout when it is a pipe or a file rather than a
    # terminal, which for a live stream means frames sit unflushed until the
    # buffer fills or the process exits -- so piping this into anything, or
    # killing it, silently loses everything captured so far.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:      # pragma: no cover - very old Python
        pass

    if args.stdin:
        run(sys.stdin, require_plausible=not args.all_frames)
        return

    if not HELPER.exists():
        raise SystemExit(
            f"capture helper not built: {HELPER}\n"
            f"  cd tools/ipad_capture && swiftc -O capture.swift -o capture")

    cmd = [str(HELPER), "--live"]
    if args.fast:
        cmd.append("--fast")
    if args.min_interval:
        cmd += ["--min-interval", str(args.min_interval)]

    # stderr is left attached to the terminal so the helper's own diagnostics
    # -- "device attached but exposes no capture interface", which means
    # unlock it or quit QuickTime -- reach the user unfiltered.
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, bufsize=1)
    try:
        run(proc.stdout, require_plausible=not args.all_frames)
    except KeyboardInterrupt:
        pass
    finally:
        proc.terminate()
        proc.wait(timeout=5)
    raise SystemExit(proc.returncode or 0)


if __name__ == "__main__":
    main()
