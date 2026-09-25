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
import threading
from pathlib import Path

from collections import namedtuple

from . import dashboard, regions, seat_anchors, vision_reader
from .table_state import is_plausible_table
from .text_summary import format_summary

HELPER = Path(__file__).resolve().parent.parent / "tools" / "ipad_capture" / "capture"

Layout = namedtuple("Layout", "anchors board_slots rank_offset hero_cards")

# Frame size -> that device's layout. Keyed on the captured resolution because
# that is what identifies the device; a frame from anything not listed here is
# skipped rather than run against the nearest layout, since guessing would
# produce a plausible-looking but wrong table.
_LAYOUT_BY_SIZE = {
    seat_anchors.IPAD_PORTRAIT_SIZE: Layout(
        seat_anchors.IPAD_PORTRAIT,
        seat_anchors.IPAD_BOARD_SLOTS,
        seat_anchors.IPAD_RANK_OFFSET,
        seat_anchors.IPAD_HERO_CARDS),
    # The iPhone's board geometry already lives in regions.py, measured by
    # hand for the fixed-region parser; reuse it rather than keeping a second
    # copy that can drift. Its slots are at the reference resolution, which is
    # this frame size exactly, so no rescaling is needed here.
    seat_anchors.IPHONE_PORTRAIT_SIZE: Layout(
        seat_anchors.IPHONE_PORTRAIT,
        list(regions.BOARD_SLOTS),
        regions.RANK_OFFSET,
        list(regions.HERO_CARDS)),
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
        # A plate that lands on no anchor means this device's layout has a
        # seat we do not describe. Say so: the alternative is a player who
        # quietly never appears in the output.
        if on_error:
            orphans = vision_reader.plates_without_seat(
                vision_reader.find_seat_plates(frame.observations,
                                               frame.width, frame.height),
                chosen, frame.width, frame.height)
            for plate in orphans:
                on_error(f"plate matches no seat anchor: "
                         f"{plate.name!r} at ({plate.x},{plate.y})")

        yield vision_reader.build_table_state(
            line, chosen,
            board_slots=layout.board_slots if layout else None,
            rank_offset=layout.rank_offset if layout else None,
            hero_cards=layout.hero_cards if layout else None)


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
    ap.add_argument("--text", action="store_true",
                    help="print a scrolling summary per frame instead of the dashboard")
    ap.add_argument("--fast", action="store_true",
                    help="use Vision's .fast recognition (~14ms, fewer detections)")
    ap.add_argument("--min-interval", type=float, default=0.0,
                    help="seconds to wait between captured frames")
    ap.add_argument("--blinds", help="blind sizes as SB/BB, e.g. 0.05/0.1")
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

    if args.blinds:
        record_blinds(args.blinds)

    if args.stdin:
        if args.text:
            run(sys.stdin, require_plausible=not args.all_frames)
        else:
            run_dashboard(sys.stdin)
        return

    proc = spawn_helper(fast=args.fast, min_interval=args.min_interval)

    # The helper reports its own state on stderr -- "waiting: unlock the
    # device", "lost the device" -- and those are the messages that tell the
    # user what to fix. Show them in the waiting panel rather than letting
    # them scroll past behind the dashboard.
    state = {"status": None, "frames": 0, "last": None}
    lock = threading.Lock()

    def watch_stderr():
        for raw in proc.stderr:
            message = raw.strip()
            if not message:
                continue
            with lock:
                state["status"] = message
                if args.text:
                    print(f"[{message}]", file=sys.stderr)
                elif state["last"] is None:
                    # "capturing: <device>" means the feed is up and we are
                    # simply not looking at a table yet -- the lobby, say.
                    dashboard.draw_to_terminal(dashboard.render_waiting(
                        message, connected=message.startswith("capturing:")))
                else:
                    # Frames have stopped arriving -- when the iPad locks none
                    # come at all -- so the table on screen is stale. Redraw it
                    # marked HELD rather than leaving it looking live.
                    dashboard.draw_to_terminal(
                        dashboard.render(state["last"], frames=state["frames"],
                                         status=message))

    threading.Thread(target=watch_stderr, daemon=True).start()

    if not args.text:
        dashboard.draw_to_terminal(dashboard.render_waiting())

    def draw(text):
        with lock:
            state["frames"] += 1
            dashboard.draw_to_terminal(text)

    def remember(table):
        with lock:
            state["last"] = table

    try:
        if args.text:
            run(proc.stdout, require_plausible=not args.all_frames)
        else:
            run_dashboard(proc.stdout, draw=draw, on_state=remember)
    except KeyboardInterrupt:
        pass
    except Exception as exc:                # never end on a traceback
        print(f"\ncapture stopped: {exc}", file=sys.stderr)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:   # pragma: no cover
            proc.kill()

    # Reaching here means the helper's stdout closed: it exited, or was
    # stopped. Say which, with whatever it last reported.
    with lock:
        last = state["status"]
    if state["frames"] == 0:
        print("\nno frames were captured." + (f" last status: {last}" if last else ""),
              file=sys.stderr)
    raise SystemExit(0)




def record_blinds(blinds, path=None):
    """Persist the operator-entered blinds, as the push server does.

    Kept so a later decision component can read the stakes without re-parsing
    a command line -- the capture path changed, that contract did not.
    """
    from .server import DEFAULT_SESSION_FILE, parse_blinds, write_session_info
    try:
        small, big = parse_blinds(blinds)
    except ValueError as exc:
        raise SystemExit(str(exc))
    target = Path(path) if path else DEFAULT_SESSION_FILE
    write_session_info(target, blinds, small, big)
    return small, big


def spawn_helper(helper=HELPER, fast=False, min_interval=0.0):
    """Start the capture helper, or explain how to build it.

    A missing binary is a setup problem with a one-line fix, so it reports
    that rather than a traceback.
    """
    helper = Path(helper)
    if not helper.exists():
        raise SystemExit(
            f"capture helper not built: {helper}\n"
            f"  cd tools/ipad_capture && swiftc -O capture.swift -o capture")
    cmd = [str(helper), "--live"]
    if fast:
        cmd.append("--fast")
    if min_interval:
        cmd += ["--min-interval", str(min_interval)]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, bufsize=1)


def run_dashboard(lines, draw=dashboard.draw_to_terminal, status=None,
                  on_state=None):
    """Render each frame in place. Returns how many frames were drawn.

    Nothing in here is allowed to end the session. A bad line on stdout, a
    state that will not render -- the table is mid-hand and the next frame is
    a moment away, so both are reported and stepped over.
    """
    previous = None
    drawn = 0
    for state in iter_states(lines, on_error=lambda m: None):
        # Tabbing away from the app, or a full-screen modal, yields a frame
        # with no table in it. Hold the last real one rather than blanking the
        # display -- but redraw it marked HELD, so a stale table is never
        # mistaken for a live one.
        if not state.seats:
            if previous is not None:
                try:
                    draw(dashboard.render(previous, frames=drawn,
                                          status="no table on screen"))
                except Exception:
                    pass
            continue
        try:
            draw(dashboard.render(state, previous=previous, frames=drawn + 1))
        except Exception:
            # A rendering fault must not take the capture down with it.
            continue
        previous = state
        drawn += 1
        if on_state:
            on_state(state)
    return drawn


if __name__ == "__main__":
    main()
