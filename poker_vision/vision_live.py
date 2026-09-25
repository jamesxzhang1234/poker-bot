"""Live capture runner for the Vision path.

Spawns tools/ipad_capture/capture, reads its JSON lines, and prints a parsed
summary per frame -- the same output the push-based server produces, from a
different source.

There is deliberately no work queue here, though not for the reason first
given. This side is now the slower half -- 53ms a frame on a quiet table and
216ms on a busy one, against the helper's ~90ms -- so a queue would build a
backlog without bound and the display would drift behind the game.
newest_first drains whatever piled up and renders only the latest, which
bounds the lag at one parse whatever the capture rate. Frames are also dropped
in the helper, which sets alwaysDiscardsLateVideoFrames and skips frames whose
content hasn't changed.

Usage:
    python -m poker_vision.vision_live
    python -m poker_vision.vision_live --fast --min-interval 0.5
    tools/ipad_capture/capture --dir frames/ipad | python -m poker_vision.vision_live --stdin
"""
import argparse
import json
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

from collections import namedtuple

from . import dashboard, hero_cards, position_tracker as position_tracking, \
    seat_anchors, vision_reader
from .session import DEFAULT_SESSION_FILE, parse_blinds, write_session_info
from .table_state import is_plausible_table
from .text_summary import format_summary

HELPER = Path(__file__).resolve().parent.parent / "tools" / "ipad_capture" / "capture"

Layout = namedtuple("Layout",
                    "anchors board_slots rank_offset hero_cards hero_card_slots")

# Frame size -> that device's layout. Keyed on the captured resolution because
# that is what identifies the device; a frame from anything not listed here is
# skipped rather than run against the nearest layout, since guessing would
# produce a plausible-looking but wrong table.
#
# The iPad is the only supported device. The iPhone was here, reading with
# geometry borrowed from the legacy fixed-region parser, but that geometry was
# never validated against this path and misread hero cards -- see archive/.
_LAYOUT_BY_SIZE = {
    seat_anchors.IPAD_PORTRAIT_SIZE: Layout(
        seat_anchors.IPAD_PORTRAIT,
        seat_anchors.IPAD_BOARD_SLOTS,
        seat_anchors.IPAD_RANK_OFFSET,
        seat_anchors.IPAD_HERO_CARDS,
        seat_anchors.IPAD_HERO_CARD_SLOTS),
}


def layout_for(width, height):
    """The layout for a captured frame size, or None if unrecognised."""
    return _LAYOUT_BY_SIZE.get((width, height))


def anchors_for(width, height):
    """The seat anchors for a captured frame size, or None if unrecognised."""
    layout = layout_for(width, height)
    return layout.anchors if layout else None


def iter_states(lines, anchors=None, on_error=None, hero_tracker=None,
                position_tracker=None):
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

        state = vision_reader.build_table_state(
            line, chosen,
            board_slots=layout.board_slots if layout else None,
            rank_offset=layout.rank_offset if layout else None,
            hero_cards=layout.hero_cards if layout else None)

        # Hole cards are read once a hand, at preflop, and held. See
        # hero_cards.HeroCardTracker for why the holding is safe.
        if hero_tracker is not None and layout is not None:
            cards = hero_tracker.update(
                state.board_cards,
                read=lambda: vision_reader.read_hero_cards(
                    frame, layout.hero_card_slots),
                signature=vision_reader.hero_signature(frame))
            for seat in state.seats:
                if seat.is_hero:
                    seat.hole_cards = list(cards)

        # Positions are worked out pre-flop, while the blind badges are drawn,
        # and held for the rest of the hand. See position_tracker for why.
        if position_tracker is not None:
            position_tracker.update(
                state.seats, signature=vision_reader.hero_signature(frame))

        yield state


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
    state = {"status": None, "frames": 0, "last": None, "fault": None}
    lock = threading.Lock()

    heartbeat = Heartbeat()

    def watch_stderr():
        for raw in proc.stderr:
            message = raw.strip()
            if not message:
                continue
            # Liveness, not a status: the helper reports every tick whether
            # or not anything happened, so showing these would bury the
            # messages that actually need reading.
            if message.startswith("heartbeat "):
                try:
                    heartbeat.beat(float(message.split(None, 1)[1]))
                except (ValueError, IndexError):
                    pass
                continue
            with lock:
                state["status"] = message
                # "capturing: <device>" is progress, not a fault.
                state["fault"] = None if message.startswith("capturing:") else message
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
            state["fault"] = None

    def current_fault():
        with lock:
            return state.get("fault")

    try:
        if args.text:
            run(proc.stdout, require_plausible=not args.all_frames)
        else:
            run_dashboard(newest_first(proc.stdout), draw=draw,
                          on_state=remember, status_of=current_fault,
                          liveness_of=heartbeat.fault)
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


def newest_first(stream, on_drop=None):
    """Yield lines from `stream`, skipping any that piled up while busy.

    The helper emits a frame whenever the screen content changes, which during
    active play is far faster than one can be parsed -- parsing now costs
    ~129ms and spikes higher, since it does board template matching, stack
    recovery and hole cards. Consuming the pipe in order therefore falls
    progressively further behind the table, and because the pipe is an
    unbounded buffer the lag grows without limit rather than settling.

    Dropping to the newest waiting frame fixes that by construction: the
    display is always at most one parse behind, whatever the capture rate.
    Throttling the helper alone would not, since any interval shorter than the
    parse time still accumulates.
    """
    pending = queue.Queue()
    finished = object()

    def pump():
        try:
            for line in stream:
                pending.put(line)
        finally:
            pending.put(finished)

    threading.Thread(target=pump, daemon=True).start()

    while True:
        line = pending.get()
        if line is finished:
            return
        # Anything that arrived while the previous frame was being parsed is
        # already out of date; keep only the last of it.
        dropped = 0
        while True:
            try:
                following = pending.get_nowait()
            except queue.Empty:
                break
            if following is finished:
                if on_drop and dropped:
                    on_drop(dropped)
                yield line
                return
            line = following
            dropped += 1
        if on_drop and dropped:
            on_drop(dropped)
        yield line


# How long the display may go without a frame before it says so. The helper
# emits only on a screen change, and a table can sit still between hands, so
# this has to be longer than a quiet table's gaps -- but short enough to
# notice before the next decision. The helper's own starvation warning fires
# at 5s; this is deliberately a shade under, so the display admits it is
# holding before the helper has worked out why.
STALE_AFTER = 4.0


# The helper's own starvation threshold, and how long it may go without
# saying anything at all before we assume it is wedged rather than quiet. It
# reports every 2s, so 8s is four missed reports.
HELPER_STARVED_AFTER = 5.0
HELPER_SILENT_AFTER = 8.0


class Heartbeat:
    """Liveness as the capture helper reports it, rather than as guessed here.

    This side sees only frames that got past the helper's content-diff gate,
    which a static table suppresses for as long as the players are thinking.
    So the gap between arriving frames says nothing about whether capture is
    healthy, and treating it as if it did made a quiet table flicker in and
    out of HELD -- which teaches you to ignore the one warning that matters.

    The helper tracks arrivals *before* the gate and reports every tick, so
    it can tell the two apart. Three states worth naming: frames arriving
    (fine, however static the screen), frames not arriving (the iPad is
    asleep, locked, or taken by QuickTime), and the helper not reporting at
    all (it died or wedged, which its own watchdog cannot tell us about).
    """

    def __init__(self, starved_after=HELPER_STARVED_AFTER,
                 silent_after=HELPER_SILENT_AFTER, now=time.monotonic):
        self._starved_after = starved_after
        self._silent_after = silent_after
        self._now = now
        self._last_beat = None
        self._quiet = 0.0

    def beat(self, quiet):
        """Record a report: `quiet` is seconds since the helper last had a
        frame, or negative if it has not had one yet."""
        self._last_beat = self._now()
        self._quiet = quiet

    def fault(self):
        """A reason the display should be marked stale, or None."""
        if self._last_beat is None:
            # Nothing heard yet. Startup, or a replayed stream with no helper
            # behind it; either way there is nothing to disown.
            return None
        if self._now() - self._last_beat > self._silent_after:
            return "capture helper stopped reporting -- restart start poker"
        if self._quiet > self._starved_after:
            return ("no frames arriving -- is the iPad awake and unlocked, "
                    "and is QuickTime closed?")
        return None


class StaleWatch:
    """Whether the display is showing a frame old enough to disown.

    run_dashboard redraws only when a frame arrives, so when the helper goes
    quiet -- the iPad sleeps, the capture session dies, the cable moves --
    the loop blocks and the terminal keeps showing the last frame it ever
    got. Seen live: the board and hole cards of a hand that had finished
    minutes earlier, with nothing to say so.

    A stale table that looks live is the worst thing this display can do. It
    is worse than a blank one, and much worse than an error, because every
    value on it is individually plausible.
    """

    def __init__(self, stale_after=STALE_AFTER, now=time.monotonic):
        self._stale_after = stale_after
        self._now = now
        self._last = now()
        self.ever_arrived = False

    def note_arrival(self):
        self._last = self._now()
        self.ever_arrived = True

    def overdue(self):
        return self._now() - self._last > self._stale_after


def run_dashboard(lines, draw=dashboard.draw_to_terminal, status=None,
                  on_state=None, status_of=None, hero_tracker=None,
                  position_tracker=None, stale_after=STALE_AFTER,
                  stale_poll=1.0, liveness_of=None):
    """Render each frame in place. Returns how many frames were drawn.

    Nothing in here is allowed to end the session. A bad line on stdout, a
    state that will not render -- the table is mid-hand and the next frame is
    a moment away, so both are reported and stepped over.
    """
    previous = None
    drawn = 0
    if hero_tracker is None:
        hero_tracker = hero_cards.HeroCardTracker()
    if position_tracker is None:
        position_tracker = position_tracking.PositionTracker()

    watch = StaleWatch(stale_after=stale_after)
    lock = threading.Lock()
    finished = threading.Event()

    def mark_stale_while_quiet():
        # The only part of the display not driven by an arriving frame. It
        # exists because "no frame arrived" cannot be noticed by a loop that
        # only runs when one does.
        while not finished.wait(stale_poll):
            if not watch.ever_arrived:
                continue
            if liveness_of is not None:
                # The helper knows whether frames are arriving; this side
                # only knows whether any got past its content-diff gate. Ask.
                reason = liveness_of()
                if reason is None:
                    continue
            else:
                # No helper behind the stream (a replay). Fall back to the
                # gap between frames, which is all there is to go on.
                if not watch.overdue():
                    continue
                reason = None
            with lock:
                if previous is None:
                    continue
                reported = reason or (status_of() if status_of else None)
                try:
                    draw(dashboard.render(
                        previous, frames=drawn,
                        status=reported or "no frames arriving -- "
                                           "is the iPad awake and unlocked?"))
                except Exception:
                    pass

    threading.Thread(target=mark_stale_while_quiet, daemon=True).start()
    try:
        for state in iter_states(lines, on_error=lambda m: None,
                                 hero_tracker=hero_tracker,
                                 position_tracker=position_tracker):
            watch.note_arrival()
            # Tabbing away from the app, or a full-screen modal, yields a
            # frame with no table in it. Hold the last real one rather than
            # blanking the display -- but redraw it marked HELD, so a stale
            # table is never mistaken for a live one.
            if not state.seats:
                if previous is not None:
                    # A fault the helper has reported outranks this. Locking
                    # the iPad shows the lock screen for one frame and then
                    # goes silent, and "no table on screen" would otherwise
                    # bury "unlock the iPad screen" for the rest of the lock.
                    reported = status_of() if status_of else None
                    with lock:
                        try:
                            draw(dashboard.render(
                                previous, frames=drawn,
                                status=reported or "no table on screen"))
                        except Exception:
                            pass
                continue
            with lock:
                try:
                    draw(dashboard.render(state, previous=previous,
                                          frames=drawn + 1))
                except Exception:
                    # A rendering fault must not take the capture down.
                    continue
                previous = state
                drawn += 1
            if on_state:
                on_state(state)
    finally:
        finished.set()
    return drawn


if __name__ == "__main__":
    main()
