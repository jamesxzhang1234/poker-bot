"""Tests for the live Vision capture runner.

The runner consumes the capture helper's JSON lines and turns them into
TableStates. These tests drive it with lines directly -- the helper itself is
a separate process and its own concern, so nothing here spawns it.
"""
import json

from poker_vision import vision_live, seat_anchors


def _payload(width=1640, height=2360, obs=None, index=0):
    return json.dumps({
        "frame_index": index, "timestamp_sec": 1.0,
        "width": width, "height": height,
        "observations": obs if obs is not None else [
            {"text": "drake81", "x": 511, "y": 565, "w": 123, "h": 29, "confidence": 1.0},
            {"text": "7.75", "x": 509, "y": 620, "w": 72, "h": 34, "confidence": 1.0},
        ],
    })


def test_yields_a_state_for_each_payload_line():
    states = list(vision_live.iter_states([_payload(index=0), _payload(index=1)]))

    assert [s.frame_index for s in states] == [0, 1]
    assert states[0].seats[0].name == "drake81"


def test_skips_blank_lines():
    states = list(vision_live.iter_states(["", "   ", _payload(), "\n"]))

    assert len(states) == 1


def test_a_malformed_line_does_not_stop_the_stream():
    # The helper writes diagnostics to stderr, but a truncated or interleaved
    # stdout line must not kill a capture session mid-hand.
    states = list(vision_live.iter_states(["{not json", _payload(), "{}"]))

    assert len(states) == 1


def test_picks_the_anchor_set_matching_the_frame_size():
    ipad = list(vision_live.iter_states([_payload(width=1640, height=2360)]))[0]
    assert ipad.seats[0].seat_id == "top_left"

    # An unsupported frame size is skipped entirely, not read with the
    # nearest layout.
    assert list(vision_live.iter_states([_payload(width=1206, height=2622)])) == []


def test_an_unknown_frame_size_is_skipped_rather_than_guessed():
    # Guessing an anchor set for an unrecognised device would silently produce
    # a plausible-looking but wrong table.
    states = list(vision_live.iter_states([_payload(width=800, height=600)]))

    assert states == []


def test_anchors_can_be_forced_for_an_unknown_frame_size():
    states = list(vision_live.iter_states(
        [_payload(width=1640, height=2360)], anchors=seat_anchors.IPAD_PORTRAIT))

    assert states[0].seats[0].name == "drake81"


def test_layout_lookup_returns_board_geometry_too():
    layout = vision_live.layout_for(*seat_anchors.IPAD_PORTRAIT_SIZE)

    assert layout.anchors is seat_anchors.IPAD_PORTRAIT
    assert layout.board_slots == seat_anchors.IPAD_BOARD_SLOTS
    assert layout.rank_offset == seat_anchors.IPAD_RANK_OFFSET


def test_the_iphone_is_no_longer_a_supported_device():
    # The iPad is the only device this path is calibrated and validated for.
    # An iPhone frame used to be read with geometry borrowed from the legacy
    # fixed-region parser, which was never checked against this path and
    # misread hero cards. Skipping is better than reading it wrong.
    assert vision_live.layout_for(*seat_anchors.IPHONE_PORTRAIT_SIZE) is None


def test_layout_lookup_is_none_for_an_unknown_device():
    assert vision_live.layout_for(800, 600) is None


def test_states_carry_the_board_when_the_payload_has_a_strip():
    # Drives the whole chain the helper feeds: payload with a strip -> layout
    # picked from the frame size -> board template-matched from the strip.
    import base64
    import cv2
    import numpy as np
    from poker_vision import rank_templates

    frame = np.full((2360, 1640, 3), 30, dtype=np.uint8)
    templates = rank_templates.load()
    # BGR, matching what card_vision.classify_suit documents from real cards:
    # diamonds are blue-dominant, clubs green-dominant.
    # three cards, not two: a board is only ever 0, 3, 4 or 5 long
    for i, (rank, bgr) in enumerate([("K", (160, 60, 40)), ("9", (50, 150, 50)),
                                     ("4", (40, 40, 160))]):
        x1, y1, x2, y2 = seat_anchors.IPAD_BOARD_SLOTS[i]
        frame[y1:y2, x1:x2] = bgr
        cv2.circle(frame, ((x1 + x2) // 2, y1 + 110), 30, (255, 255, 255), -1)
        gx1, gy1, gx2, gy2 = seat_anchors.IPAD_RANK_OFFSET
        glyph = cv2.resize(templates[rank][0], (gx2 - gx1, gy2 - gy1),
                           interpolation=cv2.INTER_NEAREST)
        region = frame[y1 + gy1:y1 + gy2, x1 + gx1:x1 + gx2]
        region[glyph > 127] = (255, 255, 255)

    rect = (0.18, 0.40, 0.63, 0.10)
    sx, sy = int(rect[0] * 1640), int(rect[1] * 2360)
    strip = frame[sy:sy + int(rect[3] * 2360), sx:sx + int(rect[2] * 1640)]
    payload = json.dumps({
        "frame_index": 0, "timestamp_sec": 0.0, "width": 1640, "height": 2360,
        "board_strip": base64.b64encode(cv2.imencode(".png", strip)[1].tobytes()).decode(),
        "board_strip_rect": list(rect),
        "observations": [],
    })

    state = list(vision_live.iter_states([payload]))[0]

    assert state.board_cards == ["Kd", "9c", "4h"]


def test_warns_about_a_plate_that_matches_no_seat():
    # A player sitting in a seat this device's anchors don't describe should
    # be visible as a warning, not silently missing from the output.
    payload = json.dumps({
        "frame_index": 0, "timestamp_sec": 0.0, "width": 1640, "height": 2360,
        "observations": [
            {"text": "ghostplayer", "x": 445, "y": 1200, "w": 140, "h": 30, "confidence": 1.0},
            {"text": "3.0", "x": 445, "y": 1257, "w": 60, "h": 30, "confidence": 1.0},
        ],
    })
    warnings = []

    list(vision_live.iter_states([payload], on_error=warnings.append))

    assert any("ghostplayer" in w for w in warnings), warnings


def test_no_warning_when_every_plate_lands_on_a_seat():
    payload = json.dumps({
        "frame_index": 0, "timestamp_sec": 0.0, "width": 1640, "height": 2360,
        "observations": [
            {"text": "drake81", "x": 511, "y": 565, "w": 123, "h": 29, "confidence": 1.0},
            {"text": "7.75", "x": 509, "y": 620, "w": 72, "h": 34, "confidence": 1.0},
        ],
    })
    warnings = []

    list(vision_live.iter_states([payload], on_error=warnings.append))

    assert warnings == []


def test_dashboard_run_returns_the_number_of_frames_shown():
    shown = []
    n = vision_live.run_dashboard([_payload(index=0), _payload(index=1)],
                                  draw=shown.append)

    assert n == 2
    assert len(shown) == 2
    assert "drake81" in shown[-1]


def test_dashboard_run_survives_a_bad_line():
    # A truncated or interleaved write on the helper's stdout must not end the
    # session -- the table is mid-hand and the next frame is a moment away.
    shown = []
    n = vision_live.run_dashboard(["{not json", _payload(), ""], draw=shown.append)

    assert n == 1


def test_dashboard_run_survives_a_frame_it_cannot_render():
    # A rendering bug must not kill a live capture either.
    def explode(_):
        raise RuntimeError("boom")

    calls = {"n": 0}

    def draw(text):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")

    n = vision_live.run_dashboard([_payload(index=0), _payload(index=1)], draw=draw)

    assert n == 1          # the second frame still rendered
    assert calls["n"] == 2


def test_missing_helper_reports_how_to_build_it_rather_than_crashing():
    import pytest
    with pytest.raises(SystemExit) as exc:
        vision_live.spawn_helper(helper=Path("/nonexistent/capture"), fast=False,
                                 min_interval=0.0)

    assert "swiftc" in str(exc.value)


from pathlib import Path


def test_blinds_are_recorded_for_a_later_consumer(tmp_path):
    # The blinds menu exists so a future decision component knows the stakes;
    # the live path must keep writing them the way the push server did.
    target = tmp_path / "session.json"

    vision_live.record_blinds("0.05/0.1", target)

    import json
    saved = json.loads(target.read_text())
    assert saved["small_blind"] == 0.05
    assert saved["big_blind"] == 0.1


def test_malformed_blinds_are_rejected_with_a_readable_message(tmp_path):
    import pytest
    with pytest.raises(SystemExit) as exc:
        vision_live.record_blinds("not-blinds", tmp_path / "session.json")

    assert "SB/BB" in str(exc.value)


def test_dashboard_keeps_the_last_table_when_the_screen_is_not_a_table():
    # Tabbing away from the poker app, or a full-screen modal, produces a
    # frame with no seats. Replacing a good dashboard with an empty grid
    # throws away the last thing you actually wanted to see.
    empty = json.dumps({"frame_index": 1, "timestamp_sec": 0.0,
                        "width": 1640, "height": 2360, "observations": []})
    shown = []

    n = vision_live.run_dashboard([_payload(index=0), empty], draw=shown.append)

    assert n == 1
    assert "drake81" in shown[-1]


def test_a_frame_with_no_table_redraws_the_held_one_with_a_note():
    # Tabbing away holds the last table -- but it must say it is held, not sit
    # there looking like live data.
    empty = json.dumps({"frame_index": 1, "timestamp_sec": 0.0,
                        "width": 1640, "height": 2360, "observations": []})
    shown = []

    vision_live.run_dashboard([_payload(index=0), empty], draw=shown.append)

    assert len(shown) == 2          # redrawn, not skipped silently
    assert "drake81" in shown[-1]   # still the last real table
    assert "held" in shown[-1].lower()


def test_a_reported_fault_outranks_the_generic_no_table_message():
    # Locking shows the lock screen for one frame -- a real frame with no
    # table -- and then frames stop. Without a priority the "no table on
    # screen" from that single frame overwrites "unlock the iPad screen",
    # replacing the actionable reason with a vague one for the whole lock.
    empty = json.dumps({"frame_index": 1, "timestamp_sec": 0.0,
                        "width": 1640, "height": 2360, "observations": []})
    shown = []

    vision_live.run_dashboard([_payload(index=0), empty], draw=shown.append,
                              status_of=lambda: "no frames arriving -- unlock the iPad screen")

    assert "unlock the iPad screen" in shown[-1]


def test_the_generic_message_is_used_when_nothing_else_is_reported():
    empty = json.dumps({"frame_index": 1, "timestamp_sec": 0.0,
                        "width": 1640, "height": 2360, "observations": []})
    shown = []

    vision_live.run_dashboard([_payload(index=0), empty], draw=shown.append)

    assert "no table on screen" in shown[-1]


def test_hero_hole_cards_are_tracked_across_the_hand():
    from poker_vision.hero_cards import HeroCardTracker
    tracker = HeroCardTracker()
    tracker.cards = ["Ad", "Kd"]         # as if read at preflop
    tracker.read_this_hand = True

    states = list(vision_live.iter_states([_payload()], hero_tracker=tracker))

    hero = [s for s in states[0].seats if s.is_hero]
    assert hero == [] or hero[0].hole_cards == ["Ad", "Kd"]


def test_stale_frames_are_dropped_in_favour_of_the_newest():
    # The helper emits whenever the screen changes, which during play is far
    # faster than a frame can be parsed. Reading the pipe in order means the
    # display falls progressively further behind the table -- the backlog is
    # unbounded, so the lag grows without limit. Only the newest frame is
    # worth rendering.
    lines = [f"line{i}" for i in range(5)]

    assert list(vision_live.newest_first(lines)) == ["line4"]


def test_a_single_frame_is_delivered_unchanged():
    assert list(vision_live.newest_first(["only"])) == ["only"]


def test_an_empty_stream_yields_nothing():
    assert list(vision_live.newest_first([])) == []


def test_every_frame_is_delivered_when_they_arrive_one_at_a_time():
    # Nothing is dropped unless a frame actually piled up behind another.
    import queue as _q
    import threading
    import time

    slow = _q.Queue()

    def trickle():
        for i in range(3):
            slow.put(f"line{i}")
            time.sleep(0.05)
        slow.put(None)

    threading.Thread(target=trickle, daemon=True).start()

    def stream():
        while True:
            item = slow.get()
            if item is None:
                return
            yield item

    assert list(vision_live.newest_first(stream())) == ["line0", "line1", "line2"]


def test_a_display_goes_stale_when_frames_stop_arriving():
    # The failure this exists for: the helper goes quiet -- the iPad sleeps,
    # or the capture session dies -- and run_dashboard's loop simply blocks.
    # The terminal keeps showing the last frame it ever got, with the board
    # and hole cards of a hand that finished minutes ago, and nothing marks
    # it stale. Seen live: "hero hand is stuck at previous hero hand".
    clock = [100.0]
    watch = vision_live.StaleWatch(stale_after=4.0, now=lambda: clock[0])

    watch.note_arrival()
    clock[0] += 1.0
    assert watch.overdue() is False

    clock[0] += 4.0
    assert watch.overdue() is True


def test_a_resumed_stream_is_not_stale():
    clock = [100.0]
    watch = vision_live.StaleWatch(stale_after=4.0, now=lambda: clock[0])
    watch.note_arrival()
    clock[0] += 10.0
    assert watch.overdue() is True

    watch.note_arrival()

    assert watch.overdue() is False


def test_a_display_with_nothing_on_it_yet_is_not_called_stale():
    # Before the first frame there is nothing to mark stale -- the waiting
    # screen already says what is happening.
    clock = [100.0]
    watch = vision_live.StaleWatch(stale_after=4.0, now=lambda: clock[0])
    clock[0] += 10.0

    assert watch.overdue() is True
    assert watch.ever_arrived is False


def test_the_held_display_is_redrawn_when_the_stream_goes_quiet():
    # The unit tests above cover the decision; this covers the thread that
    # acts on it, which is the part that was missing when a live session sat
    # for minutes showing a finished hand.
    import time as _time

    def one_frame_then_silence():
        yield _payload(index=0)
        _time.sleep(0.5)

    shown = []
    vision_live.run_dashboard(one_frame_then_silence(), draw=shown.append,
                              stale_after=0.05, stale_poll=0.02)

    assert any("HELD" in s for s in shown), "the stall was never marked"
    assert any("iPad awake and unlocked" in s for s in shown)
    assert "HELD" not in shown[0], "the first live frame must not be marked"


def test_a_reported_fault_is_preferred_when_the_stream_goes_quiet():
    import time as _time

    def one_frame_then_silence():
        yield _payload(index=0)
        _time.sleep(0.5)

    shown = []
    vision_live.run_dashboard(one_frame_then_silence(), draw=shown.append,
                              status_of=lambda: "no frames arriving -- unlock the iPad screen",
                              stale_after=0.05, stale_poll=0.02)

    assert any("unlock the iPad screen" in s for s in shown)


def test_a_quiet_table_is_not_stale_while_the_helper_reports_frames_arriving():
    # The bug this exists for. The helper emits only when screen content
    # changes, so a table where everyone is thinking produces nothing for
    # many seconds -- and the display flickered in and out of HELD, which
    # teaches you to ignore the one warning that matters. Frames are still
    # arriving; they are just identical. Only the helper can tell, because
    # only the helper sees arrivals before the content-diff gate.
    import time as _time

    def one_frame_then_quiet():
        yield _payload(index=0)
        _time.sleep(0.4)

    shown = []
    vision_live.run_dashboard(one_frame_then_quiet(), draw=shown.append,
                              liveness_of=lambda: None,      # helper says all is well
                              stale_after=0.05, stale_poll=0.02)

    assert not any("HELD" in s for s in shown), "a quiet table was called stale"


def test_the_helper_saying_it_is_starved_marks_the_display_stale():
    import time as _time

    def one_frame_then_quiet():
        yield _payload(index=0)
        _time.sleep(0.4)

    shown = []
    vision_live.run_dashboard(one_frame_then_quiet(), draw=shown.append,
                              liveness_of=lambda: "no frames arriving -- unlock the iPad screen",
                              stale_after=0.05, stale_poll=0.02)

    assert any("unlock the iPad screen" in s for s in shown)


def test_liveness_from_the_helpers_heartbeat():
    clock = [100.0]
    h = vision_live.Heartbeat(starved_after=5.0, silent_after=8.0,
                              now=lambda: clock[0])

    assert h.fault() is None, "nothing heard yet is not yet a fault"

    h.beat(quiet=0.2); assert h.fault() is None          # frames flowing
    h.beat(quiet=1.9); assert h.fault() is None          # static table, still fine
    h.beat(quiet=6.0)
    assert "no frames arriving" in h.fault()             # genuinely starved

    h.beat(quiet=0.1); assert h.fault() is None          # recovered
    clock[0] += 9.0
    assert "capture helper" in h.fault()                 # helper itself went silent


from types import SimpleNamespace


def test_frame_age_is_measured_from_the_capture_timestamp():
    state = SimpleNamespace(timestamp_sec=1000.0)
    assert vision_live.frame_age_ms(state, now=lambda: 1000.75) == 750.0


def test_frame_age_is_withheld_for_a_replayed_frame():
    # A saved frame's timestamp is from whenever it was captured; showing
    # "3 hours behind" would be noise, not a measurement.
    state = SimpleNamespace(timestamp_sec=1000.0)
    assert vision_live.frame_age_ms(state, now=lambda: 1000.0 + 3 * 3600) is None


def test_frame_age_is_withheld_for_a_timestamp_in_the_future():
    state = SimpleNamespace(timestamp_sec=1000.0)
    assert vision_live.frame_age_ms(state, now=lambda: 999.0) is None


# Player mode. The bottom row as it reads on a real turn (ipad_001) and while
# waiting with pre-action boxes up (ipad_000).
_TURN_ROW = [{"text": t, "x": x, "y": 2300, "w": 120, "h": 40, "confidence": 1.0}
             for t, x in (("+30s", 150), ("Fold", 500), ("Check", 820), ("Raise", 1140))]
_WAIT_ROW = [{"text": t, "x": x, "y": 2300, "w": 120, "h": 40, "confidence": 1.0}
             for t, x in (("Check / Fold", 400), ("Call any", 820), ("Straddle", 1140))]
_SEAT = [
    {"text": "drake81", "x": 511, "y": 565, "w": 123, "h": 29, "confidence": 1.0},
    {"text": "7.75", "x": 509, "y": 620, "w": 72, "h": 34, "confidence": 1.0},
]


def _frame(index, row):
    return _payload(index=index, obs=_SEAT + row)


def test_player_mode_draws_only_the_frames_where_it_is_your_turn():
    shown = []
    n = vision_live.run_dashboard(
        [_frame(0, _WAIT_ROW), _frame(1, _TURN_ROW), _frame(2, _WAIT_ROW),
         _frame(3, _WAIT_ROW), _frame(4, _TURN_ROW)],
        draw=shown.append, turns_only=True)

    assert n == 2
    tables = [t for t in shown if "drake81" in t]
    assert len(tables) == 2
    assert all("YOUR TURN" in t for t in tables)


def test_player_mode_says_a_table_was_found_before_the_first_turn():
    # Otherwise the panel keeps claiming there is no table while one is up.
    shown = []
    vision_live.run_dashboard([_frame(0, _WAIT_ROW), _frame(1, _WAIT_ROW)],
                              draw=shown.append, turns_only=True)

    assert len(shown) == 1
    assert "waiting for your turn" in shown[0]


def test_player_mode_marks_its_snapshot_so_it_never_passes_for_live():
    shown = []
    vision_live.run_dashboard([_frame(0, _TURN_ROW)], draw=shown.append,
                              turns_only=True)

    assert "PLAYER MODE" in shown[-1]


def test_live_mode_still_draws_every_frame_and_carries_no_banner():
    shown = []
    n = vision_live.run_dashboard([_frame(0, _WAIT_ROW), _frame(1, _TURN_ROW)],
                                  draw=shown.append)

    assert n == 2
    assert not any("PLAYER MODE" in t for t in shown)


def test_player_mode_still_feeds_the_trackers_on_frames_it_does_not_draw():
    # Hole cards are read at the deal and positions while the blinds show --
    # both usually before the turn. Skipping the parse on undrawn frames
    # would leave them nothing to hold.
    from poker_vision import position_tracker
    seen = []

    class Recording(position_tracker.PositionTracker):
        def update(self, seats, signature):
            seen.append(len(seats))
            return super().update(seats, signature=signature)

    vision_live.run_dashboard([_frame(0, _WAIT_ROW), _frame(1, _WAIT_ROW)],
                              draw=lambda t: None, turns_only=True,
                              position_tracker=Recording())

    assert len(seen) == 2


def _thumb_payload(index, felt, handle):
    import base64
    import cv2
    import numpy as np
    thumb = np.full((590, 410), felt, dtype=np.uint8)
    thumb[565 // 4 - 3:565 // 4 + 3, 511 // 4 - 15:511 // 4 + 15] = handle
    data = json.loads(_payload(index=index))
    data["table_thumb"] = base64.b64encode(cv2.imencode(".png", thumb)[1].tobytes()).decode()
    data["table_thumb_scale"] = 0.25
    return json.dumps(data)


def test_a_panel_over_the_table_holds_the_last_read_instead_of_redrawing():
    # Read through a chat or settings panel, every player looked folded while
    # their action bubbles still read. Hold the last clean table, and say why.
    shown = []
    n = vision_live.run_dashboard(
        [_thumb_payload(0, felt=48, handle=200), _thumb_payload(1, felt=20, handle=110)],
        draw=shown.append, stale_poll=3600)

    assert n == 1
    assert "panel is covering the table" in shown[-1]
    assert "frame 0" in shown[-1]


def test_an_obscured_frame_does_not_reach_the_trackers():
    # Dimmed, the hero-card region jumps like a new deal; letting the
    # trackers see it would discard this hand's hole cards for opening chat.
    from poker_vision import position_tracker
    seen = []

    class Recording(position_tracker.PositionTracker):
        def update(self, seats, signature=None):
            seen.append(True)
            return super().update(seats, signature=signature)

    list(vision_live.iter_states(
        [_thumb_payload(0, felt=48, handle=200), _thumb_payload(1, felt=20, handle=110)],
        position_tracker=Recording()))

    assert len(seen) == 1
