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

    # The same plate coordinates on an iPhone-sized frame are nowhere near
    # that layout's top_left anchor, so they must not resolve to it.
    iphone = list(vision_live.iter_states([_payload(width=1206, height=2622)]))[0]
    assert iphone.seats == []


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


def test_layout_lookup_covers_the_iphone_too():
    layout = vision_live.layout_for(*seat_anchors.IPHONE_PORTRAIT_SIZE)

    assert layout.anchors is seat_anchors.IPHONE_PORTRAIT
    assert len(layout.board_slots) == 5


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
    for i, (rank, bgr) in enumerate([("K", (160, 60, 40)), ("9", (50, 150, 50))]):
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

    assert state.board_cards == ["Kd", "9c"]


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
