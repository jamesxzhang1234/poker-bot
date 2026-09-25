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
