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
