"""Tests for holding seat positions across a hand.

The client draws the SB/BB badges pre-flop and takes them away once the
betting moves on -- measured across a live corpus, 93 of 143 pre-flop frames
carry a badge against 19 of 98 post-flop ones. Positions do not change within
a hand, so the ones read while the badges are up stay true after they go.
"""
from poker_vision.position_tracker import PositionTracker
from poker_vision.table_state import Seat


def _seats(**positions):
    """Six occupied seats, with `positions` naming the ones that have a badge."""
    return [Seat(seat_id=sid, name=sid.upper(), stack=10.0,
                 position=positions.get(sid))
            for sid in ("s0", "s1", "s2", "s3", "s4", "s5")]


def test_positions_read_this_frame_are_passed_straight_through():
    seats = _seats(s0="D", s1="SB", s2="BB")

    PositionTracker().update(seats, signature=[0])

    assert [s.position for s in seats] == ["D", "SB", "BB", None, None, None]


def test_positions_survive_the_badges_being_taken_away():
    # The whole point: the flop arrives, the badges vanish, and the player in
    # the big blind is still in the big blind.
    tracker = PositionTracker()
    tracker.update(_seats(s0="D", s1="SB", s2="BB"), signature=[0])

    later = _seats()
    tracker.update(later, signature=[0])

    assert [s.position for s in later] == ["D", "SB", "BB", None, None, None]


def test_a_new_hand_throws_the_old_positions_away():
    # Positions rotate every hand. Holding them across the boundary would put
    # the button on the wrong player with nothing to suggest it -- worse than
    # reporting none, because it looks like an answer.
    tracker = PositionTracker()
    tracker.update(_seats(s0="D", s1="SB", s2="BB"), signature=[0])

    dealt = _seats()
    tracker.update(dealt, signature=[9999])

    assert [s.position for s in dealt] == [None] * 6


def test_a_badge_in_the_current_frame_wins_over_the_held_one():
    tracker = PositionTracker()
    tracker.update(_seats(s0="D", s1="SB", s2="BB"), signature=[0])

    moved = _seats(s1="BB")
    tracker.update(moved, signature=[0])

    assert moved[1].position == "BB"


def test_a_seat_that_never_had_a_position_is_left_alone():
    tracker = PositionTracker()
    tracker.update(_seats(s1="SB"), signature=[0])

    later = _seats()
    tracker.update(later, signature=[0])

    assert later[3].position is None
    assert later[1].position == "SB"


def test_an_empty_seat_is_not_given_a_position():
    # A player can leave mid-hand. Their seat keeps its id, and reviving the
    # position onto an empty plate would invent a player.
    tracker = PositionTracker()
    tracker.update(_seats(s0="D", s1="SB", s2="BB"), signature=[0])

    later = _seats()
    later[0].name, later[0].stack = None, None
    tracker.update(later, signature=[0])

    assert later[0].position is None
    assert later[1].position == "SB"


def test_nothing_is_held_when_there_are_no_hero_pixels_to_detect_a_deal_with():
    # Without a signature there is no way to tell one hand from the next, and
    # holding positions blind would eventually put them on the wrong players.
    tracker = PositionTracker()
    tracker.update(_seats(s0="D", s1="SB", s2="BB"), signature=None)

    later = _seats()
    tracker.update(later, signature=None)

    assert [s.position for s in later] == [None] * 6
