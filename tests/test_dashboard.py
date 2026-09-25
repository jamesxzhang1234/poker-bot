"""Tests for the in-place live dashboard.

Rendering is pure -- a TableState goes in, a block of text comes out -- so
none of this needs a terminal.
"""
from poker_vision.dashboard import render
from poker_vision.table_state import Seat, TableState


def _state(seats, pot=None, board=None, frame_index=0):
    return TableState(frame_index=frame_index, timestamp_sec=0.0,
                      seats=seats, board_cards=board or [], pot=pot)


def test_header_shows_the_pot_and_board():
    out = render(_state([], pot=3.4, board=["3c", "6c", "6d", "Tc"]))

    assert "3.4" in out
    assert "3♣ 6♣ 6♦ T♣" in out


def test_seats_are_ordered_by_poker_position():
    seats = [
        Seat(seat_id="a", name="cutoff", stack=1.0, position="CO"),
        Seat(seat_id="b", name="dealer", stack=2.0, position="D"),
        Seat(seat_id="c", name="bigblind", stack=3.0, position="BB"),
    ]

    lines = [l for l in render(_state(seats)).splitlines() if "dealer" in l
             or "bigblind" in l or "cutoff" in l]
    order = [next(n for n in ("dealer", "bigblind", "cutoff") if n in l)
             for l in lines]

    assert order == ["dealer", "bigblind", "cutoff"]


def test_an_action_shows_the_amount_committed():
    seats = [Seat(seat_id="a", name="Earocker", stack=17.6,
                  position="HJ", status="Raise", bet=1.7)]

    assert "(Raise : 1.7)" in render(_state(seats))


def test_an_action_with_no_chips_committed_shows_no_amount():
    seats = [Seat(seat_id="a", name="gelato", stack=3.65,
                  position="BB", status="Check")]

    out = render(_state(seats))
    assert "(Check)" in out
    assert ":" not in out.split("gelato")[1].split("\n")[0]


def test_values_that_changed_since_the_last_frame_are_marked():
    before = _state([Seat(seat_id="a", name="jxz", stack=9.9, position="SB")])
    after = _state([Seat(seat_id="a", name="jxz", stack=9.5, position="SB")])

    line = [l for l in render(after, previous=before).splitlines() if "jxz" in l][0]

    assert "*" in line


def test_an_unchanged_value_is_not_marked():
    before = _state([Seat(seat_id="a", name="jxz", stack=9.9, position="SB")])
    after = _state([Seat(seat_id="a", name="jxz", stack=9.9, position="SB")])

    line = [l for l in render(after, previous=before).splitlines() if "jxz" in l][0]

    assert "*" not in line


def test_no_flag_when_the_bets_add_up():
    seats = [Seat(seat_id="a", name="x", stack=1.0, position="D", bet=0.1),
             Seat(seat_id="b", name="y", stack=1.0, position="SB", bet=0.05)]

    out = render(_state(seats, pot=0.15))
    assert "!" not in out


def test_a_seat_with_an_unknown_stack_renders_without_crashing():
    seats = [Seat(seat_id="a", name="trill04", stack=None, position="BB")]

    out = render(_state(seats))
    assert "trill04" in out
    assert "?" in out


def test_waiting_panel_explains_why_there_is_no_feed():
    from poker_vision.dashboard import render_waiting

    out = render_waiting("iOS device is attached but offers no capture interface")

    assert "waiting" in out.lower()
    assert "capture interface" in out


def test_change_marks_survive_the_handle_being_read_differently():
    # Vision returns "jxz" and "jXZ" for the same player between frames.
    # Keying the comparison on the handle made a case flip look like a new
    # player, so nothing was ever marked as changed for that seat.
    before = _state([Seat(seat_id="hero", name="jxz", stack=9.9, position="SB")])
    after = _state([Seat(seat_id="hero", name="jXZ", stack=9.5, position="SB")])

    line = [l for l in render(after, previous=before).splitlines() if "jXZ" in l][0]

    assert "*" in line


def test_a_five_character_position_does_not_break_the_columns():
    seats = [Seat(seat_id="a", name="drake81", stack=5.35, position="UTG+1"),
             Seat(seat_id="b", name="trill04", stack=17.2, position="D")]

    lines = [l for l in render(_state(seats)).splitlines()
             if "drake81" in l or "trill04" in l]
    # the handle must start at the same column on both rows
    assert lines[0].index("trill04") == lines[1].index("drake81")


def test_bets_below_the_pot_are_normal_and_not_flagged():
    # The pot is cumulative across streets while the chips on the felt are
    # only the current one, so on any street after the first the bets are
    # SUPPOSED to total less than the pot. Flagging that would fire on nearly
    # every post-flop frame.
    seats = [Seat(seat_id="a", name="x", stack=1.0, position="D", bet=0.3),
             Seat(seat_id="b", name="y", stack=1.0, position="SB", bet=0.3)]

    assert "!" not in render(_state(seats, pot=10.4))


def test_bets_exceeding_the_pot_are_flagged():
    # This direction is impossible: the pot already contains this street's
    # chips, so bets totalling more than it means something was misread.
    seats = [Seat(seat_id="a", name="x", stack=1.0, position="D", bet=9.0),
             Seat(seat_id="b", name="y", stack=1.0, position="SB", bet=0.05)]

    assert "!" in render(_state(seats, pot=0.15))


def test_chips_landing_before_the_pot_updates_is_not_flagged():
    # During the chip animation the felt shows a bet the pot total has not
    # absorbed yet, so bets briefly exceed the pot by about one blind. Seen on
    # 2 of 30 real frames -- a display lag, not a misread, and flagging it
    # would put a warning on normal play.
    seats = [Seat(seat_id="a", name="x", stack=1.0, position="D", bet=0.05),
             Seat(seat_id="b", name="y", stack=1.0, position="SB", bet=0.1),
             Seat(seat_id="c", name="z", stack=1.0, position="BB", bet=0.05)]

    assert "!" not in render(_state(seats, pot=0.15))


def test_a_held_display_says_so_rather_than_looking_live():
    # When the iPad locks, no frames arrive at all and the last table stays on
    # screen. Without a marker it looks like live data, which is the worst way
    # for this to fail -- you would act on a stale table.
    seats = [Seat(seat_id="hero", name="jxz", stack=9.9, position="SB")]

    out = render(_state(seats, pot=1.0), status="lost the device")

    assert "lost the device" in out
    assert "held" in out.lower()


def test_no_held_marker_while_frames_are_arriving():
    seats = [Seat(seat_id="hero", name="jxz", stack=9.9, position="SB")]

    assert "held" not in render(_state(seats, pot=1.0)).lower()


def test_waiting_panel_asks_for_a_table_once_the_device_is_connected():
    # Sitting in the PurePoker lobby is not a fault. Telling the user to
    # unlock a device that is plainly already captured sends them looking for
    # a problem that isn't there -- what they need to do is open a table.
    from poker_vision.dashboard import render_waiting

    out = render_waiting("capturing: James's iPad", connected=True)

    assert "table" in out.lower()
    assert "unlock" not in out.lower()


def test_waiting_panel_explains_the_setup_when_not_yet_connected():
    from poker_vision.dashboard import render_waiting

    out = render_waiting("waiting: no iOS device attached over USB")

    assert "unlock" in out.lower()


def test_the_hero_line_shows_your_hole_cards():
    seats = [Seat(seat_id="hero", name="jxz", stack=9.9, position="SB",
                  is_hero=True, hole_cards=["Ad", "Kd"])]

    line = [l for l in render(_state(seats)).splitlines() if "jxz" in l][0]

    assert "A♦ K♦" in line


def test_a_seat_with_no_known_cards_shows_none():
    seats = [Seat(seat_id="hero", name="jxz", stack=9.9, position="SB", is_hero=True)]

    line = [l for l in render(_state(seats)).splitlines() if "jxz" in l][0]

    assert "♦" not in line and "[" not in line


def test_your_turn_is_called_out_with_the_amount_to_call():
    # The one thing you most need to see at a glance, and the trigger a
    # decision engine would fire on.
    seats = [Seat(seat_id="hero", name="jxz", stack=9.9, position="SB", is_hero=True)]
    state = _state(seats, pot=1.7)
    state.your_turn = True
    state.to_call = 0.3

    out = render(state)
    assert "YOUR TURN" in out
    assert "0.3" in out


def test_nothing_is_called_out_when_it_is_not_your_turn():
    seats = [Seat(seat_id="hero", name="jxz", stack=9.9, position="SB", is_hero=True)]

    assert "YOUR TURN" not in render(_state(seats, pot=1.7))


def test_folded_seats_are_visibly_out_of_the_hand():
    seats = [Seat(seat_id="a", name="live", stack=1.0, position="D", in_hand=True),
             Seat(seat_id="b", name="gone", stack=2.0, position="SB", in_hand=False)]

    rendered = render(_state(seats)).splitlines()
    live_line = next(l for l in rendered if "live" in l)
    gone_line = next(l for l in rendered if "gone" in l)

    # the folded seat carries a marker the live one does not
    assert gone_line.lstrip().startswith(".")
    assert not live_line.lstrip().startswith(".")


def test_the_count_of_players_still_in_the_hand_is_shown():
    seats = [Seat(seat_id="a", name="x", stack=1.0, position="D", in_hand=True),
             Seat(seat_id="b", name="y", stack=1.0, position="SB", in_hand=True),
             Seat(seat_id="c", name="z", stack=1.0, position="BB", in_hand=False)]

    assert "2 in hand" in render(_state(seats))
