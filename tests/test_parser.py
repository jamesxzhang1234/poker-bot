from poker_vision.parser import infer_positions
from poker_vision.table_state import Seat


def _seat(seat_id, stack=None, position=None):
    return Seat(seat_id=seat_id, stack=stack, position=position)


def test_infers_sb_and_bb_from_known_dealer():
    # SEAT_ORDER action-order direction: top_right -> mid_right -> lower_mid_right -> ...
    seats = [
        _seat("top_right", stack=10.0, position="D"),
        _seat("mid_right", stack=10.0),
        _seat("lower_mid_right", stack=10.0),
    ]
    result = infer_positions(seats)
    by_id = {s.seat_id: s for s in result}
    assert by_id["top_right"].position == "D"
    assert by_id["mid_right"].position == "SB"
    assert by_id["lower_mid_right"].position == "BB"


def test_infers_dealer_and_sb_from_known_big_blind():
    seats = [
        _seat("top_right", stack=10.0),
        _seat("mid_right", stack=10.0),
        _seat("lower_mid_right", stack=10.0, position="BB"),
    ]
    result = infer_positions(seats)
    by_id = {s.seat_id: s for s in result}
    assert by_id["top_right"].position == "D"
    assert by_id["mid_right"].position == "SB"
    assert by_id["lower_mid_right"].position == "BB"


def test_skips_unoccupied_seats_in_rotation():
    # bottom_right (between lower_mid_right and hero in SEAT_ORDER) is empty,
    # so SB/BB should land on the next *occupied* seats, not the empty one.
    seats = [
        _seat("top_right", stack=10.0, position="D"),
        _seat("mid_right"),  # unoccupied: no stack, no name -> skipped
        _seat("lower_mid_right", stack=10.0),
        _seat("bottom_right"),  # unoccupied: skipped
        _seat("hero", stack=10.0),
    ]
    result = infer_positions(seats)
    by_id = {s.seat_id: s for s in result}
    assert by_id["lower_mid_right"].position == "SB"
    assert by_id["hero"].position == "BB"
    assert by_id["mid_right"].position is None
    assert by_id["bottom_right"].position is None


def test_never_overwrites_an_already_read_position():
    # mid_right's own badge OCR (rightly or wrongly) already read "BB";
    # inference must not clobber it even though rotation would say "SB".
    seats = [
        _seat("top_right", stack=10.0, position="D"),
        _seat("mid_right", stack=10.0, position="BB"),
        _seat("lower_mid_right", stack=10.0),
    ]
    result = infer_positions(seats)
    by_id = {s.seat_id: s for s in result}
    assert by_id["mid_right"].position == "BB"


def test_no_anchor_leaves_all_positions_unset():
    seats = [
        _seat("top_right", stack=10.0),
        _seat("mid_right", stack=10.0),
        _seat("lower_mid_right", stack=10.0),
    ]
    result = infer_positions(seats)
    assert all(s.position is None for s in result)


def test_fewer_than_two_occupied_seats_does_nothing():
    seats = [_seat("top_right", stack=10.0, position="D")]
    result = infer_positions(seats)
    assert result[0].position == "D"


def test_assigns_standard_position_names_for_a_six_seat_table():
    # SEAT_ORDER action-order: top_right, mid_right, lower_mid_right,
    # bottom_right, hero, bottom_left, lower_mid_left, mid_left, top_left.
    # 6 occupied seats -> standard 6-max naming: BTN(D), SB, BB, UTG, HJ, CO.
    seats = [
        _seat("top_right", stack=10.0, position="D"),
        _seat("mid_right", stack=10.0),
        _seat("lower_mid_right", stack=10.0),
        _seat("bottom_right", stack=10.0),
        _seat("hero", stack=10.0),
        _seat("bottom_left", stack=10.0),
    ]
    result = infer_positions(seats)
    by_id = {s.seat_id: s for s in result}
    assert by_id["top_right"].position == "D"
    assert by_id["mid_right"].position == "SB"
    assert by_id["lower_mid_right"].position == "BB"
    assert by_id["bottom_right"].position == "UTG"
    assert by_id["hero"].position == "HJ"
    assert by_id["bottom_left"].position == "CO"


def test_assigns_standard_position_names_for_a_nine_seat_table():
    seats = [
        _seat("top_right", stack=10.0, position="D"),
        _seat("mid_right", stack=10.0),
        _seat("lower_mid_right", stack=10.0),
        _seat("bottom_right", stack=10.0),
        _seat("hero", stack=10.0),
        _seat("bottom_left", stack=10.0),
        _seat("lower_mid_left", stack=10.0),
        _seat("mid_left", stack=10.0),
        _seat("top_left", stack=10.0),
    ]
    result = infer_positions(seats)
    by_id = {s.seat_id: s for s in result}
    assert by_id["top_right"].position == "D"
    assert by_id["mid_right"].position == "SB"
    assert by_id["lower_mid_right"].position == "BB"
    assert by_id["bottom_right"].position == "UTG"
    assert by_id["hero"].position == "UTG+1"
    assert by_id["bottom_left"].position == "UTG+2"
    assert by_id["lower_mid_left"].position == "LJ"
    assert by_id["mid_left"].position == "HJ"
    assert by_id["top_left"].position == "CO"


def test_never_assigns_a_role_another_seat_already_holds():
    # Rotation math would put BB on "hero" here, but "bottom_left" already
    # holds BB from its own OCR read -- the model doesn't fit this hand's
    # occupied-seat set, so the conflicting inferred assignment must be
    # dropped rather than producing two seats both labeled BB.
    seats = [
        _seat("top_left", stack=10.0, position="D"),
        _seat("lower_mid_right", stack=10.0, position="SB"),
        _seat("bottom_left", stack=10.0, position="BB"),
        _seat("hero", stack=10.0),
    ]
    result = infer_positions(seats)
    by_id = {s.seat_id: s for s in result}
    assert by_id["hero"].position is None
    assert by_id["bottom_left"].position == "BB"
