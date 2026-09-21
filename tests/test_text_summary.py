from poker_vision.text_summary import format_summary


def test_format_summary_includes_pot_and_board():
    state = {
        "frame_index": 3,
        "timestamp_sec": 12.5,
        "pot": 45.0,
        "board_cards": ["7c", "3d", "6c"],
        "seats": [],
    }
    out = format_summary(state)
    assert "frame 3" in out
    assert "Pot: 45.0" in out
    assert "Board: 7♣ 3♦ 6♣" in out


def test_format_summary_lists_seats_with_details():
    state = {
        "frame_index": 0,
        "timestamp_sec": 0.0,
        "pot": None,
        "board_cards": [],
        "seats": [
            {
                "seat_id": "top_left",
                "name": "Balboni",
                "stack": 17.2,
                "position": "SB",
                "status": "Folded",
                "in_hand": False,
                "is_hero": False,
                "hole_cards": [],
            },
            {
                "seat_id": "hero",
                "name": None,
                "stack": 100.0,
                "position": "BB",
                "status": None,
                "in_hand": True,
                "is_hero": True,
                "hole_cards": ["Ah", "Kd"],
            },
        ],
    }
    out = format_summary(state)
    assert "Pot: ?" in out
    assert "Board: (none)" in out
    assert "Balboni" in out
    assert "[SB]" in out
    assert "$17.2" in out
    assert "(Folded)" in out
    assert "HERO" in out
    assert "A♥ K♦" in out


def test_format_summary_maps_all_four_suit_letters_to_symbols():
    state = {
        "frame_index": 0,
        "timestamp_sec": 0.0,
        "pot": None,
        "board_cards": ["2c", "3d", "4h", "5s"],
        "seats": [],
    }
    out = format_summary(state)
    assert "Board: 2♣ 3♦ 4♥ 5♠" in out


def test_format_summary_leaves_unrecognized_card_code_untouched():
    # Defensive: an unexpected code (e.g. a future rank/suit this map
    # doesn't know) should pass through as-is rather than crash or mangle.
    state = {
        "frame_index": 0,
        "timestamp_sec": 0.0,
        "pot": None,
        "board_cards": ["??"],
        "seats": [],
    }
    out = format_summary(state)
    assert "Board: ??" in out


def _seat(seat_id, position):
    return {
        "seat_id": seat_id,
        "name": seat_id,
        "stack": 10.0,
        "position": position,
        "status": None,
        "in_hand": True,
        "is_hero": False,
        "hole_cards": [],
    }


def test_format_summary_orders_seats_by_poker_position_not_input_order():
    state = {
        "frame_index": 0,
        "timestamp_sec": 0.0,
        "pot": None,
        "board_cards": [],
        "seats": [
            _seat("bottom_left", "CO"),
            _seat("mid_right", "BB"),
            _seat("top_left", "D"),
            _seat("hero", None),
            _seat("lower_mid_right", "SB"),
            _seat("top_right", "UTG"),
        ],
    }
    out = format_summary(state)
    lines = [line.strip() for line in out.splitlines() if line.strip()]
    seat_order = [line.split()[0] for line in lines if line.split()[0] in
                  ("top_left", "mid_right", "bottom_left", "hero", "lower_mid_right", "top_right")]
    assert seat_order == ["top_left", "lower_mid_right", "mid_right", "top_right", "bottom_left", "hero"]
