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
    assert "Board: 7c 3d 6c" in out


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
    assert "Ah Kd" in out
