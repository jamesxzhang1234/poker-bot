from poker_vision.table_state import is_plausible_table


def _state_dict_with_stacks(n_stacks, n_total_seats=9):
    seats = [{"stack": 10.0 if i < n_stacks else None} for i in range(n_total_seats)]
    return {"frame_index": 0, "timestamp_sec": 0.0, "pot": None, "board_cards": [], "seats": seats}


def test_is_plausible_table_true_with_four_seats_showing_a_stack():
    state = _state_dict_with_stacks(4)
    assert is_plausible_table(state) is True


def test_is_plausible_table_false_with_three_seats_showing_a_stack():
    state = _state_dict_with_stacks(3)
    assert is_plausible_table(state) is False


def test_is_plausible_table_false_with_no_seats_showing_a_stack():
    state = _state_dict_with_stacks(0)
    assert is_plausible_table(state) is False
