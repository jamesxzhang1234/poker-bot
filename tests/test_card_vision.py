"""Tests for suit classification from a card's background colour.

Values here are real measurements sampled from 163 card backgrounds across
both devices, not invented colours.
"""
from poker_vision import card_vision


def test_classifies_each_suit_from_a_real_background_colour():
    # (b, g, r) medians measured per suit per device.
    for bgr, suit in [
        ((45, 28, 20), "s"),     # iPad navy
        ((41, 25, 18), "s"),     # iPhone navy
        ((152, 74, 48), "d"),    # iPad blue
        ((167, 86, 64), "d"),    # iPhone blue
        ((60, 137, 26), "c"),    # iPad green
        ((55, 136, 28), "c"),    # iPhone green
        ((40, 58, 192), "h"),    # iPad red
        ((38, 59, 191), "h"),    # iPhone red
    ]:
        assert card_vision.classify_suit(bgr) == suit, bgr


def test_the_brightest_ipad_spade_is_still_a_spade():
    # This is the bug: the old rule called anything with b+g+r below 110 a
    # spade, and iPad spades measured 92-107 -- a 3-point margin. The SAME
    # card read 107 in one frame and 116 in the next, so it came out "4s"
    # then "4d". Spades top out at max-channel 55 while the next suit up
    # starts at 110, so that statistic has room to spare.
    assert card_vision.classify_suit((55, 34, 23)) == "s"
    assert card_vision.classify_suit((59, 34, 23)) == "s"


def test_the_dimmest_non_spade_is_not_a_spade():
    # Dimmest real club measured max-channel 110.
    assert card_vision.classify_suit((44, 110, 22)) == "c"


def test_a_fanned_cards_suit_comes_from_its_exposed_edge():
    # The hero's two cards overlap, so the card behind has its centre covered
    # by the card in front. Sampling the centre picked up the neighbour and
    # called a 9 of spades a diamond; the exposed left edge is still its own
    # face. Colours here are real measurements from frames/ipad/ipad_000.png.
    import numpy as np
    card = np.zeros((170, 115, 3), dtype=np.uint8)
    card[:, :] = (116, 100, 95)           # centre: washed out by the card in front
    card[55:130, 5:40] = (54, 40, 37)     # exposed left edge: navy

    assert card_vision.classify_suit(card_vision.sample_face_color(card)) == "s"


def test_an_unobstructed_card_reads_the_same_either_way():
    import numpy as np
    card = np.full((170, 115, 3), (160, 60, 40), dtype=np.uint8)   # blue: diamonds

    assert card_vision.classify_suit(card_vision.sample_face_color(card)) == "d"
