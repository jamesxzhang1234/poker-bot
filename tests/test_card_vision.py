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
