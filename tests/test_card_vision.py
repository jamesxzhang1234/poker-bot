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


def test_a_dimmed_hero_card_keeps_its_suit():
    # When a hand ends the client dims the cards. Absolute brightness then
    # drops a blue diamond into the navy range and it reads as a spade --
    # measured on a real frame, (60,40,23) for a 5 of diamonds. Channel
    # ratios survive dimming: that card's b/r is 2.62, right alongside the
    # bright one's 2.84, while real navy sits at 1.47-1.54.
    assert card_vision.classify_hero_suit((60, 40, 23)) == "d"     # dimmed diamond
    assert card_vision.classify_hero_suit((148, 98, 52)) == "d"    # bright diamond
    assert card_vision.classify_hero_suit((29, 20, 19)) == "s"     # dimmed spade
    assert card_vision.classify_hero_suit((54, 40, 37)) == "s"     # bright spade


def test_hero_suits_are_told_apart_by_which_channel_leads():
    assert card_vision.classify_hero_suit((47, 44, 80)) == "h"     # red leads
    assert card_vision.classify_hero_suit((34, 55, 31)) == "c"     # green leads


def test_a_ten_is_recognised_from_its_two_character_glyph():
    # The client renders a ten as "10". The old code stripped the zero and
    # was left with "1", which is not a rank, so it returned nothing and the
    # psm cascade kept guessing until it landed on something valid -- a ten
    # came back as an ace. It is also why read_rank never once returned T
    # across every board card in the corpus.
    assert card_vision._extract_rank("10") == "T"
    assert card_vision._extract_rank("1O") == "T"
    assert card_vision._extract_rank(" 10 ") == "T"


def test_a_plain_rank_is_unaffected():
    for text, rank in [("A", "A"), ("K", "K"), ("7", "7"), ("T", "T")]:
        assert card_vision._extract_rank(text) == rank


def test_an_unreadable_string_is_still_rejected():
    for text in ["", "  ", "xyz", "%"]:
        assert card_vision._extract_rank(text) is None


def test_a_two_character_read_ending_in_zero_is_a_ten():
    # A ten is the only rank drawn with two characters, and the cascade never
    # sees it: the smaller windows crop the "10" and return a confident wrong
    # single letter before a larger window can show the whole thing. Read
    # without a whitelist at a large window, the real ten came back as "40" --
    # the 1 misread as 4 -- while all 17 other cards in the corpus gave
    # garbage that does not end in a zero.
    assert card_vision.looks_like_ten("40") is True
    assert card_vision.looks_like_ten("10") is True
    assert card_vision.looks_like_ten("1O") is True


def test_other_rank_misreads_are_not_taken_for_a_ten():
    for text in ["eo,", "a,", "e)", "*)", "an", "+", "Fi", "Q", "", ":", "A", "0"]:
        assert card_vision.looks_like_ten(text) is False, text


def _fanned_card(rank, bgr=(160, 60, 40)):
    """A card whose suit pips dwarf its rank glyph, as the real ones do."""
    import cv2
    import numpy as np
    from poker_vision import rank_templates
    card = np.full((170, 115, 3), (12, 10, 10), dtype=np.uint8)
    face = np.full((150, 100, 3), bgr, dtype=np.uint8)
    glyph = cv2.resize(rank_templates.load()[rank][0], (24, 32),
                       interpolation=cv2.INTER_NEAREST)
    face[8:40, 6:30][glyph > 127] = (255, 255, 255)
    cv2.circle(face, (26, 70), 18, (255, 255, 255), -1)
    cv2.circle(face, (58, 110), 26, (255, 255, 255), -1)
    card[8:158, 8:108] = face
    return card


def test_the_rank_is_read_past_much_larger_suit_pips():
    # Handing Tesseract a window of the card's top gives it the rank plus the
    # pips plus a lot of empty space, and it reads the pips as characters. On
    # a real frame a 5 came back as a 3, because the window held a diamond pip
    # far bigger than the glyph; this fixture reproduces that, reading 2.
    for rank in ["5", "7", "9", "Q"]:
        assert card_vision.read_rank_from_top(_fanned_card(rank)) == rank, rank


def test_the_rank_is_found_on_a_dark_card_too():
    # Face colours run from navy through green to dimmed blue, so the glyph
    # cannot be located with one fixed brightness cutoff.
    assert card_vision.read_rank_from_top(_fanned_card("5", bgr=(54, 40, 37))) == "5"


def test_a_card_slot_overhanging_onto_felt_still_reads_its_suit():
    # The board slots are wider than the cards, so the rightmost slot can
    # extend past its card onto the blue table. Sampling the top-right corner
    # then measures felt, not card -- on a real frame that turned the ace of
    # spades into a diamond. Taking the median of the card face ignores a
    # minority of felt the same way it ignores the white pips.
    import cv2
    import numpy as np
    card = np.zeros((154, 118, 3), dtype=np.uint8)
    card[:, 0:100] = (45, 30, 25)          # navy card face
    card[:, 100:] = (86, 42, 18)           # blue felt the slot overhangs onto
    cv2.circle(card, (50, 100), 30, (255, 255, 255), -1)   # white pip

    assert card_vision.classify_suit(card_vision.sample_card_color(card)) == "s"


def test_white_ink_does_not_drag_the_sampled_colour():
    import cv2
    import numpy as np
    card = np.full((154, 118, 3), (160, 60, 40), dtype=np.uint8)
    cv2.circle(card, (59, 90), 45, (255, 255, 255), -1)

    assert card_vision.classify_suit(card_vision.sample_card_color(card)) == "d"
