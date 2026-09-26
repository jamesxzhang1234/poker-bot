"""Tests for suit classification from a card's background colour.

Values here are real measurements sampled from 163 card backgrounds across
both devices, not invented colours.
"""
import cv2
import numpy as np

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


def test_a_fanned_rank_is_read_by_ocr_not_templates():
    # Measured, not assumed: on fanned cards the templates score 0.62-0.66
    # with a 0.01-0.03 margin, so right and wrong answers are
    # indistinguishable. Tesseract reads 20/21 of the hands checked by eye
    # against the templates' 16.
    assert card_vision.read_rank_from_top(_fanned_card("8")) == "8"


def test_tesseract_still_covers_a_glyph_the_templates_do_not_know():
    # Templates only know the exemplars harvested so far, so the OCR path has
    # to stay behind them rather than be replaced.
    import numpy as np
    assert card_vision._read_isolated(np.zeros((30, 24), dtype=np.uint8)) is None


def test_a_confident_fanned_template_match_is_taken_over_ocr():
    # Harvested from live play: on fanned cards the templates' correct answers
    # score 0.90-1.00 while their mistakes score 0.40-0.53, so a high bar
    # separates them. Below the bar Tesseract decides, which is why the two
    # together read every hand checked by eye where neither does alone.
    import cv2
    import numpy as np
    from poker_vision import rank_templates

    # Built from scratch rather than on top of _fanned_card, which draws its
    # own glyph -- pasting over it leaves two overlapping eights and the
    # isolator returns the mess.
    fanned = rank_templates.load_fanned()
    card = np.full((170, 115, 3), (160, 60, 40), dtype=np.uint8)
    glyph = cv2.resize(fanned["8"][0], (24, 32), interpolation=cv2.INTER_NEAREST)
    card[10:42, 8:32][glyph > 127] = (255, 255, 255)
    cv2.circle(card, (26, 70), 18, (255, 255, 255), -1)
    cv2.circle(card, (58, 110), 26, (255, 255, 255), -1)

    assert card_vision.read_rank_from_top(card) == "8"


def _plain_card(bgr=(160, 60, 40)):
    card = np.full((170, 115, 3), bgr, dtype=np.uint8)
    cv2.circle(card, (58, 110), 26, (255, 255, 255), -1)      # suit pip
    return card


def test_a_single_character_rank_has_no_second_half():
    # isolate_rank_pair is offered alongside isolate_rank_glyph, so it must
    # stay silent on the twelve ranks drawn as one character. The suit pip is
    # the thing most likely to be mistaken for a second half; it sits below
    # the rank rather than beside it, so the baseline test rejects it.
    card = _plain_card()
    cv2.putText(card, "7", (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    assert card_vision.isolate_rank_pair(card) is None


def test_the_two_halves_of_a_ten_are_returned_together():
    card = _plain_card()
    cv2.putText(card, "10", (6, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    one = card_vision.isolate_rank_glyph(card)
    both = card_vision.isolate_rank_pair(card)

    assert both is not None
    assert both.shape[1] > one.shape[1], "the pair should be wider than the 1 alone"


def test_a_distant_neighbouring_glyph_is_not_paired():
    # On a fanned hand the next card's rank is the only other glyph nearby.
    # It sits far to the right, well beyond a ten's inter-character gap.
    card = _plain_card()
    cv2.putText(card, "7", (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    cv2.putText(card, "4", (66, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    assert card_vision.isolate_rank_pair(card) is None


def test_a_dimmed_board_card_keeps_its_suit():
    # At showdown the client greys out the cards not in the winning hand.
    # Dimming drops a club's green from ~110 to 79, and the board rule called
    # anything under 80 a spade -- so a real showdown board read "As 7c Jc 6s
    # 5s" when it was Ac 7c Jc 6s 8c. Brightness cannot separate a dimmed
    # card from navy; the channel ratios can, and they survive dimming.
    #
    # Measured on 377 board-card samples: navy tops out at b/r 2.38 and the
    # dimmest real diamond starts at 4.00.
    for bgr, suit in [
        ((29, 79, 25), "c"),     # dimmed club   -- was read "s"
        ((21, 52, 19), "c"),     # dimmed club, darker still
        ((78, 44, 17), "d"),     # dimmed diamond -- was read "s"
        ((77, 45, 18), "d"),
        ((25, 23, 87), "h"),     # dimmed heart
        ((40, 25, 21), "s"),     # a real spade in the same brightness band
    ]:
        assert card_vision.classify_suit(bgr) == suit, bgr


def test_a_card_back_is_not_read_as_a_rank():
    # During the deal the hero's cards arrive face down, and the back's
    # centred logo was being isolated as though it were a rank glyph and
    # then read -- live, a back came back as "4s" and HeroCardTracker latched
    # it for the whole hand, because the deal happens at preflop which is
    # exactly when the tracker is allowed to read.
    #
    # The rank is drawn in one fixed font, so its glyph is a consistent
    # fraction of the card: measured over 558 real hero glyphs, every one is
    # 12-22% of the crop's height, while the backs land at 24-38%.
    # Sized to the measured article: the real back's logo isolated at 31% of
    # the crop's height, which the previous bound (85% of the search window,
    # i.e. 47% of the crop) let straight through.
    card = np.full((170, 115, 3), (60, 45, 35), dtype=np.uint8)
    cv2.circle(card, (57, 46), 26, (235, 235, 235), -1)   # the back's logo

    assert card_vision.isolate_rank_glyph(card) is None


def test_a_normal_rank_glyph_is_still_isolated():
    # The counterpart: the bound above must not reject a real rank.
    card = np.full((170, 115, 3), (160, 60, 40), dtype=np.uint8)
    cv2.putText(card, "8", (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)
    cv2.circle(card, (58, 115), 26, (255, 255, 255), -1)

    glyph = card_vision.isolate_rank_glyph(card)

    assert glyph is not None
    assert glyph.shape[0] <= 0.23 * card.shape[0]


def _card_back():
    """The hero's card during the deal, face down: a centred logo, no rank."""
    card = np.full((170, 115, 3), (60, 45, 35), dtype=np.uint8)
    cv2.circle(card, (57, 46), 26, (235, 235, 235), -1)
    cv2.circle(card, (57, 104), 26, (235, 235, 235), -1)
    return card


def test_a_card_back_reads_as_no_rank_at_all():
    # Rejecting the back's logo as a glyph is not enough on its own: the
    # window cascade behind it then OCRs the whole top of the card and
    # invents a rank from the logo. Live, a back read "4s" and then "5s"
    # once the glyph was rejected. A card with no isolable rank is not a
    # card this can read, and saying so is the only safe answer -- the deal
    # happens at preflop, which is exactly when HeroCardTracker latches.
    assert card_vision.read_rank_from_top(_card_back()) is None


def test_a_real_fanned_rank_is_unaffected():
    card = np.full((170, 115, 3), (160, 60, 40), dtype=np.uint8)
    cv2.putText(card, "8", (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)
    cv2.circle(card, (58, 115), 26, (255, 255, 255), -1)

    assert card_vision.read_rank_from_top(card) == "8"
