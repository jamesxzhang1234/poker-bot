"""Tests for reading the board from a frame.

Cards are read by template matching the rank glyph and classifying the suit
from the card's background colour -- no OCR engine is involved, because the
board is a closed alphabet in a fixed font (see rank_templates).
"""
import cv2
import numpy as np

from poker_vision import board_reader, rank_templates

SLOTS = [(10 + i * 140, 10, 10 + i * 140 + 120, 170) for i in range(5)]
RANK_OFFSET = (8, 6, 70, 50)
# Background colours matching what card_vision.classify_suit documents from
# real cards: spades are dark navy (b+g+r measured 46-96, NOT black), and the
# other three sit well clear of that, starting around 125.
SUIT_BGR = {
    "s": (50, 25, 20),     # navy,  sum 95
    "h": (40, 40, 160),    # red,   sum 240
    "d": (160, 60, 40),    # blue,  sum 260
    "c": (50, 150, 50),    # green, sum 250
}


def _frame(cards):
    """Build a frame with `cards` (e.g. ["Qd", "5s"]) in the left-most slots.

    A real card is a dark suit-coloured face carrying a lot of white ink --
    the rank glyph and, dominating it, a large suit pip. That white fraction
    is what tells a card apart from felt, so the fixture has to draw the pip
    too; a card with only its rank on it is not what the reader sees.

    The rank is drawn at the size the device draws it: measured across the
    live corpus, a real rank glyph is 11-22% of its card's height, median
    19%. This fixture used to fill the whole RANK_OFFSET box, which is 27%,
    and that is larger than any rank the iPad has ever rendered -- so it
    kept passing when the reader started rejecting over-tall glyphs as card
    backs, which is a real distinction drawn at 23%.
    """
    frame = np.zeros((200, 10 + 5 * 140, 3), dtype=np.uint8)
    frame[:] = (90, 60, 30)                      # felt
    templates = rank_templates.load()
    for i, card in enumerate(cards):
        rank, suit = card[0], card[1]
        x1, y1, x2, y2 = SLOTS[i]
        slot_height = y2 - y1
        frame[y1:y2, x1:x2] = SUIT_BGR[suit]
        cv2.circle(frame, ((x1 + x2) // 2, y1 + 115), 34, (255, 255, 255), -1)  # suit pip
        gx1, gy1, _, _ = RANK_OFFSET
        gh = int(slot_height * 0.19)
        gw = int(gh * 0.75)
        glyph = cv2.resize(templates[rank][0], (gw, gh), interpolation=cv2.INTER_NEAREST)
        region = frame[y1 + gy1:y1 + gy1 + gh, x1 + gx1:x1 + gx1 + gw]
        region[glyph > 127] = (255, 255, 255)
    return frame


def test_reads_a_flop():
    # The smallest real board is three cards. This used to assert that a
    # single card read back, which is not a state the game can be in.
    frame = _frame(["Qd", "5s", "8h"])

    assert board_reader.read_board(frame, SLOTS, RANK_OFFSET) == ["Qd", "5s", "8h"]


def test_reads_a_full_five_card_board():
    cards = ["2s", "7h", "Tc", "Kd", "As"]

    assert board_reader.read_board(_frame(cards), SLOTS, RANK_OFFSET) == cards


def test_an_empty_board_reads_as_no_cards():
    frame = np.zeros((200, 10 + 5 * 140, 3), dtype=np.uint8)
    frame[:] = (90, 60, 30)

    assert board_reader.read_board(frame, SLOTS, RANK_OFFSET) == []


def test_reads_only_the_slots_that_hold_a_card():
    # The board is dealt left to right; the later slots stay empty felt.
    assert board_reader.read_board(_frame(["5s", "8h", "Qd"]), SLOTS, RANK_OFFSET) == \
        ["5s", "8h", "Qd"]


def test_a_slot_of_bare_felt_is_not_a_card():
    # card_vision.is_card_present keys on edge density and reports True on
    # empty slots -- on IMG_8425 (a pre-flop frame with no board at all) that
    # produced three phantom cards. Brightness separates them cleanly: felt
    # tops out around gray 92 where a card face is far brighter.
    frame = np.zeros((200, 10 + 5 * 140, 3), dtype=np.uint8)
    frame[:] = (90, 60, 30)
    cv2.circle(frame, (70, 90), 40, (100, 70, 40), 2)   # texture, still dark

    assert board_reader.read_board(frame, SLOTS, RANK_OFFSET) == []


def test_a_dimmed_card_is_still_read():
    # At showdown the client greys out the cards not in the winning hand. A
    # single bright threshold finds no ink on those at all -- a real five-card
    # board came back as one card, because read_board stops at the first slot
    # it cannot read and the second card was dimmed.
    import cv2
    import numpy as np
    from poker_vision import rank_templates
    frame = np.zeros((200, 10 + 5 * 140, 3), dtype=np.uint8)
    frame[:] = (90, 60, 30)
    templates = rank_templates.load()
    for i, (rank, bgr, ink) in enumerate([
            ("7", SUIT_BGR["c"], 255),          # normal card, white ink
            ("T", (20, 18, 70), 130),           # dimmed card, grey ink
            ("6", SUIT_BGR["h"], 255)]):
        x1, y1, x2, y2 = SLOTS[i]
        frame[y1:y2, x1:x2] = bgr
        cv2.circle(frame, ((x1 + x2) // 2, y1 + 115), 34, (ink, ink, ink), -1)
        gx1, gy1, gx2, gy2 = RANK_OFFSET
        glyph = cv2.resize(templates[rank][0], (gx2 - gx1, gy2 - gy1),
                           interpolation=cv2.INTER_NEAREST)
        region = frame[y1 + gy1:y1 + gy2, x1 + gx1:x1 + gx2]
        region[glyph > 127] = (ink, ink, ink)

    board = board_reader.read_board(frame, SLOTS, RANK_OFFSET)

    assert len(board) == 3, board          # the dimmed card must not truncate it
    assert board[0][0] == "7" and board[1][0] == "T" and board[2][0] == "6"


def test_an_impossible_board_length_is_rejected():
    # A board goes 0, 3, 4, 5 -- never 1 or 2. Mid-deal the flop slides in and
    # the cards are briefly offset from their slots, so a real frame read
    # "7c Qh": two cards, one of them wrong. Length alone rules that out
    # without needing a confidence threshold that would risk good reads.
    frame = _frame(["5s", "8h"])

    assert board_reader.read_board(frame, SLOTS, RANK_OFFSET) == []


def test_a_flop_is_kept():
    cards = ["5s", "8h", "Qd"]

    assert board_reader.read_board(_frame(cards), SLOTS, RANK_OFFSET) == cards


def test_a_single_card_is_rejected_too():
    assert board_reader.read_board(_frame(["5s"]), SLOTS, RANK_OFFSET) == []


def test_a_card_offset_within_its_slot_is_still_read():
    # The board slots drift right of the cards by the fifth position, so the
    # rank glyph falls outside the fixed corner box and the river went unread
    # -- costing the whole river card on a live hand. Locating the glyph
    # instead of assuming where it sits removes the dependence on the slot
    # being perfectly aligned.
    import cv2
    import numpy as np
    from poker_vision import rank_templates
    frame = np.zeros((200, 10 + 5 * 140, 3), dtype=np.uint8)
    frame[:] = (90, 60, 30)
    templates = rank_templates.load()
    for i, rank in enumerate(["5", "Q", "7"]):
        x1, y1, x2, y2 = SLOTS[i]
        frame[y1:y2, x1:x2] = SUIT_BGR["c"]
        cv2.circle(frame, ((x1 + x2) // 2, y1 + 115), 34, (255, 255, 255), -1)
        gx1, gy1, _, _ = RANK_OFFSET
        # 19% of the card's height, the median real rank glyph -- see _frame.
        gh = int((y2 - y1) * 0.19)
        gw = int(gh * 0.75)
        glyph = cv2.resize(templates[rank][0], (gw, gh), interpolation=cv2.INTER_NEAREST)
        # the last card sits shifted, as the real fifth slot does
        shift = 26 if i == 2 else 0
        region = frame[y1 + gy1:y1 + gy1 + gh,
                       x1 + gx1 + shift:x1 + gx1 + gw + shift]
        region[glyph > 127] = (255, 255, 255)

    board = board_reader.read_board(frame, SLOTS, RANK_OFFSET)

    assert len(board) == 3, board
    assert [c[0] for c in board] == ["5", "Q", "7"]


def test_a_real_ipad_ace_on_the_board_is_read():
    # A live hand's whole board went unread because its first card was an
    # ace. The flat set held only iPhone aces; the iPad's scored 0.47, under
    # the 0.55 floor, and led "4" by 0.01. read_board stops at the first
    # unreadable slot, so all five cards were lost for fourteen frames.
    #
    # This is the one case a drawn fixture cannot reproduce -- the glyph
    # matched nothing precisely because of how the device renders it -- so
    # the card crop is a real one, kept beside the test.
    from pathlib import Path

    from poker_vision import seat_anchors

    crop = cv2.imread(str(Path(__file__).parent / "data" / "ipad_board_ace.png"))
    assert crop is not None, "fixture missing"

    assert board_reader.best_rank(crop, seat_anchors.IPAD_RANK_OFFSET) == "A"


def test_a_real_ipad_ten_on_the_board_is_read():
    # "10" is the one rank drawn as two characters, and isolate_rank_glyph
    # returns a single component -- so a board ten came back as just its "1",
    # which scores as a J. A live showdown opened Td Tc and the board was
    # lost from the first slot. The fixed corner box holds both characters
    # but also the top of the suit pip, which is too large to be dropped as
    # a fragment, so that path reads the ten no better.
    from pathlib import Path

    from poker_vision import seat_anchors

    crop = cv2.imread(str(Path(__file__).parent / "data" / "ipad_board_ten.png"))
    assert crop is not None, "fixture missing"

    assert board_reader.best_rank(crop, seat_anchors.IPAD_RANK_OFFSET) == "T"


def _ipad_frame(cards):
    """A frame with `cards` drawn where the iPad actually renders them.

    Positions are measured off seven real five-card boards, not taken from
    the slot constants this exercises: the cards start at x=494, repeat every
    132px and are 125px wide, sitting at y=985 over 154px. The rank glyph
    sits 14px in from the card's left edge and 13px down, which is what makes
    the accumulated slot drift bite.
    """
    frame = np.zeros((1300, 1700, 3), dtype=np.uint8)
    frame[:] = (90, 60, 30)                          # felt
    templates = rank_templates.load()
    for i, card in enumerate(cards):
        rank, suit = card[0], card[1]
        x1, y1 = 494 + i * 132, 985
        frame[y1:y1 + 154, x1:x1 + 125] = SUIT_BGR[suit]
        cv2.circle(frame, (x1 + 62, y1 + 115), 30, (255, 255, 255), -1)   # suit pip
        # 29px is 19% of the 154px card, the median real rank glyph.
        glyph = cv2.resize(templates[rank][0], (24, 29), interpolation=cv2.INTER_NEAREST)
        region = frame[y1 + 13:y1 + 42, x1 + 14:x1 + 38]
        region[glyph > 127] = (255, 255, 255)
    return frame


def test_the_board_slots_line_up_with_where_the_ipad_draws_the_cards():
    # The slots were derived as pitch 136 / width 118, but the cards repeat
    # every 132px and are 125px wide. The error accumulates left to right, so
    # by the fifth slot the crop sits 17px right of its card and cuts off the
    # rank glyph -- the river read as the wrong card or not at all. Measured
    # across a live corpus, this cost 11 of 16 known boards.
    from poker_vision import seat_anchors

    # A ten in the fifth slot, as the live board that exposed this had. The
    # drift clips only a few pixels, which a single glyph survives; "10" is
    # two characters and loses its "1", leaving a bare "0" that matched Q.
    cards = ["Jd", "4h", "9s", "8h", "Tc"]
    frame = _ipad_frame(cards)

    assert board_reader.read_board(
        frame, seat_anchors.IPAD_BOARD_SLOTS, seat_anchors.IPAD_RANK_OFFSET) == cards
