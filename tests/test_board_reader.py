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
    """
    frame = np.zeros((200, 10 + 5 * 140, 3), dtype=np.uint8)
    frame[:] = (90, 60, 30)                      # felt
    templates = rank_templates.load()
    for i, card in enumerate(cards):
        rank, suit = card[0], card[1]
        x1, y1, x2, y2 = SLOTS[i]
        frame[y1:y2, x1:x2] = SUIT_BGR[suit]
        cv2.circle(frame, ((x1 + x2) // 2, y1 + 115), 34, (255, 255, 255), -1)  # suit pip
        gx1, gy1, gx2, gy2 = RANK_OFFSET
        glyph = cv2.resize(templates[rank][0], (gx2 - gx1, gy2 - gy1),
                           interpolation=cv2.INTER_NEAREST)
        region = frame[y1 + gy1:y1 + gy2, x1 + gx1:x1 + gx2]
        region[glyph > 127] = (255, 255, 255)
    return frame


def test_reads_a_single_board_card():
    frame = _frame(["Qd"])

    assert board_reader.read_board(frame, SLOTS, RANK_OFFSET) == ["Qd"]


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
