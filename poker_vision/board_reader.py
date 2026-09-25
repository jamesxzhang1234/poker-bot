"""Reads the community board from a frame.

Rank comes from template matching (rank_templates) and suit from the card's
background colour (card_vision.classify_suit) -- no OCR engine is involved.
The board is a closed alphabet rendered in one fixed font, which is the case
OCR handles worst and template matching handles best.
"""
import cv2
import numpy as np

from . import card_vision, rank_templates

# A card face is bright; empty felt is not. Measured on a real pre-flop frame
# (IMG_8425), an empty board slot tops out at gray 92 while a card face is far
# brighter. card_vision.is_card_present keys on edge density instead and
# reports True for those empty slots -- on that frame it produced three
# phantom cards, one of which OCR then "read" a rank from. Brightness
# separates the two cleanly, so this path uses its own test.
_CARD_BRIGHT_LEVEL = 150
# A real Qd on the iPad measured 11.7% bright, so 12% cut it out by a hair and
# the "stop at the first gap" rule then truncated the rest of the board. Felt
# measures ~0%, so there is a lot of room below this before anything empty
# starts reading as a card.
_CARD_BRIGHT_FRACTION = 0.08

# Brightness cutoffs tried for the rank glyph, brightest first. One value is
# not enough: at showdown the client greys out the cards not in the winning
# hand, and their ink falls below any threshold that suits a normal card. A
# real five-card board came back as a single card because the second was
# dimmed. Measured across a dimmed board, every slot reads correctly at 110
# while bright cards score best around 140-170.
_GLYPH_LEVELS = (170, 140, 110)


def card_present(crop):
    """Whether this slot holds a face-up card rather than empty felt."""
    if crop is None or crop.size == 0:
        return False
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float((gray > _CARD_BRIGHT_LEVEL).mean()) > _CARD_BRIGHT_FRACTION


def rank_glyph(card_crop, rank_offset, level=_GLYPH_LEVELS[0]):
    """The binary rank glyph from a card's top-left corner."""
    x1, y1, x2, y2 = rank_offset
    corner = card_crop[y1:y2, x1:x2]
    if corner.size == 0:
        return None
    gray = cv2.cvtColor(corner, cv2.COLOR_BGR2GRAY)
    return (gray > level).astype(np.uint8) * 255


def best_rank(card_crop, rank_offset):
    """The rank this card shows.

    Tries the glyph located by shape first -- as one character and as two,
    since "10" is the one rank drawn as a pair -- then the fixed corner box
    at each brightness cutoff, and keeps the highest-scoring match. Locating it matters
    because the slots drift right of the cards by the fifth position, so the
    river's rank falls outside the fixed box and went unread entirely -- on a
    live hand that cost the whole river card. The cutoffs matter because a
    bright card scores best at a high one while a dimmed showdown card only
    registers at a low one.
    """
    best, best_score = None, 0.0

    # The located glyph and, when the rank is drawn as two characters, the
    # pair. Scoring both and keeping the winner means the ten is found
    # without the other twelve ranks having to change: a single-character
    # rank yields no pair at all.
    for isolate in (card_vision.isolate_rank_glyph, card_vision.isolate_rank_pair):
        located = isolate(card_crop)
        if located is None:
            continue
        scores = rank_templates.score_all(located)
        if scores and scores[0][1] > best_score:
            best, best_score = scores[0]

    for level in _GLYPH_LEVELS:
        glyph = rank_glyph(card_crop, rank_offset, level)
        if glyph is None:
            continue
        scores = rank_templates.score_all(glyph)
        if scores and scores[0][1] > best_score:
            best, best_score = scores[0][0], scores[0][1]

    return best if best_score >= rank_templates.MIN_SCORE else None


def read_card(card_crop, rank_offset):
    """One card as e.g. "Qd", or None if the slot has no readable card.

    A matched rank is itself the presence test. The brightness gate used to
    come first and rejected dimmed cards outright; felt, chips and the table
    watermark score only 0.18-0.39 against the templates, well under the bar,
    so nothing is gained by asking twice.
    """
    rank = best_rank(card_crop, rank_offset)
    if rank is None:
        return None
    suit = card_vision.classify_suit(card_vision.sample_card_color(card_crop))
    return f"{rank}{suit}"


# A board is dealt 0, then 3, then 4, then 5. One or two cards is not a state
# the game can be in, so a read of that length is a misread rather than a
# partial truth. Mid-deal the flop slides in and the cards sit briefly offset
# from their slots; a real frame read "7c Qh" that way, two cards with one of
# them wrong. Rejecting by length needs no confidence threshold, which would
# have had to sit at 0.10 margin to catch it and would risk good reads.
_POSSIBLE_BOARD_SIZES = (0, 3, 4, 5)


def read_board(frame, slots, rank_offset):
    """The community cards, left to right, or [] if the read is not a board.

    Stops at the first slot without a card: the board is dealt left to right,
    so a gap means the rest are undealt rather than unreadable.
    """
    board = []
    height, width = frame.shape[:2]
    for x1, y1, x2, y2 in slots:
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, width), min(y2, height)
        if x2 <= x1 or y2 <= y1:
            break
        card = read_card(frame[y1:y2, x1:x2], rank_offset)
        if card is None:
            break
        board.append(card)
    return board if len(board) in _POSSIBLE_BOARD_SIZES else []
