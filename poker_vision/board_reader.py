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

# The rank glyph renders white on every suit's background.
_GLYPH_LEVEL = 170


def card_present(crop):
    """Whether this slot holds a face-up card rather than empty felt."""
    if crop is None or crop.size == 0:
        return False
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float((gray > _CARD_BRIGHT_LEVEL).mean()) > _CARD_BRIGHT_FRACTION


def rank_glyph(card_crop, rank_offset):
    """The binary rank glyph from a card's top-left corner."""
    x1, y1, x2, y2 = rank_offset
    corner = card_crop[y1:y2, x1:x2]
    if corner.size == 0:
        return None
    gray = cv2.cvtColor(corner, cv2.COLOR_BGR2GRAY)
    return (gray > _GLYPH_LEVEL).astype(np.uint8) * 255


def read_card(card_crop, rank_offset):
    """One card as e.g. "Qd", or None if the slot has no readable card."""
    if not card_present(card_crop):
        return None
    rank = rank_templates.match(rank_glyph(card_crop, rank_offset))
    if rank is None:
        return None
    suit = card_vision.classify_suit(card_vision.sample_card_color(card_crop))
    return f"{rank}{suit}"


def read_board(frame, slots, rank_offset):
    """The community cards, left to right.

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
    return board
