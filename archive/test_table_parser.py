"""Synthetic-pixel tests for TableParser.parse_seat's orchestration logic:
the status fallback (bubble action, then persistent label), the in_hand
combination (card-presence heuristic overridden by a Folded status), and
hero's hole-card override -- plus _read_badge's multi-candidate fallback,
which had a real historical bug (badges relocate depending on in_hand;
see regions.py's badge-position notes).

Draws real, Tesseract-legible color/text patches onto a blank
reference-resolution frame and calls parse_seat directly, no mocking,
matching the rest of this test suite (see test_number_ocr.py).
"""
import cv2
import numpy as np

from archive import regions
from archive.parser import TableParser


def _blank_frame():
    return np.zeros((regions.REF_H, regions.REF_W, 3), dtype=np.uint8)


def _draw_textured(frame, box, step=8, value=120):
    """Enough edge density for card_vision.is_card_present to read True,
    without needing a specific rank/suit -- used for seats where only
    "a card is present" matters, not which card. Kept below the
    white/red text-mask thresholds (ocr_utils.white_mask/label_mask) so
    it doesn't bleed into an overlapping status crop as false text."""
    x1, y1, x2, y2 = box
    frame[y1:y2:step, x1:x2] = value
    frame[y1:y2, x1:x2:step] = value


def _draw_label_text(frame, box, text, color=(40, 40, 220)):
    """Persistent status text (Folded/Sitting Out/All-in): red/pink or
    white, per ocr_utils.label_mask."""
    x1, y1, x2, y2 = box
    cv2.putText(frame, text, (x1 + 8, y2 - 12), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3, cv2.LINE_AA)


def _draw_hero_card(frame, box, rank, suit_bgr):
    """A hero hole card: colored background (suit), a double border so
    is_card_present's edge-density heuristic reliably triggers even for
    a narrow single-digit rank glyph, and a rank glyph high enough in the
    crop for read_rank_from_top.

    The inner border is a hairline on purpose. Drawn 2px thick it merged
    with the rank glyph in the narrower of the two slots, making one
    component as tall as the card -- which no real card produces, and which
    the reader now rejects as a card back. The border is here for edge
    density, not because cards have one ruled inside their edge."""
    x1, y1, x2, y2 = box
    frame[y1:y2, x1:x2] = suit_bgr
    cv2.rectangle(frame, (x1 + 2, y1 + 2), (x2 - 2, y2 - 2), (255, 255, 255), 5)
    cv2.rectangle(frame, (x1 + 10, y1 + 10), (x2 - 10, y2 - 10), (255, 255, 255), 1)
    cv2.putText(frame, rank, (x1 + 22, y1 + 60), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (230, 230, 230), 4, cv2.LINE_AA)


def _draw_badge(frame, box, text):
    x1, y1, x2, y2 = box
    frame[y1:y2, x1:x2] = (200, 190, 170)
    cv2.putText(frame, text, (x1 + 4, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (90, 85, 75), 2, cv2.LINE_AA)


def test_folded_status_overrides_card_presence_in_hand():
    frame = _blank_frame()
    cfg = regions.SEATS["bottom_left"]
    _draw_textured(frame, cfg["cards"])
    _, label_zone = regions.status_sub_zones(cfg)
    _draw_label_text(frame, label_zone, "Folded")

    seat = TableParser().parse_seat(frame, "bottom_left")

    assert seat.status == "Folded"
    assert seat.in_hand is False


def test_card_presence_sets_in_hand_when_no_status_text():
    frame = _blank_frame()
    cfg = regions.SEATS["bottom_left"]
    _draw_textured(frame, cfg["cards"])

    seat = TableParser().parse_seat(frame, "bottom_left")

    assert seat.status is None
    assert seat.in_hand is True


def test_hero_hole_cards_set_in_hand_even_without_a_card_back_crop():
    frame = _blank_frame()
    _draw_hero_card(frame, regions.HERO_CARDS[0], "K", suit_bgr=(40, 160, 40))  # green -> clubs
    _draw_hero_card(frame, regions.HERO_CARDS[1], "7", suit_bgr=(30, 30, 190))  # red -> hearts

    seat = TableParser().parse_seat(frame, "hero")

    assert seat.is_hero
    assert seat.hole_cards == ["Kc", "7h"]
    assert seat.in_hand is True


def test_read_badge_falls_through_to_the_second_candidate_position():
    frame = _blank_frame()
    _, second_box = regions.SEATS["top_left"]["badge"]
    _draw_badge(frame, second_box, "BB")

    seat = TableParser().parse_seat(frame, "top_left")

    assert seat.position == "BB"
