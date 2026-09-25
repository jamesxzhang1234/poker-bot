"""Card recognition: suit by background color, rank by OCR on the corner glyph."""
import cv2
import numpy as np

from . import regions
from .ocr_utils import _cascade, _ocr

# Reference BGR colors sampled from real card backgrounds (see calibration
# notes in regions.py). Suits are color-coded by this client, so we classify
# by dominant channel + overall brightness rather than needing per-rank
# templates for every suit.
_RANK_WHITELIST = "23456789TJQKA"


# Spades render on a dark navy background, not truly black. The test is on
# the BRIGHTEST channel rather than the sum of all three, because the sum
# leaves almost no margin: sampled across 163 real card backgrounds on both
# devices, spades reach a sum of 107 while the old cutoff sat at 110. The same
# iPad spade measured 107 in one frame and 116 in the next and so read "4s"
# then "4d". By brightest channel the classes are far apart -- spades top out
# at 55 and the dimmest non-spade (a club) starts at 110 -- so this threshold
# sits in a wide empty gap instead of on top of the data.
#
# This is for BOARD cards, sampled from a corner patch of an unobstructed
# card. Hero cards are sampled differently and need their own rule; see
# classify_hero_suit.
_SPADE_MAX_CHANNEL = 80


def classify_suit(bgr_mean):
    b, g, r = bgr_mean
    if max(b, g, r) < _SPADE_MAX_CHANNEL:
        return "s"
    if g >= b and g >= r:
        return "c"  # clubs: green
    if b >= g and b >= r:
        return "d"  # diamonds: blue
    return "h"  # hearts: red


# Hero cards need a different test from the board's. Their suit is sampled
# from the exposed left edge of a fanned card rather than a clean corner, and
# the client dims them when a hand ends -- a dimmed diamond measured
# (60,40,23), inside the navy brightness range, so the board's brightness test
# calls it a spade. Ratios survive dimming where absolute levels do not: that
# card's blue is 2.6x its red, matching the bright diamond's 2.8x, while hero
# navy sits at 1.47-1.54.
#
# The board rule cannot simply adopt this: sampled from a corner, board navy
# runs 2.25-2.5 and board blue 2.61, which ratios cannot separate. Two
# sampling regions, two rules.
_HERO_DIAMOND_BLUE_RATIO = 2.0


def classify_hero_suit(bgr_mean):
    b, g, r = bgr_mean
    if r >= b and r >= g:
        return "h"
    if g >= b and g >= r:
        return "c"
    return "d" if b >= _HERO_DIAMOND_BLUE_RATIO * max(r, 1.0) else "s"


def _sample_bg_color(card_crop):
    h, w = card_crop.shape[:2]
    patch = card_crop[int(h * 0.06):int(h * 0.16), w - int(w * 0.16):w - int(w * 0.06)]
    if patch.size == 0:
        patch = card_crop
    return patch.reshape(-1, 3).mean(axis=0)


def is_card_present(crop, edge_density_threshold=0.02):
    """Heuristic: an empty felt background is smooth; a card (face-up or
    face-down back design) has much higher edge density."""
    if crop is None or crop.size == 0:
        return False
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 160)
    density = np.count_nonzero(edges) / edges.size
    return density > edge_density_threshold


def _extract_rank(text):
    """Pull a valid rank character out of a raw OCR string, or None."""
    text = text.replace("0", "").strip()  # stray zero from a misread "10"
    if not text:
        return None
    rank = text[0].upper()
    return rank if rank in _RANK_WHITELIST else None


def _read_rank(mask, psms):
    """Upscale a black-text-on-white rank mask and try it under each psm in
    `psms` in turn (ocr_utils._cascade), taking the first valid rank read.
    Shared by _ocr_rank and _ocr_rank_bright, which differ only in how the
    mask itself gets built and in which psm order works best for that mask.
    """
    upscaled = cv2.resize(mask, (mask.shape[1] * 4, mask.shape[0] * 4), interpolation=cv2.INTER_CUBIC)
    return _cascade(psms, lambda psm: _extract_rank(_ocr(upscaled, whitelist=_RANK_WHITELIST, psm=psm)))


def _ocr_rank(region_bgr):
    if region_bgr.size == 0:
        return None
    gray = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2GRAY)
    # Rank glyph is bold white/light text on a darker background; compression
    # softens it by varying amounts (board cards vs. the hero's own card
    # overlay render at different sharpness), so an adaptive Otsu threshold
    # is more robust here than one fixed brightness cutoff.
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # psm 10 ("treat as a single character") is wrong surprisingly often on
    # this font -- confirmed on real hands: a clean, legible "5" and "6"
    # both came back empty at psm 10 whitelisted, while psm 8/13 read them
    # correctly.
    return _read_rank(mask, (10, 8, 13, 7, 6))


def read_rank(card_crop):
    dx1, dy1, dx2, dy2 = regions.RANK_OFFSET
    corner = card_crop[dy1:dy2, dx1:dx2]
    return _ocr_rank(corner)


def _ocr_rank_bright(region_bgr, thresh=100):
    """Like _ocr_rank, but masks by a fixed brightness cutoff instead of
    adaptive Otsu thresholding. Used for hero's rotated cards (see
    read_rank_from_top): Otsu is a single threshold over the *whole* crop,
    which is more sensitive to a background gradient and other shapes
    (a suit pip, the neighboring overlapping card's edge) bleeding into a
    looser, position-tolerant crop than adaptive thresholding can absorb --
    confirmed on a real hand where Otsu misread a clearly-legible "K" as
    "7", but a fixed brightness mask read it correctly. psm 6 tried first:
    empirically the most reliable of the cascade on this glyph style,
    unlike flat/corner-cropped cards where psm 10 usually wins.
    """
    if region_bgr.size == 0:
        return None
    b = region_bgr[:, :, 0].astype(int)
    g = region_bgr[:, :, 1].astype(int)
    r = region_bgr[:, :, 2].astype(int)
    bright = (r > thresh) & (g > thresh) & (b > thresh)
    mask = (~bright).astype(np.uint8) * 255
    return _read_rank(mask, (6, 10, 8, 13, 7))


def read_rank_from_top(card_crop, top_fractions=(0.4, 0.45, 0.5, 0.55, 0.6, 0.65)):
    """Like read_rank, but for a card whose rank doesn't sit at a reliable
    fixed corner offset (a rotated/fanned card -- see HERO_CARDS): searches
    a top portion of the whole crop instead of one tiny fixed sub-box.

    Tries a cascade of window sizes rather than one fixed fraction: the
    card's vertical position within its (deliberately generous) box isn't
    fixed frame to frame -- confirmed on two real hands where the rank sat
    at different depths. A single window big enough to reach the lower
    position also drags a suit pip into the crop for the higher position,
    confusing the OCR into misreading a rank that was already legible on
    its own.

    Known remaining gap: a dim/faded render state (seen on a post-showdown
    frame) can still misread even with this cascade -- not yet solved.
    """
    h = card_crop.shape[0]
    for fraction in top_fractions:
        rank = _ocr_rank_bright(card_crop[0:int(h * fraction), :])
        if rank is not None:
            return rank
    return None


def parse_card(card_crop):
    """Given a cropped card slot image, return e.g. '7c', 'Th', or None if
    the slot is empty / unreadable."""
    if not is_card_present(card_crop):
        return None
    suit = classify_suit(_sample_bg_color(card_crop))
    rank = read_rank(card_crop)
    if rank is None:
        return None
    return f"{rank}{suit}"


def _sample_center_color(card_crop):
    """Suit-background sample for a rotated card (see parse_hero_card):
    _sample_bg_color's fixed top-right corner assumes an upright card and
    can land off the actual (rotated) card surface entirely -- confirmed
    on a real hand, where it sampled empty background and misclassified a
    diamond as a spade. The crop's center is reliably on the card
    regardless of rotation direction, as long as the crop reasonably
    bounds the card.
    """
    h, w = card_crop.shape[:2]
    patch = card_crop[int(h * 0.35):int(h * 0.55), int(w * 0.35):int(w * 0.65)]
    if patch.size == 0:
        patch = card_crop
    return patch.reshape(-1, 3).mean(axis=0)


def sample_face_color(card_crop):
    """Average colour of a fanned card's own face.

    Samples the left portion rather than the centre. The hero's two cards
    overlap, so the one behind has its middle covered by the one in front and
    a centre sample averages the two together -- that read a 9 of spades as a
    diamond, and read dimmed red queens as spades. The left edge is the part
    of a fanned card that stays exposed.

    Swept over all 122 hero cards in the real screenshots: 13 suits changed,
    and every one checked by eye was a correction.
    """
    h, w = card_crop.shape[:2]
    patch = card_crop[int(h * 0.35):int(h * 0.75), int(w * 0.05):int(w * 0.35)]
    if patch.size == 0:
        patch = card_crop
    return patch.reshape(-1, 3).mean(axis=0)


def parse_hero_card(card_crop):
    """Like parse_card, but for one of hero's own (fanned/rotated) cards --
    see HERO_CARDS. HERO_CARDS' boxes are deliberately generous (rotated
    cards don't sit at one exact fixed position frame to frame -- confirmed
    on two real hands with a ~65px vertical difference), so both the rank
    search and the suit sample use rotation/position-tolerant strategies
    instead of parse_card's fixed corner offsets.
    """
    if not is_card_present(card_crop):
        return None
    suit = classify_hero_suit(sample_face_color(card_crop))
    rank = read_rank_from_top(card_crop)
    if rank is None:
        return None
    return f"{rank}{suit}"
