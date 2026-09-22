"""Card recognition: suit by background color, rank by OCR on the corner glyph."""
import cv2
import numpy as np
import pytesseract

from . import regions

# Reference BGR colors sampled from real card backgrounds (see calibration
# notes in regions.py). Suits are color-coded by this client, so we classify
# by dominant channel + overall brightness rather than needing per-rank
# templates for every suit.
_RANK_WHITELIST = "23456789TJQKA"


def classify_suit(bgr_mean):
    b, g, r = bgr_mean
    brightness = b + g + r
    # Spades render on a dark navy background, not truly black -- measured
    # ~46-96 brightness on real cards (confirmed: a threshold of 60 let
    # several through as false diamonds, since navy's blue channel still
    # edges out red/green once you're past pure black). Real diamond/club/
    # heart backgrounds measured well clear of this, starting at ~125.
    if brightness < 110:
        return "s"
    if g >= b and g >= r:
        return "c"  # clubs: green
    if b >= g and b >= r:
        return "d"  # diamonds: blue
    return "h"  # hearts: red


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


def read_rank(card_crop):
    dx1, dy1, dx2, dy2 = regions.RANK_OFFSET
    corner = card_crop[dy1:dy2, dx1:dx2]
    if corner.size == 0:
        return None
    gray = cv2.cvtColor(corner, cv2.COLOR_BGR2GRAY)
    # Rank glyph is bold white/light text on a darker background; compression
    # softens it by varying amounts (board cards vs. the hero's own card
    # overlay render at different sharpness), so an adaptive Otsu threshold
    # is more robust here than one fixed brightness cutoff.
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    mask = cv2.resize(mask, (mask.shape[1] * 4, mask.shape[0] * 4), interpolation=cv2.INTER_CUBIC)
    # psm 10 ("treat as a single character") is wrong surprisingly often on
    # this font -- confirmed on real hands: a clean, legible "5" and "6"
    # both came back empty at psm 10 whitelisted, while psm 8/13 read them
    # correctly. Try a short cascade rather than trust one psm, matching
    # the pattern already used for badges/stack amounts.
    for psm in (10, 8, 13, 7, 6):
        config = f'--psm {psm} -c tessedit_char_whitelist={_RANK_WHITELIST}'
        text = pytesseract.image_to_string(mask, config=config).strip()
        text = text.replace("0", "").strip()  # stray zero from a misread "10"
        if not text:
            continue
        rank = text[0].upper()
        if rank in _RANK_WHITELIST:
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
