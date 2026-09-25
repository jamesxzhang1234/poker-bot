"""Card-rank recognition by template matching.

The board renders 13 ranks in one fixed font at one fixed size, which makes
this a closed-alphabet matching problem rather than an OCR problem -- and the
OCR engines measurably struggle with it. Apple Vision reads only 2 of 4 board
ranks on a whole frame and 1 of 4 on per-card crops, because an isolated glyph
in a small crop often isn't detected as text at all. Tesseract manages it but
costs ~490ms a frame, which would dominate the 90ms Vision pass it sits beside.

Templates are stored bbox-normalised to a fixed size, and each rank may hold
SEVERAL exemplars. Normalising alone does not make a template device-
independent, which was the original assumption and is wrong: the two devices
render the same rank at different weights and proportions, so an iPhone-built
"8" scored below "3" on a perfectly clean iPad 8, and an iPad-built "10"
scored 0.21 against iPhone tens. Matching takes the best score across a rank's
exemplars, so adding a device means adding exemplars rather than retuning
anything.

Exemplars are stored in one npz keyed "<rank>#<n>".
"""
from pathlib import Path

import cv2
import numpy as np

_TEMPLATE_FILE = Path(__file__).resolve().parent / "data" / "rank_templates.npz"

TEMPLATE_SIZE = (24, 32)   # (width, height) every glyph is normalised to

# Below this correlation the glyph is not any rank we know. An empty board slot
# is dark felt, and a slot part-way through a deal animation is a fragment --
# reporting a confident rank for either would put a card on the table that
# isn't there, so a weak match returns None instead of a best guess.
MIN_SCORE = 0.55

# Smallest ink a real glyph leaves once normalised. Guards against matching
# a speck of noise against whichever template happens to correlate best.
_MIN_INK_PIXELS = 20

# A component smaller than this fraction of the largest one is not part of the
# rank. The rank crop reaches far enough down to catch the top of the suit pip
# on some cards, and that fragment stretches the bounding box so normalising
# squashes the digit -- on IMG_8437 a perfectly clean "5" scored 0.60 as "9"
# and never ranked "5" at all. The threshold stays well below 1.0 because
# "10" is two components of comparable size and both must survive.
_FRAGMENT_FRACTION = 0.25

_cache = None


def load():
    """The rank -> list-of-exemplars mapping, read once and cached."""
    global _cache
    if _cache is None:
        grouped = {}
        with np.load(_TEMPLATE_FILE) as data:
            for key in data.files:
                rank = key.split("#", 1)[0]
                grouped.setdefault(rank, []).append(data[key])
        _cache = grouped
    return _cache


def _drop_fragments(binary):
    """Remove blobs far smaller than the largest -- suit-pip bleed, specks."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (binary > 0).astype(np.uint8), connectivity=8)
    if count <= 2:
        return binary
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = {i + 1 for i, a in enumerate(areas) if a >= _FRAGMENT_FRACTION * areas.max()}
    cleaned = np.zeros_like(binary)
    for label in keep:
        cleaned[labels == label] = 255
    return cleaned


def normalise(mask):
    """Crop a binary glyph to its ink and resize it to the template size.

    Returns None when there is too little ink to be a rank.
    """
    if mask is None or mask.size == 0:
        return None
    binary = mask if mask.dtype == np.uint8 else mask.astype(np.uint8)
    binary = _drop_fragments(binary)
    ys, xs = np.nonzero(binary)
    if len(ys) < _MIN_INK_PIXELS:
        return None
    cropped = binary[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    if cropped.shape[0] < 8 or cropped.shape[1] < 4:
        return None
    return cv2.resize(cropped, TEMPLATE_SIZE, interpolation=cv2.INTER_AREA)


def score_all(mask):
    """Every rank's correlation with `mask`, best first. Empty if unusable."""
    glyph = normalise(mask)
    if glyph is None:
        return []
    probe = glyph.astype(np.float32)
    scores = []
    for rank, exemplars in load().items():
        best = max(
            float(cv2.matchTemplate(probe, e.astype(np.float32),
                                    cv2.TM_CCOEFF_NORMED)[0][0])
            for e in exemplars
        )
        scores.append((rank, best))
    return sorted(scores, key=lambda rs: -rs[1])


def match(mask, min_score=MIN_SCORE):
    """The rank this glyph is, or None if nothing matches well enough."""
    scores = score_all(mask)
    if not scores or scores[0][1] < min_score:
        return None
    return scores[0][0]
