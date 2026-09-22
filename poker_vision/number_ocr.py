"""OCR for stack/pot/bet amounts.

Numeric crops need meaningfully more recovery logic than the other OCR
targets in ocr_utils.py -- Tesseract reliably drops the decimal point
(the glyph is only a few pixels tall against ~25px digit strokes) and, on
this font, sometimes renders a digit as a lookalike letter once the digit
whitelist forces it to choose. Both failure modes get fixed up here rather
than trusted to a single Tesseract pass; see the docstrings below for the
real misreads each fix was built against.
"""
import re

import cv2
import numpy as np

from .ocr_utils import _cascade, _ocr, name_stack_mask


def _text_components(mask):
    """Connected components of the dark (text) pixels in a black-text-on-
    white mask, split into digit-height glyphs and everything smaller,
    with decorative border components (spanning most of the crop's width --
    this UI draws a horizontal gold border across the whole stack-row crop)
    already excluded. Returns (digit_glyphs, small_components, max_h), each
    glyph/component as (x, y, w, h), glyphs sorted left to right.
    """
    ch, cw = mask.shape
    inv = (mask == 0).astype(np.uint8) * 255
    _, _, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    # w < cw*0.6 drops decorative borders spanning most of the crop width.
    # w >= 3 drops thin edge-artifact slivers (seen: a 1px-wide line taller
    # than the real digits, which -- being tallest -- corrupted max_h and
    # got miscounted as an extra digit glyph).
    comps = [(x, y, w, h) for i, (x, y, w, h, area) in enumerate(stats)
             if i > 0 and 3 <= w < cw * 0.6]
    if not comps:
        return [], [], 0
    max_h = max(h for (x, y, w, h) in comps)
    digit_glyphs = sorted([c for c in comps if c[3] >= max_h * 0.6], key=lambda c: c[0])
    small = [c for c in comps if c[3] < max_h * 0.35]
    return digit_glyphs, small, max_h


def _find_decimal_point(digit_glyphs, small, max_h):
    """Pick the decimal point out of `small`, disambiguating it from other
    small components that show up in the same size band: decorative UI
    chrome (sits well above the digit line) and a digit's own disconnected
    descender fragment (sits directly under that digit's own x-span).
    A true decimal point sits low, near the digit baseline, AND in the
    horizontal gap *between* two digit glyphs -- confirmed against a real
    misread where a border decoration got picked as "the dot" instead of
    the real one, because it was simply the first small component found.
    Returns the dot's x-coordinate, or None if no candidate fits.
    """
    if len(digit_glyphs) < 2:
        return None
    digit_bottom = max(y + h for (x, y, w, h) in digit_glyphs)

    def in_gap(cx):
        for i in range(len(digit_glyphs) - 1):
            left_end = digit_glyphs[i][0] + digit_glyphs[i][2]
            right_start = digit_glyphs[i + 1][0]
            if left_end < cx < right_start:
                return True
        return False

    for x, y, w, h in small:
        if y + h > digit_bottom - max_h * 0.5 and in_gap(x + w / 2):
            return x
    return None


def _insert_decimal(digit_text, digit_glyphs, dot_x):
    n_left = sum(1 for (x, y, w, h) in digit_glyphs if x < dot_x)
    if 0 < n_left < len(digit_text):
        return digit_text[:n_left] + "." + digit_text[n_left:]
    return digit_text


# This game's font renders some digits in a way Tesseract classifies as a
# letter instead, when constrained to the digit whitelist -- confirmed
# against real misread crops: '9' (a near-closed loop with only a short
# tail) read as O/Q ("9.55" whitelisted-OCR'd as just "55", unconstrained
# read "O55"), '4' read as A ("43.05" whitelisted-OCR'd as "205" --
# dropping the 4 entirely -- unconstrained read "A205"), '0' read as N,
# and '5' read as S (a well-known general OCR ambiguity, not specific to
# this font -- 5 and S share a similar curved-top shape) -- "40.5"
# whitelisted-OCR'd as just "5" alone (dropping BOTH the 4 and the 0),
# unconstrained read "ANS". Per-character bounding boxes (via
# pytesseract.image_to_boxes) confirmed A/N/S line up positionally with
# the three real digit glyphs, not just a coincidental same-length guess.
# The whitelisted pass then just drops the glyph rather than coercing it
# to the nearest digit, silently shortening the read. Substituting on the
# *whole-string* unconstrained read (rather than isolating and re-OCRing
# the single glyph, which loses the surrounding-text context Tesseract
# needs and fails outright even on an unambiguous glyph -- tried first,
# ruled out) recovers it.
#
# Known unrecovered case: '3' is sometimes read as '2' in this font, but
# unlike the confusions above, Tesseract gives '2' both with AND without
# the whitelist -- there's no distinguishing signal to substitute on, so
# a real '2' and a misread '3' are indistinguishable from OCR output
# alone. Not solved; would need a different approach (e.g. shape analysis
# of the glyph itself) if it turns out to matter enough.
_DIGIT_CONFUSIONS = str.maketrans({
    "O": "9", "Q": "9", "o": "9", "q": "9",
    "A": "4", "a": "4",
    "N": "0", "n": "0",
    "S": "5", "s": "5",
})


def _recover_dropped_digit(mask, digit_text, expected_digits, psm):
    """If the whitelisted read has fewer digits than there are actual
    digit-height glyphs in the mask, re-OCR the same crop unconstrained,
    substitute this font's known look-alikes for '9' back in, and use
    that instead if the *entire* unconstrained read becomes clean digits
    (+ optional dot) once substituted, accounting for every glyph.

    Requiring the whole string to match (not just finding a same-length
    digit run somewhere inside it via re.search) matters: a real misread
    once produced unconstrained text "B14" for a true "5.1" (2 real
    glyphs) -- re.search happily found "14" as a plausible-length digit
    run and returned it, silently fabricating a wrong number ("1.4")
    that looked just as plausible as the right one. "B" has no known
    confusion mapping, so requiring the whole string to resolve cleanly
    correctly rejects that case instead of guessing.

    Falls back to the original digit_text if the substituted read still
    doesn't fully resolve, rather than risk assembling a wrong number.
    """
    if len(digit_text.replace(".", "")) >= expected_digits:
        return digit_text
    raw = _ocr(mask, psm=psm)
    subbed = raw.translate(_DIGIT_CONFUSIONS)
    match = re.fullmatch(r"\d+(\.\d+)?", subbed)
    if match and len(match.group(0).replace(".", "")) >= expected_digits:
        return match.group(0)
    return digit_text


def _insert_geometric_decimal(mask, digit_text):
    """If Tesseract dropped the decimal point (common failure mode: the
    glyph is only a few pixels tall against ~25px digit strokes, so it
    gets discarded as noise -- confirmed by inspecting a real misread
    crop), find it directly as a small connected component instead of
    relying on Tesseract to recognize it as a character, and splice '.'
    into digit_text at the matching position. Returns digit_text
    unchanged if no such component is found (e.g. genuinely no decimal
    point, or this psm's read already includes one).
    """
    if "." in digit_text or not digit_text.isdigit():
        return digit_text
    digit_glyphs, small, max_h = _text_components(mask)
    dot_x = _find_decimal_point(digit_glyphs, small, max_h)
    if dot_x is None:
        return digit_text
    return _insert_decimal(digit_text, digit_glyphs, dot_x)


def ocr_number(crop):
    """OCR a stack/pot/bet amount and return it as a float, or None.

    Different crops (mostly a function of how much border/decoration
    surrounds the digits) respond best to different Tesseract page-
    segmentation modes -- empirically, no single psm is reliable across all
    of them. Try a short cascade and take the first plausible numeric read.
    """
    if crop is None or crop.size == 0:
        return None
    mask = name_stack_mask(crop)
    digit_glyphs, _, _ = _text_components(mask)
    expected_digits = len(digit_glyphs)

    def attempt(psm):
        text = _ocr(mask, whitelist="0123456789.", psm=psm)
        match = re.search(r"\d+(\.\d+)?", text)
        digit_text = match.group(0) if match else ""
        digit_text = _recover_dropped_digit(mask, digit_text, expected_digits, psm)
        if not digit_text:
            return None
        digit_text = _insert_geometric_decimal(mask, digit_text)
        try:
            return float(digit_text)
        except ValueError:
            return None

    return _cascade((6, 7, 11, 3), attempt)
