"""OCR helpers for nameplates, stacks, pot, badges, and status text.

Generic grayscale thresholding turned out to be unreliable on this UI: avatar
art (arbitrary photos/illustrations) shares the same crop as the text and
confuses both Otsu thresholding and Tesseract's page-segmentation. Instead we
key on the small, fixed palette this client actually uses for text --
white (names, badge letters, bubble text), warm orange/yellow (stack/pot
amounts), and red/pink (the "Folded" status label) -- and mask everything
else out before handing a clean black-on-white image to Tesseract.
"""
import re

import cv2
import numpy as np
import pytesseract

_STATUS_WORDS = ["Sitting Out", "All-in", "Check", "Call", "Raise", "Bet", "Fold"]
_BADGE_WORDS = ["BB", "SB", "D"]  # check 2-letter badges before the 1-letter one


def _channels(crop):
    b = crop[:, :, 0].astype(int)
    g = crop[:, :, 1].astype(int)
    r = crop[:, :, 2].astype(int)
    return b, g, r


def _mask_to_binary(mask):
    """mask: boolean array, True where text is. Returns black-text-on-white
    image for Tesseract."""
    return (~mask).astype(np.uint8) * 255


def white_mask(crop):
    b, g, r = _channels(crop)
    return _mask_to_binary((r > 150) & (g > 150) & (b > 150))


def name_stack_mask(crop):
    """White (names, 'Total Pot' label) OR warm orange/yellow (amounts)."""
    b, g, r = _channels(crop)
    white = (r > 150) & (g > 150) & (b > 150)
    orange = (r > 140) & (g > 90) & (b < 130) & ((r - b) > 40)
    return _mask_to_binary(white | orange)


def label_mask(crop):
    """White OR red/pink -- for persistent status text near an avatar,
    deliberately excluding orange since that overlaps common skin tones."""
    b, g, r = _channels(crop)
    white = (r > 170) & (g > 170) & (b > 170)
    red = (r > 150) & ((r - g) > 60) & ((r - b) > 30)
    return _mask_to_binary(white | red)


def _ocr(binary_img, whitelist=None, psm=6):
    config = f"--psm {psm}"
    if whitelist:
        config += f" -c tessedit_char_whitelist={whitelist}"
    return pytesseract.image_to_string(binary_img, config=config).strip()


def ocr_name(crop):
    if crop is None or crop.size == 0:
        return ""
    text = _ocr(name_stack_mask(crop), psm=6)
    text = re.sub(r"[^A-Za-z0-9_\-\. ]", "", text).strip()
    return text


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
    for psm in (6, 7, 11, 3):
        text = _ocr(mask, whitelist="0123456789.", psm=psm)
        match = re.search(r"\d+(\.\d+)?", text)
        if match:
            try:
                return float(match.group(0))
            except ValueError:
                continue
    return None


def ocr_badge(crop):
    """Return 'D', 'SB', 'BB', or None."""
    if crop is None or crop.size == 0:
        return None
    text = _ocr(white_mask(crop), whitelist="SBD", psm=6).upper()
    for word in _BADGE_WORDS:
        if word in text:
            return word
    return None


def _match_status_word(text):
    text_norm = text.replace("\n", " ").strip()
    for word in _STATUS_WORDS:
        if word.lower() in text_norm.lower():
            return word
    return None


def ocr_bubble_action(crop):
    """Transient action bubble text (Check/Call/Raise/Bet/Fold), or None."""
    if crop is None or crop.size == 0:
        return None
    text = _ocr(white_mask(crop), psm=6)
    return _match_status_word(text)


def ocr_persistent_label(crop):
    """Persistent status text (Folded/Sitting Out/All-in), or None."""
    if crop is None or crop.size == 0:
        return None
    text = _ocr(label_mask(crop), psm=6)
    word = _match_status_word(text)
    if word == "Fold":
        return "Folded"
    return word
