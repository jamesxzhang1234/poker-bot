import cv2
import numpy as np

from poker_vision import number_ocr


def _blank_mask(w, h):
    return np.full((h, w), 255, dtype=np.uint8)


def _draw_glyph(mask, x, y, w, h):
    cv2.rectangle(mask, (x, y), (x + w - 1, y + h - 1), 0, -1)


def test_text_components_ignores_a_thin_edge_artifact():
    # Reproduces a real misread: a 1px-wide vertical line artifact at the
    # crop's edge (taller than the actual digits) was getting counted as a
    # "digit glyph" purely because of its height, corrupting max_h and the
    # expected digit count for three real digit-height glyphs elsewhere.
    mask = _blank_mask(160, 40)
    _draw_glyph(mask, 1, 0, 1, 32)  # the artifact: 1px wide, tall
    _draw_glyph(mask, 20, 5, 17, 25)
    _draw_glyph(mask, 50, 5, 17, 25)
    _draw_glyph(mask, 80, 5, 17, 25)

    digit_glyphs, small, max_h = number_ocr._text_components(mask)

    assert len(digit_glyphs) == 3
    assert max_h == 25


def test_find_decimal_point_prefers_the_gap_between_digits_over_a_border_dot():
    mask = _blank_mask(180, 60)
    _draw_glyph(mask, 121, 6, 4, 2)  # decorative border dot, sits high up
    _draw_glyph(mask, 80, 30, 21, 25)
    _draw_glyph(mask, 118, 30, 19, 25)
    _draw_glyph(mask, 142, 30, 19, 25)
    _draw_glyph(mask, 107, 53, 4, 2)  # the real decimal point, low, in the gap

    digit_glyphs, small, max_h = number_ocr._text_components(mask)
    dot_x = number_ocr._find_decimal_point(digit_glyphs, small, max_h)

    assert dot_x == 107
