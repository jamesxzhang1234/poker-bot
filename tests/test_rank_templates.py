"""Tests for rank recognition by template matching.

The board renders 13 possible ranks in one fixed font at one fixed size, so
matching a glyph against a stored exemplar beats running an OCR engine on it:
Vision misses isolated glyphs (2/4 board ranks on a whole frame, 1/4 on
per-card crops) and Tesseract costs ~490ms a frame for the same job.
"""
import numpy as np

from poker_vision import rank_templates


def test_every_rank_has_at_least_one_exemplar():
    templates = rank_templates.load()

    assert sorted(templates) == sorted("23456789TJQKA")
    assert all(len(v) >= 1 for v in templates.values())


def test_every_exemplar_matches_its_own_rank():
    templates = rank_templates.load()

    for rank, exemplars in templates.items():
        for i, glyph in enumerate(exemplars):
            assert rank_templates.match(glyph) == rank, f"{rank}#{i}"


def test_a_template_still_matches_after_being_rescaled():
    # Real glyphs arrive at whatever size the device renders them; the iPad's
    # are larger than the iPhone's. Matching normalises before comparing, so
    # one template set serves both.
    import cv2
    templates = rank_templates.load()
    big = cv2.resize(templates["Q"][0], (96, 128), interpolation=cv2.INTER_NEAREST)

    assert rank_templates.match(big) == "Q"


def test_an_empty_mask_matches_nothing():
    assert rank_templates.match(np.zeros((32, 24), dtype=np.uint8)) is None


def test_noise_below_the_score_floor_matches_nothing():
    # An empty board slot is dark felt, not a glyph. Returning a confident
    # rank there would invent a card that isn't on the table.
    rng = np.random.default_rng(0)
    noise = (rng.random((32, 24)) > 0.5).astype(np.uint8) * 255

    assert rank_templates.match(noise) is None


def test_a_glyph_too_small_to_be_text_matches_nothing():
    tiny = np.zeros((32, 24), dtype=np.uint8)
    tiny[15:17, 11:13] = 255

    assert rank_templates.match(tiny) is None


def test_a_small_fragment_beside_the_glyph_is_discarded():
    # The rank crop can catch the top of the suit pip below the digit. That
    # fragment stretches the bounding box, so normalising squashes the real
    # glyph and it stops matching its own template -- measured on IMG_8437,
    # where a clean "5" scored 0.60 for "9" and never ranked "5" at all.
    import cv2
    templates = rank_templates.load()
    canvas = np.zeros((60, 40), dtype=np.uint8)
    canvas[2:34, 8:32] = templates["5"][0]
    canvas[50:56, 18:24] = 255          # pip fragment below the digit

    assert rank_templates.match(canvas) == "5"


def test_both_characters_of_a_ten_are_kept():
    # "10" is two glyphs of comparable size; fragment removal must not treat
    # either as noise.
    templates = rank_templates.load()
    canvas = np.zeros((44, 34), dtype=np.uint8)
    canvas[4:36, 4:28] = templates["T"][0]

    assert rank_templates.match(canvas) == "T"
