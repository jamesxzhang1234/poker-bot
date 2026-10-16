"""The in-process Tesseract must read exactly what the CLI did.

ocr_utils holds one engine open instead of spawning `tesseract` per call --
the spawn was ~54ms of every attempt. The two are only interchangeable if
they agree, and they did not at first: the CLI falls back to 70 dpi for an
image with no resolution, while the engine guessed its own. That changed 30
of 8,145 reads recorded from real frames.

This fixture has to be a real crop. Over 240 drawn glyph variants, none read
differently under the two resolutions, so a drawn fixture passes with or
without the fix. This one is the text mask of a stack where "14.5" sits
behind a "+1.45" bet; the CLI reads nothing in it, and without the pinned
resolution the engine invented a stack of "5".
"""
from pathlib import Path

import cv2

from poker_vision import ocr_utils

DATA = Path(__file__).parent / "data"


def test_in_process_engine_is_used_when_available():
    assert ocr_utils._api() is not None


def test_does_not_invent_a_digit_the_cli_does_not_see():
    mask = cv2.imread(str(DATA / "stack_mask_overlap.png"), cv2.IMREAD_GRAYSCALE)

    assert ocr_utils._ocr(mask, whitelist="0123456789.", psm=11) == ""


def test_a_whitelist_does_not_leak_into_the_next_read():
    # Variables persist on a held engine; the CLI started clean every time.
    mask = cv2.imread(str(DATA / "stack_mask_overlap.png"), cv2.IMREAD_GRAYSCALE)
    ocr_utils._ocr(mask, whitelist="SBD", psm=6)

    unrestricted = ocr_utils._ocr(mask, psm=6)

    # Unrestricted, the CLI reads this as "| ¢P as"; a leaked SBD whitelist
    # could only ever produce those three letters.
    assert set(unrestricted) - set("SBD \n")
