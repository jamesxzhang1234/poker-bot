"""The retired iPhone push-capture pipeline. Frozen; do not extend.

This was the whole project before native iPad capture: an iOS Shortcuts
automation POSTed screenshots to a Flask server, and a fixed-region parser
cropped calibrated pixel boxes out of each 1206x2622 frame and OCR'd them.
It is superseded by poker_vision.vision_live. The iPhone is permanently out
of scope -- the iPad is the only device this project supports or will
support -- so nothing here should be fixed, extended, or used as evidence
for an iPad decision. Its pixel numbers (regions.py) are iPhone-only.

Dependencies run one way: this package imports shared machinery from
poker_vision (card_vision, ocr_utils, number_ocr, positions, seat_layout,
table_state, text_summary); nothing in poker_vision imports from here.

Still runnable, for reference:
    python -m archive.server                  # Flask upload endpoint
    python -m archive.main --image frame.png  # offline CLI
"""
