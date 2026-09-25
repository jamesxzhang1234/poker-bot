"""The iPhone push-capture pipeline, retired but kept runnable.

This was the whole project before native iPad capture: an iOS Shortcuts
automation POSTed screenshots to a Flask server, and a fixed-region parser
cropped calibrated pixel boxes out of each 1206x2622 frame and OCR'd them.
It is superseded by poker_vision.vision_live, which is ~100x faster and more
accurate, and the iPad is now the only supported device.

Kept rather than deleted so the iPhone path still exists if the cable or the
iPad does not, and because regions.py holds the only hand-measured pixel
calibration for that device -- expensive to reproduce.

Still runnable:
    python -m archive.server                  # Flask upload endpoint
    python -m archive.main --image frame.png  # offline CLI

Its tests run with the rest of the suite. Shared machinery (card_vision,
ocr_utils, number_ocr, positions, seat_layout, table_state, text_summary)
stays in poker_vision and is imported from here.
"""
