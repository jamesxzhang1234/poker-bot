# Poker Vision

A vision layer that reads a live PurePoker **practice / play-money** table off
a USB-attached **iPad** (portrait, 1640x2360) and emits structured table state
per frame: seat positions, stacks, board and hole cards, pot, bets, per-seat
status, whether it is your turn and the amount to call. Class project.

The iPad is the only supported device. An earlier iPhone pipeline is retired
and kept, frozen, in `archive/`; nothing in `poker_vision/` uses it.

## Setup

macOS only (the capture helper uses AVFoundation and Apple Vision).

```
python3 -m venv venv
source venv/bin/activate
brew install tesseract            # needed by the pipeline and the tests
pip install -r requirements.txt

cd tools/ipad_capture && swiftc -O capture.swift -o capture   # build the helper once
```

## Usage

Plug in the iPad, unlock it, open a PurePoker table, and make sure QuickTime is
closed (only one process can hold the device).

```
python -m poker_vision.vision_live                     # live dashboard, redrawn in place
python -m poker_vision.vision_live --blinds 0.05/0.1   # also record blind sizes
python -m poker_vision.vision_live --mode player       # redraw only when it is your turn
python -m poker_vision.vision_live --text              # scrolling summary instead of the dashboard

# Replay saved frames instead of capturing
tools/ipad_capture/capture --dir frames/ipad | python -m poker_vision.vision_live --stdin

# Save every frame read live, for later backtests
tools/ipad_capture/capture --live --save-dir frames/my_session
```

## How it works

```
iPad ──USB──▶ tools/ipad_capture/capture (Swift)
                AVFoundation capture, Apple Vision text recognition,
                plus a text mask and a few pixel strips (board, hero cards)
                    │ JSON lines on stdout
                    ▼
              poker_vision.vision_live ── newest_first: drop stale frames
                    │
              vision_reader.parse  ── seats, stacks, pot, bets, status, badges, board
                    │
              HeroCardTracker / PositionTracker  ── hold hole cards and positions within a hand
                    │
              TableState ──▶ dashboard (terminal)
```

- Text (names, stacks, pot, bubbles, buttons) comes from Apple Vision in the
  helper, and is assigned to a seat by distance from per-seat anchors in
  `seat_anchors.py`.
- Card ranks are template-matched (`rank_templates.py`); suits are classified
  from the card's background colour (`card_vision.py`).
- Short numbers Vision misses (a stack of "0" or "5") are re-read with
  Tesseract from the text mask (`number_ocr.py`, `ocr_utils.py`).

`seat_id` is the fixed screen slot (top_left, top_right, mid_left, mid_right,
lower_mid_left, lower_mid_right, bottom_left, bottom_right, hero), stable across
hands since this client doesn't rotate seats.

See `CLAUDE.md` for the measured design decisions, latency budget, and how to
change a reader safely.

## Tests

```
python -m pytest tests/ -q
```
