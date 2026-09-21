# Poker Vision

A vision layer that reads a PurePoker (mobile, portrait) practice-table
recording and extracts structured table state per frame: seat positions,
stack sizes, board/hole cards, pot size, and per-seat status.

## Setup

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
brew install tesseract   # if not already installed
```

## Usage

```
# Single screenshot
python -m poker_vision.main --image path/to/frame.png

# Video, sampled at 1 frame/sec, written to a JSONL file
python -m poker_vision.main --video path/to/gameplay.mp4 --fps 1 --out states.jsonl
```

Each output line is a JSON object:

```json
{
  "frame_index": 0,
  "timestamp_sec": 0.0,
  "pot": 12.8,
  "board_cards": ["7c", "3d", "6c", "9c", "2h"],
  "seats": [
    {"seat_id": "top_left", "name": "Balboni", "stack": 17.2,
     "position": "SB", "status": "Folded", "in_hand": false,
     "is_hero": false, "hole_cards": []},
    ...
  ]
}
```

`seat_id` is the fixed screen slot (top_left, top_right, mid_left, mid_right,
lower_mid_left, lower_mid_right, bottom_left, bottom_right, hero) -- stable
across hands, since players don't visually rotate seats in this client.

## Live mode (screen-shared phone)

`live_app.py` opens a live source, parses it in the background (parsing is
OCR-heavy so it doesn't run every frame), and shows an annotated view plus a
live-updating panel of everything it's reading. Press `q` to quit, `p` to
force an immediate re-parse.

```
python -m poker_vision.live_app --source <SOURCE> --parse-interval 1.5
```

**Getting your phone's screen into a `<SOURCE>` Python can read:**

Important: `python -m poker_vision.capture --list` (via ffmpeg) and OpenCV's
own device numbering **do not match** -- OpenCV's AVFoundation backend
silently excludes some device types (seen on this machine: it dropped "Desk
View Camera" and "Capture screen 0" entirely, leaving only 3 of ffmpeg's 5
devices). So always find the right index with the OpenCV-based probe tool,
never from the ffmpeg list:

```
python tools/probe_cameras.py
```

This opens every index 0-8 through the same backend `live_app.py` uses,
saves a snapshot from each one that opens, and prints their indices/shapes.
Open the snapshots in `frames/crops/probe/` and see which index (if any)
shows the poker table.

1. **Try the iPhone device first**: start QuickTime's "New Movie Recording"
   with the iPhone selected as camera (so Continuity Camera is actively
   streaming), then run the probe again and check that index's snapshot --
   it may show the phone's screen, or it may just show the physical camera
   lens's view. Either way you'll be able to tell from the image.
2. **OBS Virtual Camera (most reliable)**: in OBS, add a "macOS Screen
   Capture" or "Window Capture" source pointed at the QuickTime preview
   window so OBS is actually outputting that video, click "Start Virtual
   Camera", then re-run the probe -- the OBS Virtual Camera index's snapshot
   should now show the QuickTime window instead of the OBS placeholder logo.
3. **Screen region (no OBS needed)**: keep the QuickTime window in a fixed
   spot, run `python tools/select_region.py`, drag a box around just the
   phone screen inside the window, and use the `region:x,y,w,h` string it
   prints, e.g. `--source region:100,100,600,1300`.

Once you know the right index, run the live app with e.g. `--source camera:2`.

Whichever source you use, **the frame must be just the phone's poker
screen** (no window title bar / other chrome) for the calibrated regions in
`regions.py` to line up -- crop tightly in OBS/QuickTime or size the region
box precisely.

**First run**: macOS will ask for Camera and/or Screen Recording permission
for your terminal app the first time you run this -- grant both in System
Settings -> Privacy & Security, then re-run.

## How it works

- **`poker_vision/regions.py`** -- pixel regions for every seat's avatar,
  card slot, name/stack rows, position badge, and status area, calibrated
  against a 1206x2622 reference frame and rescaled to the actual frame size
  at runtime. Regenerate/verify with `tools/region_preview.py` and
  `tools/grid_overlay.py` if the client's layout changes.
- **`poker_vision/card_vision.py`** -- cards are recognized by (1)
  classifying the suit from the card's background color (this client color-
  codes suits: green=clubs, blue=diamonds, red=hearts, near-black=spades)
  and (2) OCR-ing the rank glyph in the card's corner. No image templates
  needed.
- **`poker_vision/ocr_utils.py`** -- text (names/stacks/pot/badges/status) is
  read by masking pixels to the small set of colors this UI actually uses
  for text (white / orange-yellow / red-pink) before handing a clean
  black-on-white image to Tesseract. Generic grayscale thresholding was
  tried first and was unreliable whenever a colorful avatar image shared the
  crop with the text.
- **`poker_vision/parser.py`** -- orchestrates the above into a `TableState`.

## Known limitations

- **Digit misreads**: Tesseract occasionally drops a decimal point or
  confuses a stylized digit (seen: "79.6" -> "796", a "1" misread as "4") on
  small/compressed stack-amount crops. A single frame's stack read isn't
  100% reliable. **Mitigation**: since a stack value is constant across many
  consecutive frames between actions, sample several frames per decision
  point and take the majority/median value rather than trusting one frame.
- **Action-bubble misses**: transient "Raise"/"Call"/"Check" bubbles are
  matched with a white-text color mask to avoid confusing them with avatar
  art, but this can still fail if a player's avatar has a lot of white
  clothing/highlights in the bubble zone.
- Calibration is specific to this PurePoker portrait layout (1206x2622). A
  different client, orientation, or major app redesign will need re-running
  the calibration tools in `tools/`.
