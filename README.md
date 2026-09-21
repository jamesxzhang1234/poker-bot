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

## Live mode (phone screenshot push)

Screenshots taken on the phone are pushed automatically to a local Flask
server, which parses each one as it arrives and prints the table state to
stdout.

```
python -m poker_vision.server
```

This listens on `http://0.0.0.0:8765/upload`.

**One-time iOS setup:** Settings -> Shortcuts -> Automation -> "When I Take
a Screenshot" -> action "Get Contents of URL", method POST, request body =
the screenshot. Turn off "Ask Before Running" so it fires silently. Point
the URL at `http://<your-mac's-local-ip>:8765/upload` -- find your Mac's IP
with `ipconfig getifaddr en0` (or whichever interface is active). If your
router supports a DHCP reservation for the Mac, use it so this URL doesn't
break when the IP changes.

**Testing without the phone:** POST a saved screenshot directly:

```
curl -X POST --data-binary @frames/frame_001.png http://localhost:8765/upload
```

You should see a parsed summary printed within a couple of seconds.

Whichever way a frame arrives, **the screenshot must be just the phone's
poker screen** (no browser chrome / other UI) for the calibrated regions in
`regions.py` to line up.

**First run**: macOS may prompt for local network permission for your
terminal app the first time the server receives a connection -- grant it in
System Settings -> Privacy & Security.

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
- **Recalibration after switching to native screenshots**: `regions.py` was
  originally calibrated against QuickTime-mirrored frames. Native iOS
  screenshots may differ slightly in resolution/cropping -- re-run
  `tools/region_preview.py` against a real screenshot and adjust
  `regions.py` if the region boxes don't line up.
