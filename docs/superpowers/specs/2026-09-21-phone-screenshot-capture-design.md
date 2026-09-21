# Phone Screenshot Capture Pipeline — Design

Date: 2026-09-21
Status: Approved (pending implementation)

## Background

The project previously captured the PurePoker table by mirroring the iPhone
into QuickTime Player and grabbing a fixed screen region (`poker_vision/capture.py`
`ScreenRegionSource`, driven by `poker_vision/live_app.py`). Direct camera capture
of the iPhone was ruled out earlier (macOS AVFoundation camera devices are
exclusive-access, so QuickTime holding the recording session blocks any
concurrent `cv2.VideoCapture` on the same device — see project memory).

This design replaces that whole video-capture layer with a push-based pipeline:
the user takes a native iOS screenshot of the table, and it is delivered
automatically to the Mac for parsing, with no QuickTime mirroring, screen-region
selection, or camera device probing involved.

## Goals

- Get each screenshot from phone to Mac automatically, with no manual
  transfer step per screenshot.
- Keep latency low and predictable (target: a couple of seconds from
  screenshot to parsed output).
- Reuse the existing parsing pipeline (`poker_vision/parser.py`,
  `card_vision.py`, `ocr_utils.py`) unchanged — this design only changes how
  frames arrive, not how they're parsed.
- No custom iOS app / Swift code — use built-in iOS Shortcuts automation.
- Keep the Mac-side addition small: no new heavyweight infra (no message
  broker, no database).

## Non-goals

- No decision/recommendation engine (fold/call/raise advice) — that is future
  work layered on top of the parsed `state_dict` output, out of scope here.
- No handling for phone/Mac on different networks — same-WiFi only for now.
- No retry/durability guarantees beyond "the file is on disk once the POST
  succeeds" — a screenshot taken while the server is down is simply missed.
- No authentication on the upload endpoint — trusted local-network use only.

## Architecture & data flow

1. User takes a screenshot on the phone during play (manual, whenever they
   want a frame captured — no fixed timer on the phone side).
2. An iOS Personal Automation, "When I Take a Screenshot" (with "Ask Before
   Running" disabled so it fires silently), runs a Shortcut whose action is
   "Get Contents of URL": POST to `http://<mac-local-ip>:8765/upload` with
   the screenshot image as the raw request body.
3. A new Flask server (`poker_vision/server.py`) receives the POST, writes
   the image to `frames/incoming/<timestamp>.jpg`, pushes the file path onto
   an in-process `queue.Queue`, and returns `200` immediately.
4. A background worker thread pulls paths off the queue one at a time, loads
   the image, runs the existing `TableParser.parse()`, and prints a compact
   human-readable summary (positions, stacks, cards, pot, status) to stdout.
5. Bursts (screenshots taken faster than OCR keeps up) queue up and are
   processed in arrival order rather than dropped.

## iOS side (one-time setup, no code)

Settings → Shortcuts → Automation → "When I Take a Screenshot" → action
"Get Contents of URL", method POST, request body = the screenshot. Disable
"Ask Before Running" so it fires silently.

The Mac's local IP must stay reasonably stable for the hardcoded URL to keep
working — a DHCP reservation on the router is the clean fix; otherwise the
Shortcut's URL needs manual updating when the IP changes. This is an
operational note, not something the code can fix.

## Mac side: new code

### `poker_vision/server.py`

Three pieces:

**1. Flask endpoint** — one route, `POST /upload`:
- Reads the raw request body (`request.get_data()`) — no multipart parsing
  needed since Shortcuts posts the image directly as the body.
- Writes it to `frames/incoming/<timestamp>.jpg` (timestamp includes
  microseconds to avoid collisions).
- Pushes the file path onto the queue.
- Returns `("", 200)` immediately, decoupling receipt from parsing.

**2. In-process queue** (`queue.Queue()`) — decouples "receiving a POST"
from "running OCR" so the HTTP response is instant even mid-parse. Holds
file paths (not raw bytes) to keep memory bounded and leave a disk record
for later debugging/re-running the parser against a specific frame.

**3. Background worker thread** — same shape as the `parse_worker` that
already exists in `live_app.py`, minus the shared-frame/lock machinery
needed for synchronizing with a live display (no display anymore):
- Loop: pull a path off the queue, `cv2.imread` it, run
  `TableParser().parse(frame).to_dict()`.
- On success: format and print via a new small `print_summary(state_dict)`
  helper (plain text — positions/stacks/cards/pot/status — not reusing
  `overlay.py`, which draws onto a cv2 canvas for the live window this
  design removes).
- On exception (parse error, or `cv2.imread` returning `None` for a
  corrupt/unreadable file): log `[parse error] ...` and continue the loop
  rather than crashing.

**Running it**: `python -m poker_vision.server` starts the worker thread,
then `app.run(host="0.0.0.0", port=8765)`.

**New dependency**: `flask`, added to `requirements.txt`.

**No auth** — trusted local-network use only, matching class-project scope.

## Removed components

Fully obsoleted by the push-based design and deleted:

- `poker_vision/capture.py` (`CameraSource`, `ScreenRegionSource`,
  `list_avfoundation_devices`, `open_source`)
- `poker_vision/live_app.py` (video display loop + old `parse_worker`)
- `tools/select_region.py` (drag-select a screen rectangle — no longer a
  capture source)
- `tools/probe_cameras.py` (camera device index probing — no longer needed)

Kept:

- `tools/grid_overlay.py`, `tools/region_preview.py` — still useful for
  calibrating `regions.py` regardless of how the frame arrived.

## Error handling & edge cases

- **Server restart**: `frames/incoming/` persists on disk, so nothing already
  received is lost across a restart. A screenshot taken while the server is
  down is simply missed (Shortcuts does not auto-retry) — acceptable for
  this project's scope.
- **Malformed/corrupt image**: `cv2.imread` returns `None` on a bad file;
  the worker checks for this and logs `[parse error] unreadable image`
  instead of crashing.
- **Parser exceptions**: wrapped in try/except (matching the existing
  pattern in the old `parse_worker`) so one bad frame doesn't kill the
  worker thread.
- **Port already in use**: Flask fails to start with a clear `OSError` — no
  special handling needed, the failure is immediately visible.
- **Wrong/stale Mac IP in the Shortcut**: fails silently from the phone's
  side (POST just fails/times out). No code fix — operational note to
  re-verify the IP if the network changes.
- **Disk fills up over a long session**: out of scope; `frames/incoming/`
  accumulates and can be cleared manually between sessions.

## Testing plan

1. Unit-level: none needed beyond what already exists for `TableParser` —
   this change doesn't touch parsing logic, only how frames arrive.
2. Manual integration test: start the server, POST a saved sample
   screenshot to `/upload` via `curl` (no phone needed), confirm it lands
   in `frames/incoming/` and prints a parsed summary.
3. End-to-end test: set up the Shortcuts automation, take a real screenshot
   of the PurePoker table, confirm it arrives and parses within a couple of
   seconds.
4. Recalibration check: since frames are now genuine iOS screenshots rather
   than QuickTime-mirrored frames, verify `regions.py`'s pixel boxes still
   line up — re-run `tools/region_preview.py` against a real screenshot and
   adjust if needed.
