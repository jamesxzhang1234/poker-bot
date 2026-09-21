# Phone Screenshot Capture Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the QuickTime/screen-region video-capture pipeline with a push-based one: an iOS Shortcuts automation POSTs each screenshot to a local Flask server, which queues and parses it with the existing `TableParser` and prints a summary.

**Architecture:** A new `poker_vision/server.py` exposes `POST /upload`, which writes the incoming image to `frames/incoming/` and enqueues its path on an in-process `queue.Queue`. A background worker thread drains the queue, runs each image through `TableParser.parse()`, and prints the result via a new pure-function text formatter (`poker_vision/text_summary.py`). The old video-capture layer (`capture.py`, `live_app.py`, and the camera/region calibration tools) is deleted since it's fully superseded.

**Tech Stack:** Python, Flask (new dependency), OpenCV (`cv2`), existing `poker_vision` parsing modules, pytest (new dev dependency, for the first tests this project has).

**Spec:** `docs/superpowers/specs/2026-09-21-phone-screenshot-capture-design.md`

## Global Constraints

- No custom iOS app or Swift code — capture automation is iOS Shortcuts only.
- No new heavyweight infra — no message broker, no database, just an in-process `queue.Queue`.
- No authentication on `/upload` — trusted local-network use only.
- Same-WiFi assumption — no handling for phone/Mac on different networks.
- No decision/recommendation engine in this plan — only `TableParser`'s existing `state_dict` output, printed as text.

---

## Task 1: Text summary formatter

**Files:**
- Create: `poker_vision/text_summary.py`
- Test: `tests/test_text_summary.py`

**Interfaces:**
- Produces: `format_summary(state: dict) -> str`, where `state` matches the shape of `TableState.to_dict()` (`poker_vision/table_state.py:26-33`): keys `frame_index`, `timestamp_sec`, `pot`, `board_cards`, `seats` (list of dicts with `seat_id`, `name`, `stack`, `position`, `status`, `in_hand`, `is_hero`, `hole_cards`).

- [ ] **Step 1: Install pytest into the project venv**

```bash
cd ~/Desktop/Poker && source venv/bin/activate && pip install pytest
```

- [ ] **Step 2: Add pytest to requirements.txt**

Add a line `pytest` to `requirements.txt` (alongside the existing `opencv-python`, `numpy`, `pytesseract`, `pillow`, `mss`).

- [ ] **Step 3: Write the failing tests**

Create `tests/test_text_summary.py`:

```python
from poker_vision.text_summary import format_summary


def test_format_summary_includes_pot_and_board():
    state = {
        "frame_index": 3,
        "timestamp_sec": 12.5,
        "pot": 45.0,
        "board_cards": ["7c", "3d", "6c"],
        "seats": [],
    }
    out = format_summary(state)
    assert "frame 3" in out
    assert "Pot: 45.0" in out
    assert "Board: 7c 3d 6c" in out


def test_format_summary_lists_seats_with_details():
    state = {
        "frame_index": 0,
        "timestamp_sec": 0.0,
        "pot": None,
        "board_cards": [],
        "seats": [
            {
                "seat_id": "top_left",
                "name": "Balboni",
                "stack": 17.2,
                "position": "SB",
                "status": "Folded",
                "in_hand": False,
                "is_hero": False,
                "hole_cards": [],
            },
            {
                "seat_id": "hero",
                "name": None,
                "stack": 100.0,
                "position": "BB",
                "status": None,
                "in_hand": True,
                "is_hero": True,
                "hole_cards": ["Ah", "Kd"],
            },
        ],
    }
    out = format_summary(state)
    assert "Pot: ?" in out
    assert "Board: (none)" in out
    assert "Balboni" in out
    assert "[SB]" in out
    assert "$17.2" in out
    assert "(Folded)" in out
    assert "HERO" in out
    assert "Ah Kd" in out
```

- [ ] **Step 4: Run the tests to verify they fail**

```bash
cd ~/Desktop/Poker && source venv/bin/activate && python -m pytest tests/test_text_summary.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'poker_vision.text_summary'`.

- [ ] **Step 5: Write the implementation**

Create `poker_vision/text_summary.py`:

```python
"""Formats a parsed TableState dict (see TableState.to_dict()) as a compact
human-readable summary for the server's stdout output.
"""


def format_summary(state):
    lines = [f"--- frame {state['frame_index']} @ {state['timestamp_sec']:.1f}s ---"]

    pot = state.get("pot")
    lines.append(f"Pot: {pot if pot is not None else '?'}")

    board = state.get("board_cards") or []
    lines.append(f"Board: {' '.join(board) if board else '(none)'}")

    for seat in state.get("seats", []):
        tag = "HERO" if seat.get("is_hero") else seat.get("seat_id", "?")
        parts = [tag]
        if seat.get("name"):
            parts.append(seat["name"])
        if seat.get("position"):
            parts.append(f"[{seat['position']}]")
        if seat.get("stack") is not None:
            parts.append(f"${seat['stack']}")
        if seat.get("status"):
            parts.append(f"({seat['status']})")
        if seat.get("hole_cards"):
            parts.append(" ".join(seat["hole_cards"]))
        lines.append("  " + " ".join(parts))

    return "\n".join(lines)
```

- [ ] **Step 6: Run the tests to verify they pass**

```bash
cd ~/Desktop/Poker && source venv/bin/activate && python -m pytest tests/test_text_summary.py -v
```

Expected: 2 passed.

- [ ] **Step 7: Commit**

```bash
cd ~/Desktop/Poker
git add requirements.txt poker_vision/text_summary.py tests/test_text_summary.py
git commit -m "Add text_summary formatter for parsed table state"
```

---

## Task 2: Upload server (Flask endpoint + queue + worker)

**Files:**
- Create: `poker_vision/server.py`
- Test: `tests/test_server.py`

**Interfaces:**
- Consumes: `format_summary(state: dict) -> str` from `poker_vision/text_summary.py` (Task 1); `TableParser` (`poker_vision/parser.py:18`) with `TableParser().parse(frame, frame_index=0, timestamp_sec=0.0) -> TableState`, and `TableState.to_dict() -> dict` (`poker_vision/table_state.py`).
- Produces: `create_app(incoming_dir, work_queue) -> flask.Flask` (route `POST /upload`), `worker_loop(work_queue, parser_factory=TableParser, frame_index_start=0)` (blocks until `None` is read off `work_queue`, used by tests and as the target of the background thread in `main()`), `main()` (CLI entry point, wires everything together and calls `app.run(...)`).

- [ ] **Step 1: Add flask to requirements.txt and install it**

```bash
cd ~/Desktop/Poker && source venv/bin/activate && pip install flask
```

Add a line `flask` to `requirements.txt`.

- [ ] **Step 2: Write the failing tests**

Create `tests/test_server.py`:

```python
import queue

import cv2
import numpy as np

from poker_vision.server import create_app, worker_loop


def test_upload_writes_file_and_enqueues_path(tmp_path):
    work_queue = queue.Queue()
    app = create_app(tmp_path / "incoming", work_queue)
    client = app.test_client()

    resp = client.post("/upload", data=b"fake-image-bytes", content_type="image/jpeg")

    assert resp.status_code == 200
    files = list((tmp_path / "incoming").glob("*.jpg"))
    assert len(files) == 1
    assert files[0].read_bytes() == b"fake-image-bytes"
    assert work_queue.get_nowait() == str(files[0])


def test_upload_rejects_empty_body(tmp_path):
    work_queue = queue.Queue()
    app = create_app(tmp_path / "incoming", work_queue)
    client = app.test_client()

    resp = client.post("/upload", data=b"", content_type="image/jpeg")

    assert resp.status_code == 400
    assert work_queue.empty()


def test_worker_loop_logs_parse_error_for_unreadable_image(tmp_path, capsys):
    bad_path = tmp_path / "not-an-image.jpg"
    bad_path.write_bytes(b"not a real jpeg")
    work_queue = queue.Queue()
    work_queue.put(str(bad_path))
    work_queue.put(None)  # sentinel: stop the loop

    worker_loop(work_queue)

    out = capsys.readouterr().out
    assert "[parse error] unreadable image" in out


class _FakeState:
    def __init__(self, data):
        self._data = data

    def to_dict(self):
        return self._data


class _FakeParser:
    def parse(self, frame, frame_index=0, timestamp_sec=0.0):
        return _FakeState({
            "frame_index": frame_index,
            "timestamp_sec": timestamp_sec,
            "pot": 10.0,
            "board_cards": [],
            "seats": [],
        })


def test_worker_loop_prints_summary_for_valid_image(tmp_path, capsys):
    img_path = tmp_path / "frame.jpg"
    cv2.imwrite(str(img_path), np.zeros((10, 10, 3), dtype=np.uint8))

    work_queue = queue.Queue()
    work_queue.put(str(img_path))
    work_queue.put(None)

    worker_loop(work_queue, parser_factory=_FakeParser)

    out = capsys.readouterr().out
    assert "Pot: 10.0" in out


class _RaisingParser:
    def parse(self, frame, frame_index=0, timestamp_sec=0.0):
        raise ValueError("boom")


def test_worker_loop_logs_parse_error_when_parser_raises(tmp_path, capsys):
    img_path = tmp_path / "frame.jpg"
    cv2.imwrite(str(img_path), np.zeros((10, 10, 3), dtype=np.uint8))

    work_queue = queue.Queue()
    work_queue.put(str(img_path))
    work_queue.put(None)

    worker_loop(work_queue, parser_factory=_RaisingParser)

    out = capsys.readouterr().out
    assert "[parse error] boom" in out
```

- [ ] **Step 3: Run the tests to verify they fail**

```bash
cd ~/Desktop/Poker && source venv/bin/activate && python -m pytest tests/test_server.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'poker_vision.server'`.

- [ ] **Step 4: Write the implementation**

Create `poker_vision/server.py`:

```python
"""Flask server: receives screenshots pushed from the phone, queues them for
processing, and a background worker thread parses each one with
TableParser and prints a summary to stdout.

See docs/superpowers/specs/2026-09-21-phone-screenshot-capture-design.md.

Usage:
    python -m poker_vision.server
"""
import queue
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
from flask import Flask, request

from .parser import TableParser
from .text_summary import format_summary

DEFAULT_INCOMING_DIR = Path("frames/incoming")


def create_app(incoming_dir, work_queue):
    incoming_dir = Path(incoming_dir)
    incoming_dir.mkdir(parents=True, exist_ok=True)
    app = Flask(__name__)

    @app.route("/upload", methods=["POST"])
    def upload():
        data = request.get_data()
        if not data:
            return ("empty body", 400)
        ts = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        path = incoming_dir / f"{ts}.jpg"
        path.write_bytes(data)
        work_queue.put(str(path))
        return ("", 200)

    return app


def worker_loop(work_queue, parser_factory=TableParser, frame_index_start=0):
    """Pulls image paths off work_queue and prints a parsed summary for
    each. Put None on the queue to stop the loop (used by tests and by a
    clean shutdown)."""
    parser = parser_factory()
    frame_index = frame_index_start
    while True:
        path = work_queue.get()
        if path is None:
            break
        frame = cv2.imread(path)
        if frame is None:
            print(f"[parse error] unreadable image: {path}")
            continue
        try:
            state = parser.parse(frame, frame_index=frame_index, timestamp_sec=time.time())
        except Exception as exc:
            print(f"[parse error] {exc}")
            continue
        frame_index += 1
        print(format_summary(state.to_dict()))


def main():
    work_queue = queue.Queue()
    app = create_app(DEFAULT_INCOMING_DIR, work_queue)

    worker = threading.Thread(target=worker_loop, args=(work_queue,), daemon=True)
    worker.start()

    app.run(host="0.0.0.0", port=8765)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
cd ~/Desktop/Poker && source venv/bin/activate && python -m pytest tests/test_server.py -v
```

Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
cd ~/Desktop/Poker
git add requirements.txt poker_vision/server.py tests/test_server.py
git commit -m "Add push-based upload server with queue and parse worker"
```

---

## Task 3: Remove obsolete video-capture layer and update docs

**Files:**
- Delete: `poker_vision/capture.py`
- Delete: `poker_vision/live_app.py`
- Delete: `tools/select_region.py`
- Delete: `tools/probe_cameras.py`
- Modify: `README.md`

**Interfaces:** None — this task only removes dead code and updates documentation; nothing in Tasks 1-2 depends on these files (confirmed: `poker_vision/server.py` does not import `capture` or `live_app`).

- [ ] **Step 1: Delete the obsolete files**

```bash
cd ~/Desktop/Poker
git rm poker_vision/capture.py poker_vision/live_app.py tools/select_region.py tools/probe_cameras.py
```

- [ ] **Step 2: Replace the "Live mode" section of README.md**

In `README.md`, replace the entire section from `## Live mode (screen-shared phone)` through the end of the paragraph starting `**First run**: macOS will ask for Camera and/or Screen Recording permission...` (currently lines 47-100) with:

```markdown
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
```

- [ ] **Step 3: Add a recalibration note to Known limitations**

In `README.md`, under `## Known limitations`, add a new bullet:

```markdown
- **Recalibration after switching to native screenshots**: `regions.py` was
  originally calibrated against QuickTime-mirrored frames. Native iOS
  screenshots may differ slightly in resolution/cropping -- re-run
  `tools/region_preview.py` against a real screenshot and adjust
  `regions.py` if the region boxes don't line up.
```

- [ ] **Step 4: Verify no stale references remain**

```bash
cd ~/Desktop/Poker && grep -rn "capture\.py\|live_app\|select_region\|probe_cameras" README.md poker_vision/ tools/ 2>/dev/null
```

Expected: no output (the only remaining `capture` hit before this change was `poker_vision/overlay.py:1`'s docstring, which does not reference the deleted module — re-check that line stays as-is since it's an unrelated comment about "capturing" in the general sense).

- [ ] **Step 5: Run the full test suite to confirm nothing broke**

```bash
cd ~/Desktop/Poker && source venv/bin/activate && python -m pytest -v
```

Expected: all tests from Tasks 1-2 still pass (deleting `capture.py`/`live_app.py` doesn't touch anything they depend on).

- [ ] **Step 6: Commit**

```bash
cd ~/Desktop/Poker
git add -A
git commit -m "Remove obsolete video-capture layer, document push-based live mode"
```

---

## Task 4: End-to-end verification (manual, no code changes)

**Files:** None.

**Interfaces:** None — this task exercises Tasks 1-3 as a whole system.

- [ ] **Step 1: Start the server**

```bash
cd ~/Desktop/Poker && source venv/bin/activate && python -m poker_vision.server
```

Leave it running in this terminal.

- [ ] **Step 2: Test with a saved sample frame (no phone needed)**

In a second terminal:

```bash
cd ~/Desktop/Poker && curl -X POST --data-binary @frames/frame_001.png http://localhost:8765/upload
```

Confirm: the curl command returns immediately with no error, a new file appears in `frames/incoming/`, and the server's terminal prints a parsed summary (positions/stack/pot/board lines) within a couple of seconds.

- [ ] **Step 3: Set up the iOS Shortcuts automation**

Follow the "One-time iOS setup" instructions added to `README.md` in Task 3, Step 2, using your Mac's current local IP (`ipconfig getifaddr en0` on the Mac).

- [ ] **Step 4: Test with a real screenshot**

With the PurePoker table visible on the phone, take a screenshot. Confirm: within a few seconds, a new file appears in `frames/incoming/` on the Mac and the server prints a parsed summary.

- [ ] **Step 5: Check region calibration against the real screenshot**

```bash
cd ~/Desktop/Poker && source venv/bin/activate && python tools/region_preview.py frames/incoming/<latest-file>.jpg
```

(Use the actual filename from Step 4.) Confirm the drawn region boxes line up with the seats/cards/pot in the image. If they're off, adjust `poker_vision/regions.py` — this is exploratory follow-up work, not covered by this plan's tasks, since the fix depends on what's actually misaligned.

- [ ] **Step 6: Stop the server**

Press Ctrl+C in the server's terminal.

No commit for this task — it's verification only, and any region-calibration fixes found in Step 5 are a separate follow-up.
