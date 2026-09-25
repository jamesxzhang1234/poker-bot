# Native Device Capture + Apple Vision OCR — Design

Date: 2026-09-24
Status: Core implemented and backtested; integration points listed under "Not done"

## Background

Two separate questions got tangled together and were pulled apart by
measurement:

1. **Would an iPad improve OCR?** No. Its frame holds more total pixels, but
   it renders @2x onto an 820x1180pt canvas where the iPhone renders @3x onto
   402x874pt, so stack digits are *smaller*: 24px vs 26px median, with a worse
   sub-16px tail (16.2% vs 11.7%). More screen resolution buys more UI area,
   not more pixels per glyph.

2. **Could capture be hands-free?** Yes, and this was the real prize. Setting
   the CoreMediaIO property `kCMIOHardwarePropertyAllowScreenCaptureDevices`
   exposes a USB-attached iOS device to AVFoundation at its native resolution.
   The previous spec recorded this as impossible; exclusive access means *one
   consumer at a time*, not "never".

The QuickTime-mirror alternative was rejected on measurement: a mirror window
is bounded by the Mac display, dropping stack digits from 25px to ~17px and
putting 98% of them under Tesseract's usable floor.

A third finding then reframed the whole thing: the pipeline's cost was never
recognition. `pytesseract` spawns a process per call — 49.9ms of fixed
overhead against ~6ms of real work — and the parser made 129 calls per frame.

## Goals

- Hands-free capture: no manual screenshot per frame.
- Cut latency-to-decision from ~9.3s/frame to well under a second.
- Do not regress read accuracy.
- Add a device by writing down a handful of coordinates, not by recalibrating
  every field.

## Non-goals

- Interaction / actuation. Tracked as a separate workstream, deliberately
  kept apart: the vision layer ends at `TableState`, and an actuation layer
  would consume that without touching `regions.py` or the OCR modules.
- Replacing the push-based Shortcuts pipeline. It still works and is the
  fallback; this is a second source, not a removal.
- Cross-frame memory. The single-frame constraint is unchanged.

## Architecture

```
USB iOS device
   │  CoreMediaIO flag → AVFoundation .muxed device (native resolution)
   ▼
tools/ipad_capture/capture.swift
   │  VNRecognizeTextRequest per frame (~90ms), 16x16 luma diff gate
   ▼  JSON lines: {frame_index, timestamp_sec, width, height, source?, observations[]}
poker_vision/vision_reader.py
   │  parse_payload → find_seat_plates → assign_to_seats → build_table_state
   ▼
TableState  (same dataclass the existing parser produces)
```

Capture and OCR live in the same Swift process because the device only
appears to a client that sets the CMIO property, and Vision is a system
framework — doing both there means Python receives *located text* rather than
pixels, and the transport carries ~2KB of JSON instead of a 4MB frame.

### Why anchors instead of boxes

`regions.py` must know the exact rectangle of every field because it crops
before it reads. Vision reads first and reports where the text was, so the
Python side only needs a rough (x, y) per seat. Adding the iPad cost nine
coordinates; the equivalent under the old approach was every seat's five
zones plus a redone badge survey.

The anchors were derived by clustering name-above-stack pairings across real
frames rather than measured by hand. As a check, the iPhone anchors derived
this way landed within **4–23px** of the `name_row` centres `regions.py`
records from manual measurement — two independent methods agreeing.

## Results (backtest, 47 iPhone frames, both engines, same frames)

| | Tesseract | Vision |
|---|---|---|
| Latency / frame | 9,305 ms | **90 ms** (104x) |
| Stack reads | 236 | 209 |
| Pot read | 36 / 47 | 34 / 47 |
| Both read, agree | 166 | |
| Both read, disagree | 41 | |
| Stacks invented at an "Invite" seat | 4 / 109 | **0 / 109** |

Six disagreements were adjudicated by hand against the source images. **All
six went to Vision.** Most were Tesseract reading `3` as `2` — the failure
`CLAUDE.md` records as unrecoverable. Vision also returns a confidence score,
which is what suppresses its own misfires (it read card-back artwork as
Cyrillic at ~0.3 while real table text scores 1.0).

## Known limits

- **Vision reads ranks, not suits.** Card suits stay with `card_vision`'s
  colour classification. Board ranks came back 4/8 on a hand-checked frame,
  so cards are *not* handled by this path and `board_cards` is left empty.
- **Badges are intermittent** (~5/9). D/SB/BB still want a cropped read.
- **Occlusion is the dominant error source, not recognition.** A chip graphic
  over a nameplate, a badge over the last letter of "Folded", a bet-sizing
  panel over a seat. `parse_amount` refuses a trailing-separator read
  ("19.") rather than fabricate the hidden digits.
- **The device must be unlocked and QuickTime must be closed.** A locked
  device sits on USB configuration 5; capture needs configuration 6, which
  publishes the `Valeria` / `iOSScreenCaptureAssistant` interface. Both
  failure modes look identical to AVFoundation (an empty discovery list), so
  the helper cross-checks `ioreg` and reports which one it is.
- **`bottom_left` on the iPad is an unverified mirrored estimate** — no player
  ever sat there in the 30-frame sample.

## Not done

- Wiring `vision_reader` into `server.py` / `main.py`. The helper's stdout is
  not yet consumed by anything; `build_table_state` is called directly.
- The queue still has no latest-frame-wins policy. At 90ms/frame this is far
  less urgent than it was at 9.3s, but an unbounded `queue.Queue` fed by a
  live source can still back up.
- Status (`Folded`/`Check`/...) and `in_hand` are not populated by this path,
  though Vision does return the text.
- `infer_positions` is not wired in; `seat_layout.SEAT_ORDER` asserts against
  `regions.SEATS`, so an iPad seat order needs its own handling.
