# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A vision layer that reads a screenshot of a PurePoker **practice / play-money**
mobile table (portrait, 1206x2622) and emits structured table state per frame:
seat positions, stacks, board/hole cards, pot, per-seat status. Class project.
Not a real-money-site tool, and not intended to become one.

Two deliberate scope constraints that shape the code — don't "fix" them:

- **Single-frame only.** No cross-frame/session memory. Players can leave
  mid-hand, so anything that remembers state across screenshots was judged
  more harmful than helpful. `infer_positions` works within one frame.
- **Capture: push-based today, native device capture is the chosen next step.**
  An iOS Shortcuts automation POSTs every screenshot to a local Flask server;
  that is what `server.py` implements and it is still the only working path.
  Two corrections to the older note here, both measured rather than assumed
  (2026-09-24):

  - **Don't go back to the QuickTime-mirror / screen-region grab.** It isn't
    blocked — it's just bad. A mirror window is bounded by the Mac display, so
    the device screen is downscaled before you ever see a pixel. Measured over
    48 real frames, it drops stack digits from 25px to ~17px and puts 98% of
    them under Tesseract's usable ~20px floor (against 25% today). You cannot
    recover it in software afterwards: re-running the pipeline on upscaled
    frames *lost* more reads than it gained (236 → 229 → 223 stack reads at
    1.0/1.5/2.0×), because `ocr_utils`' masks threshold on hard colour cutoffs
    that interpolation softens.

  - **Direct AVFoundation capture works — the old "unworkable" note is wrong.**
    Exclusive access means *one consumer at a time*, not "never". With
    QuickTime closed, a process that sets the CoreMediaIO property
    `kCMIOHardwarePropertyAllowScreenCaptureDevices` can see and open the
    USB-attached device. Verified on macOS 26.4.1 with a ~50-line Swift helper:
    the iPad appears under mediaType **`.muxed`** — not `.video`, which is
    exactly why a plain `cv2.VideoCapture` sweep never finds it — as an
    `AVCaptureDeviceTypeExternal`, and delivers frames at the panel's native
    1640×2360 with no display in the loop.

## Commands

```bash
source venv/bin/activate          # Python 3.9 venv at ./venv
brew install tesseract            # OCR binary; required for the pipeline AND the tests

python -m pytest tests/ -q                                   # full suite (~4s)
python -m pytest tests/test_table_parser.py -q               # one file
python -m pytest tests/test_parser.py::test_infers_sb_and_bb_from_known_dealer -q   # one test
```

Use `python -m pytest`, not bare `pytest` — there's no `pyproject.toml` /
`conftest.py` pinning `pythonpath`, so bare `pytest` can't import `poker_vision`.

```bash
python -m poker_vision.server                 # live mode: listens on 0.0.0.0:8765/upload
python -m poker_vision.server --blinds 1/2    # also writes frames/session.json (SB/BB) for a future consumer
curl -X POST --data-binary @some_screenshot.png http://localhost:8765/upload   # test without the phone

python -m poker_vision.main --image frame.png                             # one-shot CLI
python -m poker_vision.main --video gameplay.mp4 --fps 1 --out states.jsonl

# Vision capture path (macOS only; needs the device unlocked and QuickTime closed)
cd tools/ipad_capture && swiftc -O capture.swift -o capture    # build once
tools/ipad_capture/capture --live                     # stream JSON lines from the attached device
tools/ipad_capture/capture --dir frames/ipad          # replay saved frames (backtests, calibration)
tools/ipad_capture/capture --dir frames/ipad --fast   # .fast recognition: ~14ms vs ~90ms, fewer detections

python -m poker_vision.vision_live                    # live: spawn the helper, print table state per frame
python -m poker_vision.vision_live --min-interval 0.5 # throttle; --fast for .fast recognition
tools/ipad_capture/capture --dir frames/ipad | python -m poker_vision.vision_live --stdin   # replay

python tools/region_preview.py <frame.png> <out.png>   # draw regions.py boxes over a frame to check calibration
python tools/grid_overlay.py ...                       # pixel grid for measuring new boxes
```

There's also a `start poker [SB/BB]` zsh function in `~/.zshrc`. It now runs
the **live Vision path** with the in-place dashboard, builds the Swift helper
if it is missing, and kills any stale helper first (only one process can hold
the device). The push server is still reachable as
`python -m poker_vision.server`.

**Real screenshots** live in `PurePokerScreenshots/` and `frames/incoming/`.
Both are gitignored (personal data) — use them for manual before/after
verification of OCR changes, never as committed test fixtures.

## Architecture

### Data flow

```
phone screenshot ──POST──▶ server.create_app  ──queue.Queue(path)──▶ server.worker_loop
                                                                        │
                                          TableParser.parse(frame) ◀────┘
                                                │
                              parse_seat × 9 seats ─▶ infer_positions ─▶ parse_board / parse_pot
                                                │
                                           TableState.to_dict()
                                                │
                     is_plausible_table? ──no──▶ "[skipped]"      yes ──▶ text_summary.format_summary ──▶ stdout
```

### Module map (why each exists — the split is intentional)

| Module | Owns | Consumed by |
|---|---|---|
| `regions.py` | **Pure pixel calibration.** `SEATS` (per-seat boxes), `BOARD_SLOTS`, `HERO_CARDS`, `POT_BOX`, `RANK_OFFSET`, `scale_box`, `status_sub_zones`. Nothing here has poker meaning. | parser, card_vision, seat_layout |
| `seat_layout.py` | `SEAT_ORDER` — this layout's seat-ids in *action-order direction* (D→SB→BB→UTG…). Asserts at import that it matches `regions.SEATS.keys()`. | `parser.infer_positions` |
| `positions.py` | Canonical poker position names: `MIDDLE_POSITION_NAMES` (per table size) and `POSITION_ORDER` derived from it. Generic poker knowledge, layout-agnostic. | parser, text_summary |
| `ocr_utils.py` | Color-keyed text masks (`white_mask`, `name_stack_mask`, `label_mask`, `badge_mask`), the single Tesseract call site `_ocr`, and `_cascade` (try psms in order, first non-None wins). Simple readers: `ocr_name`, `ocr_badge`, `ocr_bubble_action`, `ocr_persistent_label`. | parser, number_ocr, card_vision |
| `number_ocr.py` | `ocr_number` for stacks/pot: geometric decimal-point recovery + digit-confusion recovery (`_DIGIT_CONFUSIONS`). Half the OCR complexity in the project lives here on purpose. | parser |
| `card_vision.py` | Suit = background color class; rank = OCR of the corner glyph. Hero cards are fanned/rotated so `parse_hero_card` uses different crop/threshold strategies than flat `parse_card`. | parser |
| `parser.py` | `TableParser` (per-seat orchestration: name/stack/badge/status fallback/in_hand) and `infer_positions` (fills unread positions by rotating `SEAT_ORDER` from one OCR'd badge). | server, main |
| `table_state.py` | `Seat` / `TableState` dataclasses; `is_plausible_table` (≥4 seats with a stack ⇒ it's a poker table, not some other phone screen). | server |
| `text_summary.py` | stdout formatter; seats sorted by `POSITION_ORDER`, suits as symbols. | server |
| `server.py` | Flask upload endpoint + background worker thread, joined only by a `queue.Queue` of file paths. | — |
| `main.py` | Offline CLI over image/video files. | — |
| `vision_reader.py` | **The Vision path.** Turns the capture helper's JSON into a `TableState`: seats, stacks, pot, status, badges, and the board. Locates text by recognition, then assigns it to a seat by proximity — no per-field boxes. Reuses `parser.infer_positions` to fill unread positions. | vision_live |
| `seat_anchors.py` | One (x, y) anchor per seat, per device: `IPAD_PORTRAIT`, `IPHONE_PORTRAIT`. Derived by clustering real frames, not hand-measured. | vision_reader, vision_live |
| `rank_templates.py` | Card ranks by template matching against stored exemplars (`data/rank_templates.npz`). Several exemplars per rank, because one set does **not** serve both devices. | board_reader |
| `board_reader.py` | Reads the community board: rank from `rank_templates`, suit from `card_vision.classify_suit`. Has its own brightness-based card-presence test. | vision_reader |
| `rank_templates.py` (see above) and the text mask together mean **three** readers share the frame: Vision for free-form text, template matching for card ranks, Tesseract for short stacks. Each is used where the others measurably fail. | | |
| `hero_cards.py` | `HeroCardTracker` — reads the hero's hole cards once a hand at preflop and holds them. The only cross-frame memory in the pipeline; it reasons solely about whether a read is still valid, and blanks rather than risk showing a stale hand. | vision_live |
| `dashboard.py` | Renders a `TableState` as a fixed block rewritten in place, `*` marking what moved and `[HELD]` marking a stale one. Pure: state in, string out. Holds the previous state **for display only** — that memory never reaches the parser. | vision_live |
| `vision_live.py` | Live runner: spawns the capture helper, turns its JSON lines into `TableState`, prints `format_summary`. Picks the anchor set from the frame size and skips a size it doesn't know rather than guessing. **No work queue on purpose** — the helper spends ~90ms/frame and this side ~0.15ms, so the consumer cannot fall behind; frames are dropped in the helper, where the diff gate lives. | — |
| `tools/ipad_capture/capture.swift` | Swift helper: sets the CoreMediaIO flag, opens the USB iOS device, runs one `VNRecognizeTextRequest` per frame, emits JSON lines. `--dir` replays saved frames, `--live` captures. | — |
| `overlay.py` | **Dead code.** Nothing imports it (leftover from the deleted cv2-window UI). Safe to delete. | — |

### Things that only make sense if you know the history

- **Text is found by color, not brightness.** Every mask in `ocr_utils` keys on
  the UI's small text palette (white / orange-yellow / red-pink / badge
  mid-band). Plain grayscale/Otsu thresholding was tried first and broke
  whenever avatar art shared a crop with the text.
- **Every OCR reader tries several Tesseract `--psm` modes** via
  `ocr_utils._cascade`. No single psm was reliable across crops; the per-site
  psm *order* is empirically tuned and documented inline — don't "normalize" them.
- **`regions.SEATS[...]["badge"]` is always a list** of candidate boxes, even
  for seats with one. Badges physically relocate on several seats depending on
  whether the player holds cards (verified on 76 real sightings). Seats marked
  "mirrored / unverified" in `regions.py` have never had a real player in the
  sample data.
- **`infer_positions` is deliberately conservative:** never overwrites an
  OCR'd position, never assigns a role another seat already holds, and only
  counts a seat as occupied if it has a name or stack (`in_hand` alone
  false-positives on empty seats).
- **`number_ocr` recovery is whole-string-match only.** A looser version once
  fabricated a plausible wrong number from noise. If a substituted read doesn't
  resolve to clean digits end-to-end, it falls back rather than guessing.
- **A bigger screen is not more pixels per glyph.** The iPad was expected to
  improve OCR because its frame holds more total pixels than the iPhone's.
  Measured over 30 native iPad frames vs 47 iPhone screenshots, the iPad is
  *worse on every metric*: stack digits 24px vs 26px median, sub-20px rate
  20.3% vs 19.1%, sub-16px tail 16.2% vs 11.7%. The iPad renders @2x onto an
  820×1180pt canvas while the iPhone renders @3x onto 402×874pt, so its extra
  pixels buy more UI area, not larger text. Don't re-derive this by switching
  devices again — and note the corollary: **device choice and capture
  mechanism are independent.** Native AVFoundation capture works over USB for
  any iOS device, so hands-free capture does not require leaving the iPhone,
  and staying on the iPhone costs no recalibration.

- **Apple Vision replaced per-crop Tesseract on the capture path, and it is
  both faster and more accurate.** Backtested over the same 47 iPhone frames:
  **9,305ms -> 90ms per frame (104x)**. 166 stack reads agreed, 41 disagreed;
  six disagreements were adjudicated by hand against the images and **all six
  went to Vision**. Most were Tesseract reading `3` as `2` — the gap recorded
  below as unrecoverable, which Vision simply does not have. Tesseract also
  invented a stack at a seat showing "Invite" in 4 of 109 empty slots (Vision:
  0), always `4.0` at `top_right`, because the orange stack mask reads the
  orange **Raise button**. Why the speedup is so large: `pytesseract` shells
  out per call, and on a 10x10 blank image that round trip costs 49.9ms
  against ~6ms of actual recognition — 129 calls per frame meant ~6.4s of pure
  process overhead.

- **Card ranks are template-matched, not OCR'd, and one template set does not
  serve both devices.** The board is a closed 13-symbol alphabet in a fixed
  font, which is what OCR handles worst: Vision reads 2/4 board ranks on a
  whole frame and *1/4* on per-card crops (an isolated glyph in a small crop
  often isn't detected as text at all), and `card_vision`'s Tesseract path
  costs ~490ms a frame. Two traps found while building the templates:
  the rank crop catches the top of the suit pip on some cards, and that
  fragment stretches the bounding box so normalising squashes the digit -- a
  clean "5" scored 0.60 as "9" and never ranked "5" at all; and normalising
  does **not** make an exemplar device-independent, so an iPhone-built "8"
  lost to "3" on a clean iPad 8 while an iPad-built "10" scored 0.21 against
  iPhone tens. Hence several exemplars per rank. Validated 15/16 cards and
  7/8 boards against hand-checked frames.

- **`card_vision.is_card_present` false-positives on empty board slots.** It
  keys on edge density; on IMG_8425 (pre-flop, no board at all) it reported
  three cards, and `read_rank` then "read" a rank off dark felt -- which is
  where the bogus 'T' samples in the first template harvest came from.
  `board_reader.card_present` uses brightness instead: felt tops out around
  gray 92, a card face is far brighter.

- **`classify_suit` tests the brightest channel, not the sum of all three.**
  The sum left no margin: across 163 real card backgrounds on both devices
  spades reach a sum of 107 against the old cutoff of 110, so the *same* iPad
  spade measured 107 in one frame and 116 in the next and read `4s` then
  `4d`. By brightest channel the classes are far apart -- spades top out at 55
  and the dimmest non-spade (a club) starts at 110 -- so the threshold sits in
  an empty gap instead of on top of the data. Before/after sweep: 163
  backgrounds, 8 changed, all `d`->`s`, all the same navy card across
  consecutive frames of one hand, zero changes on the iPhone.

- **Hero hole cards: read at preflop, held, invalidated on appearance.** The
  fanned pair costs ~114ms to read and changes once a hand. An empty board is
  permission to read; a frame-to-frame jump in the hero region is what
  invalidates the held answer. Both conditions are needed and neither alone
  works. Board-only was tried: during a run-it-once showdown the board renders
  elsewhere, the fixed slots read a gap and return `[]`, and that false
  preflop triggered a mid-hand re-read that captured the wrong cards and held
  them all hand. Comparing the region against its signature *at read time* was
  also tried: the region drifts slowly as the bet slider and glow come and go,
  and the drift accumulated past the threshold and blanked cards that were
  still valid. Frame-to-frame is right because a deal is a sudden jump
  (measured 9471-9772) while animation is not (4-468). When the preflop frame
  is missed entirely, it shows **nothing** rather than the previous hand --
  blank is a failure you notice, wrong hole cards are not.

- **Tesseract reads hero ranks best; Vision and templates were both tried.**
  Measured on five hands read off the screenshots by eye, both cards each:

  | method | hands correct | per read |
  |---|---|---|
  | Tesseract (`read_rank_from_top`) | **5/5** | ~114ms |
  | Apple Vision on the hero strip | 1/5 | ~25ms |
  | rotated template matching | 2/5 | ~8ms |

  Vision found both ranks on one hand and nothing on two -- the same isolated
  glyph weakness that costs it board ranks, and the fan makes it worse.
  Template matching is fastest but scored 0.41-0.47 on its failures, meaning
  the glyph localiser grabs the wrong blob on a rotated card; deskewing the
  card first might fix it, but the speed it buys is not worth having. **The
  read happens once a hand, not once a frame** -- 4 reads across 30 frames,
  ~15ms per frame amortised -- so the slowest and most accurate option is
  effectively free. Don't swap it for a faster one without first changing
  what makes this cheap.

- **Cut the rank glyph out before reading it; don't hand OCR a window.** The
  suit pips are several times larger than the rank, so a window of the card's
  top gives Tesseract the glyph, the pips and a lot of empty felt, and it
  reads the pips as characters -- a 5 came back as a 3 because the window held
  a diamond pip bigger than the digit. `isolate_rank_glyph` takes the
  top-left-most glyph-sized component and reads that alone, which took hero
  cards from 17/20 to **20/20**. Its brightness cutoff is derived from the
  crop (`mean + 1.6*std`), not fixed: card faces run from navy through green
  to dimmed blue, and one constant finds the glyph on some and nothing on
  others. The window cascade is kept behind it as a fallback for cards
  isolation cannot handle.

- **A ten is invisible to a whitelisted read, and the cascade hides it.**
  The client draws a ten as the two characters "10", but `_RANK_WHITELIST`
  constrains Tesseract to single rank characters, so it can never report what
  it sees and picks the nearest single one instead. Worse, `read_rank_from_top`
  tries its smallest window first, and that window crops the glyph and yields
  a confident wrong letter before any larger window gets a chance -- a ten came
  back as an ace. `read_rank_from_top` now does one **unwhitelisted** read at a
  wide window first and calls it a ten if the result is two characters ending
  in a zero. On the real ten that read "40" (the 1 misread as a 4) while all 17
  other cards in the corpus produced nothing matching. Two other fixes were
  tried and rejected: deskewing the card by its minimum-area box made ranks
  *worse* (15/20 vs 16/20) because the angle estimate is unreliable, and a
  glyph width/height test failed because component extraction misses dim and
  green cards.

- **Hero suits go by channel ratio; board suits go by brightness. Two
  sampling regions, two rules.** The client dims cards when a hand ends, and a
  dimmed diamond measured (60,40,23) -- inside the navy brightness range, so
  the board's brightness test called it a spade. Its blue is still 2.6x its
  red, matching a bright diamond's 2.8x, while hero navy sits at 1.47-1.54:
  ratios survive dimming, absolute levels do not. The board rule cannot adopt
  this, because sampled from a clean corner its navy runs 2.25-2.5 against
  blue at 2.61 and ratios cannot separate those. Swept over 169 hero cards, 56
  suits changed, and against every hand checked by eye the result is **20/20
  suits correct**, up from 14/20.

- **The display drops stale frames; the pipe is not a queue you want.** The
  helper emits on every screen change, far faster than a frame parses (~129ms
  now, with board matching, stack recovery and hole cards). Consuming the pipe
  in order meant an unbounded backlog, so the lag grew without limit and the
  table on screen drifted minutes behind the game. `newest_first` drains what
  piled up and renders only the latest, which bounds the lag at one parse
  whatever the capture rate. `start poker` also throttles the helper to 0.5s,
  since each emitted frame costs it ~300ms of Vision and PNG work that the
  display cannot use. **An earlier note here said no queue was needed because
  python cost 0.15ms against the helper's 90ms. That stopped being true the
  moment this side started doing real work.**

- **A fanned card's suit comes from its left edge, not its centre.** The two
  hero cards overlap, so the one behind has its middle covered by the one in
  front and a centre sample averages the two: it called a 9 of spades a
  diamond, and called dimmed red queens spades. Swept over all 122 hero cards
  in the real screenshots, 13 suits changed and every one checked by eye was a
  correction.

- **The pot is cumulative across streets; the chips on the felt are only the
  current one.** They match pre-flop and diverge from the flop on, so "bets
  should equal the pot" is NOT an invariant -- built as one, it fired on 13 of
  30 frames. Only the other direction is a fault: bets totalling *more* than
  the pot is impossible, since the pot already contains them. Even that needs
  headroom, because chips land on the felt a frame before the pot absorbs
  them, so during the animation bets legitimately run about a blind ahead
  (2 of 30 frames). The check triggers past 2x the pot, which caught the
  hero's own hole card being counted as a 9.0 bet against a pot of 0.15.

- **Board card ranks and the hero's hole cards sit inside the bet distance
  band.** Both render mid-table or beside the hero, where committed chips
  live, so their rank glyphs read as amounts -- the 5 of spades became a 5.0
  bet for mid_left. `find_bets` excludes both regions outright; their
  positions are known, so there is no need to guess.

- **Vision misses short isolated numbers, and this is its main accuracy gap.**
  Adjudicating the 29 reads the Tesseract parser produced that the Vision path
  did not: **19 were one-character values** -- a stack of "5" or "0" where
  Vision read the handle above it and nothing else (`['BillWynn']`,
  `['kpk']`, `['MrTxx']`). It is the same isolated-glyph weakness that costs
  it card ranks and the bare "D" badge. It matters more here, because a stack
  of 0 or 5 is exactly the interesting case. **Fixed** by attaching a
  full-resolution binary text mask to the payload (~57KB base64, cheaper than
  it sounds because the mask is almost all background) and re-reading those
  stack rows with Tesseract -- which is slow per call but does read isolated
  glyphs, exactly Vision's weakness. It runs on a handful of crops per frame
  instead of 129, gated by a cheap shape check so the expensive call is only
  made where digits plausibly are. On the frame whose stacks were checked by
  eye this took the reading from 6/9 seats to 9/9. Of the rest: 4 were Tesseract reading the orange **Raise button** as
  a `4.0` stack at an empty seat, 2 were Tesseract reading a **Microsoft Teams
  notification** overlaying the screen (`24.087`, `1.1`), and 1 was Vision
  reading "IO" for 10, now folded.

- **Position coverage is 80% of frames, and that is 100% of the frames that
  matter -- don't go chasing the other 20%.** PurePoker shows SB and BB only
  while a hand is live; at showdown, run-it-once or a bomb pot it hides them
  and leaves the dealer button alone, or nothing. Measured over 30 iPad
  frames the split is exact: every frame with positions had 2-3 badges on
  screen, every frame without had **1 or 0**. So the gap is the UI not
  displaying the information, in states where the hand is already decided --
  not a detection failure. Recovering the lone "D" would reach ~97% (one
  frame shows no badge at all and never can be read) and would only label
  frames in which no decision is being made. Costed at ~110ms on 20% of
  frames; judged not worth it.

- **Vision never returns a bare "D".** A single character is too isolated for
  its text detector, so only "SB" and "BB" arrive. That is enough: one badge
  anchors `parser.infer_positions`, which rotates the rest of the table into
  place. Wiring that up took iPad position coverage from 0% to 79%, and the
  inferred labels were checked against a screenshot (D=drake81, SB=trill04,
  BB=jxz). Badges are matched to a seat by **distance** rather than separate
  dx/dy bounds -- the horizontal offset is ~100px on both devices, which is a
  different fraction of each one's width, while the diagonal fraction agrees
  (8.7-9.8%).

- **A locked device stays open and goes quiet; it does not disappear.**
  Tested by locking the iPad with capture running: USB configuration stays at
  6, the `Valeria` interface is still there, `AVCaptureSession` opens happily
  and reports "capturing" — and then delivers **zero** sample buffers. So a
  device-presence check never fires, which is why detection is a **frame
  watchdog**: an open session that has delivered nothing for 5s reports "no
  frames arriving -- unlock the iPad screen". Arrival is tracked *before* the
  content-diff gate, because that gate legitimately suppresses emission on a
  static table and starvation must not be confused with stillness.
  (An earlier note here claimed locking drops the device to configuration 5
  with no capture interface. That was wrong — config 5 was observed in some
  other state, probably before the device was unlocked at all.)

  Three display states follow, and they need different words: no frames /
  no device (unlock it, quit QuickTime), connected but no table on screen
  (you are in the lobby — nothing is wrong), and a table that has stopped
  updating (marked `[HELD]` with the reason). A frozen table that looks live
  is the only failure here that could actually be acted on.

- **Live capture supervises itself; it does not start once and hope.** The
  capture interface vanishes the instant the device locks -- iOS drops it back
  to a USB configuration that publishes none -- so a one-shot `startRunning()`
  dies silently the first time the screen sleeps. The helper polls every 2s:
  acquire when the device appears, notice when it goes, reacquire when it
  returns, carrying the frame counter across so indices stay monotonic. It
  also distinguishes "nothing plugged in" from "plugged in but not offering
  capture" by consulting `ioreg`, because AVFoundation reports both as an
  empty discovery list and the two need different fixes.

- **The helper sends text plus one small pixel strip, not the whole frame.**
  Everything except card ranks is read from recognised text, but ranks are
  template-matched and need pixels. Shipping the 4MB frame per capture would
  throw away the point of doing OCR in the helper, so it attaches just the
  board band (~120KB base64) as `board_strip`, with the crop rect it used in
  `board_strip_rect`. The consumer maps slots from that rect rather than
  keeping its own copy of the fractions, so the two sides cannot drift. The
  band works for both devices because the vertical span is nearly identical
  (0.417-0.483 of frame height on the iPad, 0.418-0.483 on the iPhone).

- **A live stdout stream must be line-buffered.** `vision_live` reconfigures
  stdout on startup: Python block-buffers when stdout is a pipe or file, so
  without it every captured frame sat unflushed and was lost when the process
  was killed -- the runner looked like it was capturing nothing while the
  helper was reporting frames fine.

- **Don't zip capture-helper payloads against a directory listing by
  position.** The helper skips any frame it cannot decode, so line N stops
  corresponding to file N. A backtest that did this silently compared each
  engine against a *different* frame and reported 72 disagreements where there
  were 41. Payloads carry a `source` field for exactly this reason; key on it.
  (`frames/incoming/20260921-175610-101256.jpg` is a 4-byte truncated upload —
  evidence the push pipeline can write corrupt files.)

- **Known unrecoverable OCR gaps:** `'3'` read as `'2'` (no signal to
  substitute on), an occasional `'5'` misread, and dim post-showdown hero-card
  renders. These are documented, not bugs to chase without new data.

## Testing conventions

- **No mocking anywhere.** Tests drive real code on synthetic pixel data:
  `test_number_ocr.py` draws glyph rectangles on a blank mask;
  `test_table_parser.py` builds a full 1206x2622 frame with `cv2.putText`
  patches in the right colors and calls `TableParser.parse_seat` end-to-end
  through real Tesseract. Follow that pattern — if you need a controlled OCR
  result, draw a legible patch in the mask's color band, don't stub the reader.
- Synthetic textures placed near a status zone must stay **below** the
  white/red mask thresholds (see `_draw_textured` in `test_table_parser.py`)
  or they bleed into the label crop as false text.
- `test_server.py` tests the queue seam from both sides: Flask `test_client`
  POSTs into `create_app`, and `worker_loop` driven directly with fake parsers.
  That's the one place fakes are appropriate (they stand in for the whole
  parser, not for an OCR call).
- When changing OCR/calibration behavior, the established verification is a
  **before/after sweep across all real screenshots** (gitignored dirs above),
  counting how many hands' output changed and eyeballing that every change is
  a correction. Report those counts.

## Docs

- `docs/superpowers/specs/` and `docs/superpowers/plans/` hold the design
  spec and implementation plan for the push-based capture pipeline.
- `README.md`'s "How it works" section predates `number_ocr.py`,
  `positions.py`, `seat_layout.py`, and the live server — the module map
  above is current.
