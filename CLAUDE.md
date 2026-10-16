# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A vision layer that reads a live PurePoker **practice / play-money** table off
a USB-attached **iPad** (portrait, 1640x2360) and emits structured table state
per frame: seat positions, stacks, board and hole cards, pot, per-seat status.
Class project. Not a real-money-site tool, and not intended to become one.

**This feeds a live decision engine.** Nothing consumes the state yet, but it
is being built to, and that sets the bar for every change here: a wrong value
is worse than a missing one, and a value that arrives after the decision is
worth nothing. Two rules follow, and most of the notes below are instances of
them.

- **Never guess to fill a gap.** A reader that cannot see something reports
  None and the engine can wait or fold; a reader that invents a plausible
  answer produces a confident bad decision with nothing to flag it. This is
  why `read_board` returns `[]` for an impossible board length, why
  `parse_amount` refuses "19.", and why both trackers blank rather than hold
  a stale value. It is also why `to_call` distinguishes 0 from None: "free"
  and "unknown" lead to opposite actions.
- **Know what each field costs.** See the latency budget below. The
  expensive readers are gated so they run once a hand rather than once a
  frame, and anything added to the per-frame path needs the same treatment.

**The iPad is the only device, permanently.** The iPhone is out of scope for
good — do not add iPhone support, calibrate for it, or keep code "working on
both devices". The retired iPhone push pipeline sits frozen in `archive/`
(see `archive/__init__.py`): don't extend it, don't fix it, and don't treat
its numbers as evidence for an iPad decision. Dependencies run one way only —
`archive/` imports from `poker_vision/`, never the reverse. An iPhone-sized
frame reaching the live path is skipped.

Some leftovers still sit in `poker_vision/` and are scheduled for removal,
not for use: `seat_anchors.IPHONE_PORTRAIT` (referenced only by a test that
checks the iPhone size is rejected), and helpers that only `archive/parser.py`
calls — `ocr_utils.ocr_name` / `ocr_badge` / `ocr_bubble_action` /
`ocr_persistent_label` / `white_mask` / `label_mask` / `badge_mask`, and
`card_vision.parse_card` / `read_rank` / `_ocr_rank` /
`_sample_bg_color`. Comments in live code that say "both devices" or quote
iPhone measurements are history: some thresholds (suit colours, the
fractional distance bands in `vision_reader`) were tuned to cover both
devices, still work on the iPad, and should not be retuned without an
iPad-only before/after sweep.

Two deliberate scope constraints that shape the code — don't "fix" them:

- **Single-frame only, with two named exceptions.** No cross-frame/session
  memory by default: players can leave mid-hand, so anything that remembers
  state across screenshots was judged more harmful than helpful.
  `infer_positions` works within one frame.

  The exceptions are `HeroCardTracker` and `PositionTracker`, and they are
  allowed for the same reason: both re-present a value that was *read* earlier
  in the same hand rather than inferring table state, both discard it at the
  next deal, and both prefer reporting nothing to reporting something stale.
  Anything that would hold a value across a hand boundary, or derive a new
  fact from remembered ones, does not qualify. They key on the same deal
  signal (the hero-card-region signature) because hole cards and positions
  turn over on exactly the same event. This is not a hand-history engine and
  should not grow into one. Before adding a third, check whether the problem
  can be solved at capture time instead -- the plate blank below was, and a
  third tracker built for it was removed.

  The pressure to add these comes from the client, not from convenience: it
  draws the hole cards fanned and the blind badges only pre-flop, so a
  strictly per-frame reader is blind to facts that are still true.
- **Capture: native AVFoundation off the USB-attached iPad, nothing else.**
  `tools/ipad_capture/capture` (Swift) opens the device, runs Apple Vision on
  each frame, and streams JSON lines to `poker_vision.vision_live`. The old
  screenshot-push path (iOS Shortcuts → Flask) is archived with the iPhone.
  Two capture facts, both measured rather than assumed (2026-09-24):

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
brew install tesseract            # OCR binary + traineddata; required for the pipeline AND the tests
pip install -r requirements.txt   # includes tesserocr: in-process Tesseract (see Latency budget)

python -m pytest tests/ -q                                   # the suite (~2s)
python -m pytest tests/test_vision_reader.py -q              # one file
python -m pytest tests/test_hero_cards.py::test_a_single_bad_first_read_is_never_shown -q   # one test
python -m pytest archive/ -q                                 # frozen iPhone pipeline's own tests, if you touch shared code it imports
```

Use `python -m pytest`, not bare `pytest` — there's no `pyproject.toml` /
`conftest.py` pinning `pythonpath`, so bare `pytest` can't import `poker_vision`.

```bash
# macOS only; needs the iPad unlocked and QuickTime closed
cd tools/ipad_capture && swiftc -O capture.swift -o capture    # build once
tools/ipad_capture/capture --live                     # stream JSON lines from the attached iPad
tools/ipad_capture/capture --live --save-dir DIR      # also save each emitted frame (corpus for backtests)
tools/ipad_capture/capture --dir frames/ipad          # replay saved frames (backtests, calibration)
tools/ipad_capture/capture --dir frames/ipad --fast   # .fast recognition: ~14ms vs ~90ms, fewer detections

python -m poker_vision.vision_live                    # live: spawn the helper, dashboard redrawn in place
python -m poker_vision.vision_live --blinds 1/2       # also writes frames/session.json (SB/BB) for a future consumer
python -m poker_vision.vision_live --mode player      # redraw only on your turn
python -m poker_vision.vision_live --min-interval 0.5 # throttle; --fast for .fast recognition; --text for scrolling output
tools/ipad_capture/capture --dir frames/ipad | python -m poker_vision.vision_live --stdin   # replay

python tools/grid_overlay.py <frame.png> <out.png>    # pixel grid for measuring geometry off a frame
```

`tools/region_preview.py` draws the archived iPhone `regions.py` boxes and is
broken (it imports a `poker_vision.regions` that no longer exists). Ignore it.

There's also a `start poker [SB/BB] [live|player]` zsh function in
`~/.zshrc`. It now runs the **live Vision path** with the in-place dashboard,
builds the Swift helper if it is missing, and kills any stale helper first
(only one process can hold the device). After the blinds it asks for a mode:
**live** redraws on every frame; **player** (`--mode player`) still parses
every frame -- the hole-card and position trackers must see the deal and the
blinds -- but redraws only on frames where `your_turn` is set, so the
snapshot holds still for checking accuracy by hand. A folded hand never
redraws. Detection checked on 54 saved frames: no false turns, 6 of 7 real
turns caught; the miss was the raise-sizing panel (`2x 3x 4x Confirm`), by
which point the turn had already drawn.

**Saved iPad frames** (1640x2360) live in `frames/ipad/` and
`frames/live_debug/`; capture more with `--save-dir`. Gitignored (personal
data) — use them for before/after verification of reader changes, never as
committed test fixtures. `PurePokerScreenshots/`, `frames/incoming/` and
`frames/frame_*.png` are **iPhone** frames (1206x2622) from the retired
pipeline: the live path skips them, so they are not evidence for an iPad
change.

## Architecture

### Data flow

```
iPad ──USB──▶ tools/ipad_capture/capture (Swift: AVFoundation + Apple Vision)
                │  JSON lines: recognised text + text_mask, board_strip, hero_strip, table thumbnail
                ▼
        vision_live.newest_first ── drain the pipe, keep only the latest frame
                │
        vision_reader.parse_payload ─▶ vision_live.layout_for(w, h) ── unknown size (e.g. iPhone) ──▶ skipped
                │
        vision_reader.build_table_state ── seats, stacks (+ recover_stack), pot, bets, status,
                │                          badges (+ find_mask_badges), infer_positions,
                │                          board (board_reader.read_board), your_turn / to_call
                ▼
        HeroCardTracker (read_hero_cards) / PositionTracker ── keyed on hero_signature
                │
        TableState ── is_plausible_table? ──no──▶ not shown as a table
                │ yes
        dashboard.render (or text_summary.format_summary with --text) ──▶ stdout
```

### Latency budget

Python side only; the helper adds ~90ms of Vision and PNG work per frame and
runs concurrently. Two corpora, because they disagree and the difference is
the point: `bt_live` is 24 quiet frames, `harvest` is 241 frames of real
seven-handed play.

| Stage | quiet | busy | When it runs |
|---|---|---|---|
| `parse_payload` (base64 + imdecode of 4 crops) | 8.8 ms | 8.8 ms | every frame |
| `read_board` (template matching, both candidates) | 3.6 ms | 3.6 ms | every frame |
| text → seats, pot, status, bets | ~13 ms | ~14 ms | every frame |
| `recover_stack` (Tesseract) | ~3 ms | ~11 ms | every frame, per missed stack |
| `read_hero_cards` | ~9 ms | ~9 ms | every pre-flop frame, via `HeroCardTracker` |
| `PositionTracker` | ~0 ms | ~0 ms | every frame |

**Tesseract now runs in-process, and that was the whole latency problem.**
Stack recovery used to dominate everything else combined: ~200-245ms per
call, because `number_ocr.ocr_number` runs a cascade of psm attempts and
`pytesseract` spawned a process for each one -- ~54ms of spawn against ~3ms
of recognition. `ocr_utils._ocr` now holds one `tesserocr` engine per thread
(falling back to the CLI if `tesserocr` is not installed). Measured on 30
iPad frames: per-frame parse p90 450ms -> 29ms, worst 479 -> 92ms; hero read
169 -> 9.4ms. The recovery itself is unchanged -- it still recovers stacks
Vision cannot see, and it is now cheap enough not to gate further.

**The two backends must read identically, and they did not at first.**
Every `_ocr` call made over all saved corpora (54 iPad frames through the
Vision path including hero reads, plus every iPhone screenshot through the
archived parser -- 8,145 calls) was recorded and replayed through both:
30 differed, deterministically, including an empty stack crop that read as
"5". The cause was resolution: the CLI reads pytesseract's temp PNG, which
has no dpi, and falls back to 70; in-process the engine picks its own.
Pinning `SetSourceResolution(70)` took it to **0/8,145**. The engine also
keeps variables between calls, so an unwhitelisted read must clear the
previous whitelist. `tests/test_ocr_backend.py` guards both, on a real crop
because no drawn glyph reproduces the dpi difference. If you change the
tesseract install or the wheel, re-run that replay.

Two more things to know before adding work here.

**Template matching was ~260x cheaper than CLI Tesseract** — 0.3ms against 78ms per
card (before the in-process engine; the gap is now far smaller). That is why ranks moved to templates. Reach for Tesseract only where
templates measurably fail, and gate it.

**The hero read is a spike that lands at the worst moment.** 169ms once a hand
sounds cheap until you notice the hand it fires on is pre-flop, which is
exactly when the first decision is needed. Inside it, the unconditional
ten-check (`_read_ten`) is 61 of the 78ms per card, because it runs an
unwhitelisted Tesseract pass before anything else gets a chance. That is now
avoidable: `isolate_rank_pair` screens for a two-character rank in ~0.3ms and
can decide whether the ten path is needed at all. Prototyped, not adopted — on
the labelled hands it held 16/16 rank accuracy, took the fast path on 8/16
cards, fired on zero non-tens and caught the one real ten; on the live corpus
the fanned template answers at >=0.90 on 40/47 crops. It reorders the most
accuracy-critical path, so it wants a bigger labelled set (one ten is not
enough) before it goes in.

### Module map (why each exists — the split is intentional)

Three readers share each frame, each used where the others measurably fail:
Apple Vision (in the helper) for free-form text, template matching for card
ranks, and Tesseract on the text mask for short numbers Vision drops.

| Module | Owns | Consumed by |
|---|---|---|
| `tools/ipad_capture/capture.swift` | Swift helper: sets the CoreMediaIO flag, opens the USB-attached iPad, runs one `VNRecognizeTextRequest` per frame, emits JSON lines with the text plus `text_mask`, `board_strip`, `hero_strip` and a table thumbnail. Rejects plate-blank frames, waits `--settle-ms`, gates on changed cells. `--live` captures, `--dir` replays saved frames. | vision_live |
| `vision_live.py` | Live runner: spawns the helper, turns JSON lines into `TableState`, drives the trackers and the dashboard. Picks the layout from the frame size and skips a size it doesn't know (`layout_for`). **No work queue on purpose**: this side is the slower half (53-216ms against the helper's ~90ms), so `newest_first` drains the pipe and renders only the latest, bounding lag at one parse. | — |
| `vision_reader.py` | **The reader.** `parse_payload` decodes a helper line; `build_table_state` makes the `TableState`: seats by proximity to anchors (no per-field boxes), stacks (+ `recover_stack`), pot, status, bets (+ `recover_bet`), badges (+ `find_mask_badges`), in-hand from `seat_has_cards`, `your_turn` / `to_call` from `find_action`, `table_obscured`, and the board via `board_reader`. Also `read_hero_cards` and `hero_signature` for `HeroCardTracker`. | vision_live |
| `seat_anchors.py` | iPad layout constants: `IPAD_PORTRAIT` (one (x, y) anchor per seat, clustered from real frames), `IPAD_BET_OFFSETS`, `IPAD_BOARD_SLOTS`, `IPAD_RANK_OFFSET`, `IPAD_HERO_CARDS`, `IPAD_HERO_CARD_SLOTS`. Still holds a dead `IPHONE_PORTRAIT`; don't use it. | vision_reader, vision_live, seat_layout |
| `seat_layout.py` | `SEAT_ORDER` — seat-ids in *action-order direction* (D→SB→BB→UTG…). Asserts at import that it matches `seat_anchors.IPAD_PORTRAIT`. | position_inference |
| `positions.py` | Canonical poker position names: `MIDDLE_POSITION_NAMES` (per table size) and `POSITION_ORDER` derived from it. Generic poker knowledge, layout-agnostic. | position_inference, text_summary |
| `position_inference.py` | `infer_positions` — names every occupied seat's position from the badges read in the same frame, skipping seats not dealt in (`_dealt_in`). | vision_reader |
| `board_reader.py` | Reads the community board: rank from `rank_templates`, suit from `card_vision.classify_suit` on `sample_card_color`. A matched rank is the presence test. | vision_reader |
| `rank_templates.py` | Card ranks by template matching against stored exemplars: `data/rank_templates.npz` (flat board cards) and `data/rank_templates_fanned.npz` (hero cards). Several exemplars per rank; a few were built from retired iPhone frames and are harmless extra candidates — don't prune without an iPad before/after sweep. | board_reader, card_vision |
| `card_vision.py` | Suit classification (`classify_suit`, `classify_hero_suit`), colour sampling, rank-glyph isolation (`isolate_rank_glyph`, `isolate_rank_pair`), and `parse_hero_card` (fanned template first, Tesseract fallback, ten check). `parse_card` / `read_rank` / `_ocr_rank` / `_sample_bg_color` are archive-only leftovers. | board_reader, vision_reader |
| `ocr_utils.py` | `name_stack_mask`, the single Tesseract call site `_ocr` (in-process `tesserocr`, CLI fallback), and `_cascade` (try psms in order, first non-None wins). `ocr_name`, `ocr_badge`, `ocr_bubble_action`, `ocr_persistent_label`, `white_mask`, `label_mask`, `badge_mask` are archive-only leftovers. | number_ocr, card_vision |
| `number_ocr.py` | `ocr_number`: geometric decimal-point recovery + digit-confusion recovery (`_DIGIT_CONFUSIONS`). Half the OCR complexity in the project lives here on purpose. | vision_reader (`recover_stack`) |
| `hero_cards.py` | `HeroCardTracker` — reads the hero's hole cards on every pre-flop frame, shows them once two consecutive reads agree, and blanks rather than risk a stale hand. One of the two sanctioned pieces of cross-frame memory. | vision_live |
| `position_tracker.py` | `PositionTracker` — holds the positions worked out while the blind badges were up, and drops them at the next deal. The other sanctioned piece of cross-frame memory. | vision_live |
| `table_state.py` | `Seat` / `TableState` dataclasses; `is_plausible_table` (≥4 seats with a stack ⇒ a poker table, not some other screen). | everything |
| `dashboard.py` | Renders a `TableState` as a fixed block rewritten in place, `*` marking what moved and `[HELD]` marking a stale one; `betting_line`. Pure: state in, string out. Holds the previous state **for display only** — that memory never reaches the parser. | vision_live |
| `text_summary.py` | Scrolling formatter (`--text`); seats sorted by `POSITION_ORDER`, suits as symbols. | vision_live, dashboard |
| `session.py` | Blind sizes typed in at launch, written to `frames/session.json` for a later consumer. | vision_live |
| **`archive/`** | **Frozen, retired iPhone pipeline** (Flask push server, fixed-region `parser.py`, iPhone-only `regions.py`, `main.py`, `overlay.py`, their tests). Imports from `poker_vision`; nothing imports it. Don't extend or consult it. | — |

### Things that only make sense if you know the history

- **Text is found by color, not brightness.** Every mask in `ocr_utils` keys on
  the UI's small text palette (white / orange-yellow / red-pink / badge
  mid-band). Plain grayscale/Otsu thresholding was tried first and broke
  whenever avatar art shared a crop with the text.
- **Every OCR reader tries several Tesseract `--psm` modes** via
  `ocr_utils._cascade`. No single psm was reliable across crops; the per-site
  psm *order* is empirically tuned and documented inline — don't "normalize" them.
- **`infer_positions` is deliberately conservative:** never overwrites an
  OCR'd position, never assigns a role another seat already holds, and only
  counts a seat as occupied if it has a name or stack (`in_hand` alone
  false-positives on empty seats).
- **`number_ocr` recovery is whole-string-match only.** A looser version once
  fabricated a plausible wrong number from noise. If a substituted read doesn't
  resolve to clean digits end-to-end, it falls back rather than guessing.
- **The iPad's text is small, despite the big frame.** It renders @2x onto an
  820×1180pt canvas, so its pixels buy UI area, not larger glyphs: measured
  over 30 native frames, stack digits are 24px median, 20.3% fall under
  Tesseract's usable ~20px floor and 16.2% under 16px. Small isolated glyphs
  are the recurring failure in every reader below.

- **Apple Vision replaced per-crop Tesseract on the capture path, and it is
  both faster and more accurate.** Backtested over 47 frames:
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

- **Card ranks are template-matched, not OCR'd, with several exemplars per
  rank.** The board is a closed 13-symbol alphabet in a fixed
  font, which is what OCR handles worst: Vision reads 2/4 board ranks on a
  whole frame and *1/4* on per-card crops (an isolated glyph in a small crop
  often isn't detected as text at all), and `card_vision`'s Tesseract path
  costs ~490ms a frame. Two traps found while building the templates:
  the rank crop catches the top of the suit pip on some cards, and that
  fragment stretches the bounding box so normalising squashes the digit -- a
  clean "5" scored 0.60 as "9" and never ranked "5" at all; and one exemplar
  per rank is not enough -- normalising does not erase differences in stroke
  weight and proportion, so a rank's exemplar can lose to a different rank on
  a clean card. Matching takes the best score across a rank's exemplars.
  Some exemplars came from the retired iPhone; they are extra candidates and
  harmless, so leave them unless an iPad before/after sweep says otherwise.
  Validated 15/16 cards and 7/8 boards against hand-checked frames.

- **`card_vision.is_card_present` false-positives on empty board slots.** It
  keys on edge density; on a pre-flop frame with no board at all it reported
  three cards, and a rank was then "read" off dark felt -- which is where the
  bogus 'T' samples in the first template harvest came from. The board does
  not use it (a matched rank is the board's presence test, see the showdown
  dimming note); only the hero-card read still calls it.

- **`classify_suit` tests the brightest channel, not the sum of all three.**
  The sum left no margin: across 163 real card backgrounds spades reach a sum of 107 against the old cutoff of 110, so the *same* iPad
  spade measured 107 in one frame and 116 in the next and read `4s` then
  `4d`. By brightest channel the classes are far apart -- spades top out at 55
  and the dimmest non-spade (a club) starts at 110 -- so the threshold sits in
  an empty gap instead of on top of the data. Before/after sweep: 163
  backgrounds, 8 changed, all `d`->`s`, all the same navy card across
  consecutive frames of one hand.

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

- **Board ranks: locate the glyph, don't assume a corner. Hero ranks: the
  opposite reader wins.** The board slots drift right of the cards by the
  fifth position, so the river's rank falls outside the fixed corner box and
  went unread -- costing the whole river card on a live hand. `best_rank` now
  tries the located glyph as well and keeps the best score; the river reads
  7@0.85 where the corner gave nothing, and every other slot scores higher
  too (0.85-0.99).
  **But do not carry that to hero cards.** Measured: on a fanned card the
  templates score 0.62-0.66 with a 0.01-0.03 margin over the runner-up, so
  right and wrong are indistinguishable, and Tesseract reads 20/21 of the
  hands checked by eye against the templates' 16. Preferring templates fixed
  one live 8-of-spades and broke three other cards. Board cards are flat and
  unobstructed; hero cards are fanned and overlapping. Same glyph shape,
  opposite answer.

- **A board is 0, 3, 4 or 5 cards. One or two is a misread, not a partial
  truth.** Mid-deal the flop slides in and the cards sit briefly offset from
  their slots; a real frame read "7c Qh" that way, two cards with one of them
  wrong. Rejecting by length costs nothing and needs no threshold -- the
  alternative was a confidence margin, which would have had to sit at 0.10 to
  catch it while a correct read scored 0.13, far too thin to trust.

- **Status labels arrive clipped, so match their start, not the whole
  phrase.** Vision returned `'Sitting (100'` -- a "Sitting Out (100 bb)" label
  cut short -- which matched nothing in the not-a-handle set and became a
  phantom player at the table.

- **At showdown the client dims the cards not in the winning hand, and one
  brightness cutoff cannot read both states.** A dimmed card's ink falls below
  any threshold that suits a normal one -- a real five-card board came back as
  a *single* card, because `read_board` stops at the first slot it cannot read
  and the second was dimmed. `best_rank` now tries 170, 140 and 110 and takes
  the highest-scoring match: bright cards score best around 140-170, dimmed
  ones only register at 110. The brightness presence gate was dropped at the
  same time; a matched rank is the better presence test, since felt, chips and
  the table watermark score 0.18-0.39 against the templates and never reach
  the 0.55 bar. Watch for this dimming generally -- it also broke hero suits.

- **Who is still in the hand comes from the cards beside a seat, not the
  "Folded" text.** The label is read on only 28% of seats, which is useless to
  anything downstream -- a decision needs the opponent count. A folded player
  simply has no cards next to their avatar, which is a visual fact: measured
  over labelled seats, a seat holding cards reads 7.3-9.5% mid-bright pixels
  in the box up-and-left of its plate against 0.0-2.7% folded, with nothing
  between. Every seat now gets an answer (14/14 correct on labelled seats,
  3.8 players per frame, a plausible poker number).

- **The helper's thumbnail must carry gamma-encoded luma.** It is drawn to
  RGBA and converted with OpenCV's weights, NOT rendered into
  `CGColorSpace.linearGray`. That colourspace applies a linear transfer
  function and crushes exactly the mid-tones this measures: the same frame
  came out with a mean of 5.6 against OpenCV's 28.1, every brightness
  threshold missed, and in-hand detection scored 4/14 instead of 14/14.

- **"Your turn" and the amount to call come from the bottom button row.**
  Neither was captured before and a decision engine needs both: one is the
  trigger, the other is half the pot odds. The catch is that two different
  button sets live on that row and mean opposite things -- `Call any`,
  `Check / Fold` and `Straddle` are the pre-select boxes shown while WAITING,
  while `Fold`, `Check`, `Bet` and `Call 0.25` mean it is your turn. Vision
  sometimes splits "Check / Fold" into separate Check and Fold tokens, which
  look exactly like the real buttons, so any pre-action word on the row
  settles it as not-your-turn whatever else is there.

- **Backtest with invariants, not just labelled frames.** Card reading was
  validated against ~10 hand-labelled frames and looked finished at 20/20.
  Running all 101 frames and checking rules that hold *without* labels found
  6 real errors immediately: **a card cannot be in the board and a hand at
  once**. The useful invariants are that one, no duplicate seat ids, at most
  one hero, at most five board cards and two hole cards, ranks and suits in
  range, and stacks non-negative. Ground truth on a few frames and invariants
  on all of them catch different things.

- **The board slots are wider than the cards, so a corner colour sample can
  land on felt.** The rightmost slot overhangs its card, and
  `_sample_bg_color` reads the top-right corner -- which for that slot is blue
  table, so the ace of spades read as a diamond. `sample_card_color` takes the
  median of the face instead, ignoring a minority of felt for the same reason
  it ignores the white pips. (`_sample_bg_color`, the corner sample, now
  serves only the archived `parse_card`.)

- **When hole cards clash with the board, report neither.** One of the two
  reads is wrong and nothing can tell which, so showing the pair presents an
  impossible hand as fact.

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

- **Neither suit rule may key on brightness. Two sampling regions, two
  rules, both scale-free.** The client dims cards when a hand ends, and that
  breaks any absolute level. On hero cards a dimmed diamond measured
  (60,40,23), inside the navy brightness range, so the board's old brightness
  test called it a spade; its blue is still 2.6x its red, matching a bright
  diamond's 2.8x, while hero navy sits at 1.47-1.54. The board rule was left
  on brightness at the time and later failed the same way: a dimmed club
  measured 79 and a dimmed diamond 78 against a spade cutoff of 80, and a
  showdown board read "As 7c Jc 6s 5s" for Ac 7c Jc 6s 8c. It now asks the
  dominant-channel questions first -- a club stays green-dominant and a heart
  red-dominant however far they dim -- and settles spades against diamonds on
  the difference b-r, not a ratio, because a ratio's margin proved too thin
  to trust while b-r leaves a wide gap (navy 7-25, diamonds 59-113). Hero
  suits: **20/20**, up from 14/20. Board suits: every one of 16 labelled boards, up from 5/16.

- **Derived pixel geometry is a guess; measure it off real frames.** The iPad
  board slots were derived as pitch 136 / width 118 and looked right, because
  the error only shows at the far end: the real pitch is 132 and width 125,
  so the fifth slot sat 17px right of its card. Single-character ranks
  survived the clipping and hid the bug; "10" lost its "1" and the bare "0"
  matched Q. Finding the card edges against the felt across seven real boards
  settled it in one pass. When a reader works on the left of a row and fails
  on the right, suspect accumulated pitch error before suspecting the glyph.

- **"10" is the only rank drawn as two characters, and every
  single-component assumption breaks on it.** `isolate_rank_glyph` returns
  one blob, so a board ten came back as its "1" and scored as a J;
  `_extract_rank` had to special-case the string; the fixed corner box catches
  both characters plus the suit pip, which is too large for the fragment
  filter. `isolate_rank_pair` supplies the two-character candidate to the
  board path. Whenever a rank change misbehaves, check the ten first.

- **The display drops stale frames; the pipe is not a queue you want.** The
  helper emits on every screen change, far faster than a frame parses (~129ms
  now, with board matching, stack recovery and hole cards). Consuming the pipe
  in order meant an unbounded backlog, so the lag grew without limit and the
  table on screen drifted minutes behind the game. `newest_first` drains what
  piled up and renders only the latest, which bounds the lag at one parse
  whatever the capture rate. `start poker` also throttles the helper to 0.5s,
  since each emitted frame costs it ~300ms of Vision and PNG work that the
  display cannot use. The throttle **holds** a frame that lands inside the
  window and reads it when the window closes, rather than dropping it: the
  device sends frames on change, so a bet followed by a still screen used to
  go unshown until something else moved. The helper stamps each frame with
  its capture time, and the dashboard footer shows `NNNms behind` -- capture
  to render -- which is the number to check when the display feels laggy. Measured live
  (2026-10-04, 14 frames): 200-360ms, of which Vision is 100-190, mask/strip
  encoding ~50, buffer conversion ~18, the pipe ~2 and the Python parse ~30.
  That is the floor of this design, not a backlog. Each payload carries
  `stage_ms` (held/convert/vision/encode) and `emitted_sec` so a slow frame
  can be pinned on a stage; one 1.6s frame was seen once and not reproduced. **An earlier note here said no queue was needed because
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

- **Badges are now read from the text mask too, and positions cover every
  frame (2026-10-04).** The note below said coverage was 80% and the rest
  not worth chasing; live, that left 2 of 4 hero turns with no positions,
  because Vision read no badge anywhere in those hands -- the badges were on
  screen, it just misses tiny isolated text. In the mask the badges are
  unmistakable: SB/BB a ~40px white disc with the letters cut out, the D
  button a ~42x28 flattened disc with a "D" cut out. `find_mask_badges`
  tells them apart by the islands inside each letter's hole (a B has
  counters, an S does not), which also rejects the look-alike "zZ"
  sitting-out marker. On 259 saved frames: a badge on 238; never disagreed
  with Vision's label (160 shared reads); 59/59 sampled detections right.
  Positions went from 164/237 frames to 237/237, hero turns 31/42 -> 42/42.
  Reading the button made the "not dealt in" bug visible and fixable: with
  two of D/SB/BB read directly, anyone seated between them cannot be in the
  hand, and "Sitting Out" seats are dropped too (`_dealt_in`). D/SB/BB were
  out of order on 17 frames before that, 0 after; the positions in between
  were checked by eye on 6 frames including a 9-handed table.
  The two notes below are kept for history.

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
  anchors `infer_positions`, which rotates the rest of the table into
  place. Wiring that up took iPad position coverage from 0% to 79%, and the
  inferred labels were checked against a screenshot (D=drake81, SB=trill04,
  BB=jxz). Badges are matched to a seat by **distance** rather than separate
  dx/dy bounds (a diagonal of 8.7-9.8% of the frame).

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
  hero's cards travel the same way, as `hero_strip`, and the binary
  `text_mask` for Tesseract recovery.

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

- **The client blanks a player's plate for one screen refresh as they act,
  and the helper rejects that frame before reading it.** Seen first as the
  player who acted just before the hero missing from the first frame of the
  hero's turn, which also shifted the inferred SB/D onto the wrong seats.
  Measured with `capture --live --min-interval 0 --save-only` (every frame,
  ~37 fps, 3 minutes): two blanks, **one frame each, back within 17-21ms** --
  invisible to the eye, which is why it looked impossible. The content-diff
  gate made it worse: a blank is a big sudden change, so the gate picked it
  out. `LiveCapture` now reads a frame only after the next one arrives, and
  rejects it if some regions went *dark* for that one frame and came straight
  back (64x64 luma grid, >=3 cells, 25 levels). Two approaches failed first:
  waiting for the screen to hold still for 35ms never triggers, because the
  turn timer animates constantly and a frame arrives every ~18ms, so 35 of 37
  reads fell through to a 100ms cap; and flagging any one-frame change
  flagged 158/6,749 frames, all chips and cards sliding over the felt, which
  *brighten* a region -- darkening alone flags 14 and catches both blanks.
  Live check (5 min, 8-handed): 21 frames rejected, 13 of them checked by
  eye as real blanks (cards beside the avatar, no plate) and the rest
  mid-table animation frames; 0 blank plates among the 31 frames actually
  read, against 2 of 46 before; added wait median 17ms, p90 28ms. With
  `--save-dir`, rejected frames are kept under `blinks/` for exactly this.
  On top of that, `--settle-ms` (default 30) waits that long after a change
  and reads the newest clean frame then, so bubbles and chips animating in
  are not caught half-drawn. Added at the user's request; not yet measured live.
  Cost: one device frame (~18ms). A `SeatTracker` that remembered the
  missing player was built first and removed: the problem is in *which*
  frame is read, and fixing that needs no memory and shows the real stack.
  Two other ways a player drops out, both still open: a deal or payout
  animation drawn over the handle (the blind chip flying in, the "+3.45" win
  burst, a dealt card), seen at the deal and payout only; and
  `infer_positions` counting a player who just sat down and is not dealt in
  as occupied, which shifts D onto them.

- **The content-diff gate must count changed cells, not sum a thumbnail --
  the summed one hid the action before the hero's.** The client lights the
  hero's buttons the instant the previous player acts, while that player's
  chips are still in flight and their bubble has not popped in; it all lands
  300-600ms later. The first frame was read, and the finished one was then
  dropped as "unchanged": the old gate summed a 16x16 `linearGray` thumbnail
  against 256, and a bubble plus a chip pile on dark felt scored 44-80 (the
  same mid-tone crushing as the thumbnail note above). Over 55 hero-turn
  frames in the saved captures the previous player's action was missing from
  nearly all, and appeared only on the next read, after the hero had acted.
  Replaying the 3-minute 37fps capture: the old gate made 26 reads, a read
  after the animation on 0/5 turns; >=6 cells of the 64x64 `lumaGrid`
  moving by >=40 made 110 reads, 5/5. Live (58s before the iPad was
  unplugged): 56 reads, capture-to-emit p50 234ms / p90 340ms, and the one
  hero turn got a second read 0.94s in with the previous player's "Fold" on
  it. The 30ms settle cannot do this job; the animation is 10-20x longer.

- **Actions and bets: what a 5-minute live capture got wrong, and the fixes
  (2026-10-04, checked over 241 saved frames, every change eyeballed).**
  - *Labels arrive damaged in two regular shapes.* "Folded" is half-covered
    by the stack badge (`Folde`, `Folder`, `Folde100`) and "Check" fuses with
    the blind badge (`Checkв`, `CheckB`). An exact match dropped 56 labels in
    58 frames. `_status_word` accepts exactly those shapes and nothing looser
    -- "Betty", "Callum" or a chat line starting "fold" must not become a move.
    146 Folded and 8 Check recovered in the sweep. A persistent label outranks
    a stale action bubble.
    Later live capture: the badge beside "Check" also arrives as Cyrillic or
    a digit ("Check зв", "Check зB", "Check3B") -- 6 of 66 check bubbles. The
    suffix is now folded through the badge lookalike table (Cyrillic ze
    added; the digit 3 only in this suffix, never for a standalone badge).
    Ground truth came from the bubble's colour, not from Vision: green blobs
    of bubble size in the saved PNGs. 65/65 real checks read after, 5/5 in a
    fresh live run; the one other green bubble was a stale Check left over
    the frame a player folded, where "Folded" is the right answer.
  - *An all-in player's stack is "0"*, a lone glyph Vision drops, so their
    seat exists only via stack recovery -- and `find_statuses` matched only
    plates whose stack Vision read. It now takes the assigned seats: 6 "All
    In" labels recovered. `is_in_hand("All In")` is True.
  - *Bets: one pile of chips, one owner, at a measured spot.* Each seat used
    to take the nearest amount in a 129-316px band independently, so one
    all-in 0.35 showed on two players (7 times), and face-up card ranks at
    showdown (~160px above a seat) became bets (13 removed). Bets now count
    only within 60px of the seat's spot in `seat_anchors.IPAD_BET_OFFSETS`,
    measured from 241 frames (one spot per seat, spread under 40px), and each
    amount goes to its nearest seat only.
  - *Lone-glyph bets are recovered* (`recover_bet`), like stacks: a raise to
    "1" read as "Raise" with no amount. The crop is not handed to Tesseract
    whole -- framing changed "0.1" into 0.71 or 9.1 -- but reduced to the one
    row of digit-shaped blobs, redrawn alone. Guards, each found by a wrong
    read: Vision saw no text there (a "+37.2" win burst), a chip icon above
    (a card back's logo mid-deal read as "6" and "2"; thumbnail patch 74-106
    above real bets, 38-52 above logos), glyphs of the bet font's 23-25px
    height, and two paddings agreeing. 7 recovered, 0 wrong after the guards;
    one true 0.05 is now declined, which is the right side to miss on.
  - *A PurePoker panel over the table* (chat, settings, profile, achievements)
    dims it: card backs vanish, everyone reads folded while their bubbles
    still read -- every "folded but checked" row in the capture. `obscured`
    is set when the felt is dark (<41; open tables 45-53) AND handles are dim
    (95th pct <170; open >=175, panels <=164). Both, because a different
    table theme darkens only the felt and the bomb-pot pop-up covers the
    middle while every seat stays readable. 21/21 flagged frames had a panel
    up, 0 clean frames flagged. Obscured frames skip the trackers (dimmed,
    the hero region jumps like a deal) and the display holds, marked.
  - *Raise -> re-raise in one frame* (`dashboard.betting_line`): within a
    round every bubble stays up and chips total each player's commitment, so
    the aggressive actions ordered by amount give the sequence. Once the
    opener calls, their 0.4 is gone from the screen and is not shown; that
    would need a per-street log, deliberately not built.
  - Still open: the hero's ten read as a 7 for two frames (pre-existing,
    same on HEAD); positions blank on some all-in frames.

- **Hole cards: 6 of 24 hands showed a wrong card; now 0, at the cost of
  ~one frame per hand.** 24 hands (354 card crops, 177 frames) labelled by
  eye from the screenshots. Four separate causes, four fixes:
  - *Tens read as 7 or 4* (8 of 22). `isolate_rank_pair` only tried the
    LEFTMOST blob as a ten's first half, and on a fanned card the suit pip
    sits left of the "1". Trying every blob finds 22/22 tens and 0/332
    false pairs, so a found pair now means "T" outright, ahead of the OCR
    ten check. Board reads unchanged on all 233 frames.
  - *Template bar too high.* Fanned-template matches >=0.70 were right
    201/201; Tesseract was wrong on 21 of them. `FANNED_MIN_SCORE` 0.90 ->
    0.75 (wrong matches top out at 0.68). A "4" exemplar was added -- the
    only labelled 4, so its recall on other 4s is unmeasured.
  - *Dimmed cards misread consistently* (a folded 4 as an ace, every
    frame). Lit cards peak at 233-255 grey, dimmed at 88-104; dimmed cards
    are no longer read. A folded hand needs no showing.
  - *First read latched.* `HeroCardTracker` now reads every pre-flop frame
    (affordable at ~9ms) and shows a hand only when two consecutive reads
    agree; a contradicting read hides it at once. Two contradictions before
    any board = a deal the signature missed (seen live: dimmed old cards
    made a deal arrive as 1714 + 1266, both under the 2000 threshold, and
    the previous hand stayed up 15 frames) and the agreeing pair replaces
    it; after a board, they only withdraw it -- the run-it-once false
    pre-flop must never install a hand.
  On lit frames: 82 right / 27 wrong / 1 blank before, 90 / 0 / 20 after;
  the blanks are the one-frame wait for agreement at the start of a hand.
  The earlier "read once at preflop, Tesseract 20/21" notes below predate
  this and describe the older design.

- **Mutation-testing here: this Python caches bytecode outside the tree.**
  The CommandLineTools Python sets `sys.pycache_prefix` to
  `~/Library/Caches/com.apple.python`, so deleting `__pycache__` does
  nothing. A mutation that moves code without changing the file size, saved
  within the same second, leaves a stale `.pyc` that the restored file then
  silently runs. Use `PYTHONDONTWRITEBYTECODE=1` when mutating.

- **Known unrecoverable OCR gaps:** `'3'` read as `'2'` (no signal to
  substitute on), an occasional `'5'` misread, and dim post-showdown hero-card
  renders. These are documented, not bugs to chase without new data.

## How to change a reader

This is the loop that has actually been finding bugs. It came out of a session
where a 241-frame live capture showed the board reader getting 5 of 16 boards
right while every test passed.

1. **Capture live frames before theorising.** The tests all passed and the
   backtest was clean; the bugs were only visible in volume. A few hundred
   frames of ordinary play is enough to make misses countable, and counting
   them is what tells you which bug is worth fixing.

2. **Label from the pixels, never from the reader's own output.** Crop the
   thing, look at it, write down what it is. I once labelled a board card
   `9s` by copying an earlier run's answer, then "fixed" a bug that did not
   exist and nearly reverted a correct read. If ground truth comes from the
   thing being tested, it is not ground truth.

3. **Measure geometry off real frames; do not derive it.** Every pixel
   constant that was computed rather than measured has been wrong. Find the
   edges against the felt across several frames and take the numbers. Derived
   geometry fails at the far end of a row, where a small pitch error has
   accumulated, so a reader that works on the left and fails on the right is
   the signature.

4. **Prefer adding a scored candidate to replacing a working path.** The ten
   was fixed by offering `isolate_rank_pair` alongside the existing glyph and
   keeping whichever scores higher. Single-character ranks produce no pair, so
   twelve of thirteen ranks could not regress. Replacing the path outright was
   tried once for hero ranks and cost 20/21 → 16/21.

5. **Separate "cannot read" from "read as nothing".** Count frames where
   something is plainly on screen and the reader returned empty. That number
   is the one that matters; total coverage hides it. It also distinguishes
   real misses from states the reader is *supposed* to reject, like the deal
   animation.

6. **Re-verify by eye after the fix, including the frames that changed.** A
   diff of before/after reads, rendered as images with both answers drawn on,
   catches the case where a number went up and an individual read got worse.

7. **Report before/after counts, not adjectives.** Every claim in this file
   has a number behind it because the ones that did not turned out to be
   wrong.

## Testing conventions

- **Synthetic fixtures for logic; real crops for rendering.** Most tests draw
  their own pixels (below). But a bug that exists *because* of how the device
  renders a glyph cannot be reproduced by a glyph you drew yourself — the
  first attempt at both the iPad ace and the board ten passed against the
  unfixed code, which proves nothing. Those two keep a real card crop in
  `tests/data/`, and the test says why it has to be real. Reach for this only
  when a drawn fixture provably cannot fail.
- **A fixture that does not fail first is not evidence.** Several drawn
  fixtures have been too forgiving to reproduce the bug: cards too plain to
  clear the edge-density gate, a glyph placed far enough from the card edge
  that the clipping under test did not touch it. Watch the test fail, and
  check the failure is the *bug* and not a typo or a missing import.
- **If a test passes the moment you write it, mutate the code to prove it can
  fail.** Break the specific guard it covers and confirm it goes red. Guard
  tests asserting `is None` are the usual offenders — they pass against a stub
  that never does anything.
- **Invariants need no labels and catch what eyeballing misses.** No card in
  both the board and a hand, ≤5 board cards, ≤2 hole cards, one hero,
  non-negative stacks, no duplicate seat ids, no role on two seats. Run these
  across every captured corpus after any reader change; the count should stay
  at zero. One invariant that looked obvious was wrong — see the note on bets
  and the pot — so a firing invariant means "check the invariant too".
- **No mocking anywhere.** Tests drive real code on synthetic pixel data:
  `test_number_ocr.py` draws glyph rectangles on a blank mask;
  `test_vision_reader.py` builds helper payloads from synthetic observations
  (`_obs`) and `cv2.putText` masks and strips at the iPad's 1640x2360, and runs
  them end-to-end through real Tesseract and template matching. Follow that
  pattern — if you need a controlled OCR result, draw a legible patch in the mask's color band, don't stub the reader.
- When changing OCR/calibration behavior, the established verification is a
  **before/after sweep across all saved iPad frames** (gitignored dirs above),
  counting how many hands' output changed and eyeballing that every change is
  a correction. Report those counts.

## Docs

- `docs/superpowers/specs/2026-09-24-vision-capture-design.md` is the design
  for the current capture path (native device capture + Apple Vision).
- `docs/superpowers/specs/2026-09-21-phone-screenshot-capture-design.md` and
  `docs/superpowers/plans/2026-09-21-phone-screenshot-capture.md` describe
  the retired iPhone push pipeline. History only.
- `README.md` is a short user-facing overview; this file is the detailed,
  current reference.
