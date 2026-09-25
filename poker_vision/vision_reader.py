"""Turns Apple Vision text observations into table state.

The Swift capture helper (tools/ipad_capture) runs one VNRecognizeTextRequest
over each captured frame and emits a JSON payload: every recognised string
with its pixel position and confidence. That is a different shape of input
from the rest of the pipeline, which crops fixed boxes out of a frame and
OCRs each one -- here the text arrives already located, so this module
assigns it to seats by proximity to a seat anchor instead of by calibrated
per-field boxes.
"""
import base64
import json
import re
from dataclasses import dataclass
from typing import List

import cv2
import numpy as np

from . import board_reader, card_vision, number_ocr, position_inference, seat_anchors
from .table_state import Seat, TableState


@dataclass
class TextObservation:
    """One recognised string and where it sits in the frame."""
    text: str
    x: int            # centre of the observation's bounding box, in pixels
    y: int
    w: int
    h: int
    confidence: float


@dataclass
class VisionFrame:
    frame_index: int
    timestamp_sec: float
    width: int
    height: int
    observations: List[TextObservation]
    # Set only in the helper's directory mode: the file this payload came
    # from. Key on it instead of on line order -- an undecodable frame is
    # skipped rather than emitted, so position does not survive a bad file.
    source: str = None
    # The board band as pixels, decoded. Card ranks are template-matched, so
    # that part of the pipeline needs pixels while everything else needs only
    # text -- the helper attaches this one small crop rather than the whole
    # 4MB frame. None when the helper sent text only.
    board_strip: "np.ndarray" = None
    # Where that crop came from, as (x, y, w, h) fractions of the frame.
    board_strip_rect: tuple = None
    # Full-resolution binary mask of the frame's text, white on black. Costs
    # ~23KB as a PNG because it is almost all background, which is cheaper
    # than the board strip while covering the whole frame -- and it is what
    # lets a stack Vision failed to detect be re-read from pixels.
    text_mask: "np.ndarray" = None
    # Quarter-scale greyscale of the whole frame, ~15KB. Carries the pixels
    # the text mask cannot: the face-down card backs beside each seat, which
    # are how you tell a folded player from a live one.
    table_thumb: "np.ndarray" = None
    table_thumb_scale: float = None
    # The hero's own cards as pixels. Separate from the board strip because
    # they sit at the bottom of the screen, nowhere near the board band, and
    # separate from the text mask because suit is a colour the mask discards.
    hero_strip: "np.ndarray" = None
    hero_strip_rect: tuple = None


def parse_payload(payload):
    """Parse one JSON payload (str or already-decoded dict) from the helper."""
    data = json.loads(payload) if isinstance(payload, (str, bytes)) else payload

    strip = None
    encoded = data.get("board_strip")
    if encoded:
        buffer = np.frombuffer(base64.b64decode(encoded), dtype=np.uint8)
        strip = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    rect = data.get("board_strip_rect")

    text_mask = None
    encoded_mask = data.get("text_mask")
    if encoded_mask:
        buffer = np.frombuffer(base64.b64decode(encoded_mask), dtype=np.uint8)
        text_mask = cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)

    hero = None
    encoded_hero = data.get("hero_strip")
    if encoded_hero:
        buffer = np.frombuffer(base64.b64decode(encoded_hero), dtype=np.uint8)
        hero = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    hero_rect = data.get("hero_strip_rect")

    thumb = None
    encoded_thumb = data.get("table_thumb")
    if encoded_thumb:
        buffer = np.frombuffer(base64.b64decode(encoded_thumb), dtype=np.uint8)
        thumb = cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)

    return VisionFrame(
        table_thumb=thumb,
        table_thumb_scale=data.get("table_thumb_scale"),
        hero_strip=hero,
        hero_strip_rect=tuple(hero_rect) if hero_rect else None,
        text_mask=text_mask,
        board_strip=strip,
        board_strip_rect=tuple(rect) if rect else None,
        frame_index=data["frame_index"],
        timestamp_sec=data["timestamp_sec"],
        width=data["width"],
        height=data["height"],
        source=data.get("source"),
        observations=[
            TextObservation(
                text=o["text"], x=o["x"], y=o["y"],
                w=o.get("w", 0), h=o.get("h", 0),
                confidence=o.get("confidence", 1.0),
            )
            for o in data["observations"]
        ],
    )


# A bare amount and nothing else: digits, optionally one decimal point with at
# least one digit after it. Anchored on both ends on purpose -- Vision returns
# plenty of strings that merely *contain* digits (the blind level
# "$0.05/$0.10", the clock, "100%", raise presets like "2x"), and treating any
# of those as a stack would invent money that isn't on the table.
_AMOUNT_RE = re.compile(r"^\d+(?:\.\d+)?$")

# Letters that are the same shape as a digit in this UI's font. Vision
# returned "IO" for a stack of 10 on a real frame, and one substitution costs
# the whole seat's stack. Folding happens before the match above, so the
# result still has to be a bare amount end to end -- "IOU" folds to "10U" and
# is refused, the same as any other word.
_DIGIT_LOOKALIKES = str.maketrans({"I": "1", "l": "1", "|": "1", "O": "0"})


def parse_amount(text):
    """Parse a recognised string as a table amount, or None if it isn't one.

    Deliberately strict. A string ending in a separator ("19.") is text that
    something on screen is covering -- a bet-sizing panel over a seat, most
    often -- and the digits we cannot see are exactly the ones that decide the
    value. Returning 19.0 there would be a confident wrong answer, so this
    refuses instead, the same way number_ocr refuses a read it cannot resolve
    end-to-end.
    """
    if text is None:
        return None
    candidate = text.strip().translate(_DIGIT_LOOKALIKES)
    if not _AMOUNT_RE.match(candidate):
        return None
    return float(candidate)


# A seat plate is a handle with its stack on the row immediately below. These
# bounds are fractions of the frame, not pixels, because the two devices we
# capture from differ in resolution but render the plate at nearly the same
# relative size: the name-to-stack gap is 2.2% of frame height on the iPhone's
# 1206x2622 and 2.3% on the iPad's 1640x2360. Expressing it this way is what
# lets one set of numbers serve both, and any future device whose UI scales.
_PLATE_GAP_MIN = 0.010   # fraction of frame height
_PLATE_GAP_MAX = 0.040
_PLATE_X_TOL = 0.060     # fraction of frame width

# Vision scores every string it returns. Real table text comes back at 1.0;
# the misfires do not. On the iPad frames it read the card-back artwork above
# a seat as Cyrillic ("УрАмі", "ЧрАмі") at ~0.3, and because that sits a
# plate's distance above the seat's stack it would otherwise invent a player.
# This floor is the whole defence against that, and it is a signal the
# fixed-region Tesseract path never had.
_MIN_CONFIDENCE = 0.5


# Strings the UI renders that are not player handles. Vision reads the whole
# screen, so the chrome comes back alongside the table data -- and some of it
# sits directly above an amount with exactly a seat plate's geometry. "Total
# Pot" above the pot is the one that actually bites; the rest are here so a
# status label or an action button can never be mistaken for a player.
_NOT_A_HANDLE = {
    "total pot", "pot", "invite", "nlh", "purepoker", "me", "all in",
    "folded", "fold", "check", "call", "bet", "raise", "show", "show all",
    "check / fold", "call any", "straddle", "confirm", "high card",
    "sitting out", "waiting", "dealer",
}

# Labels that arrive clipped. Vision returned 'Sitting (100' on a real frame
# -- a "Sitting Out (100 bb)" label cut short -- which matched nothing in the
# set above and became a phantom player at the table. Matching the start of
# the string catches it however the tail is mangled.
_NOT_A_HANDLE_PREFIX = ("sitting ", "waiting ", "show ", "total ")


def _is_handle(text):
    """Whether a recognised string could be a player's handle."""
    cleaned = text.strip().lower()
    if not cleaned or cleaned in _NOT_A_HANDLE:
        return False
    if cleaned.startswith(_NOT_A_HANDLE_PREFIX):
        return False
    # Raise presets ("2x"), the battery ("100%"), the turn timer ("+30s").
    if re.match(r"^[+-]?\d+(\.\d+)?\s*(x|%|s)$", cleaned):
        return False
    return True


# Text in the stack row that holds digits but is not a clean amount. That is
# what an occluded stack looks like: something on screen covers part of the
# number and Vision reads the visible fragment ("19." of "19.45"). Worth
# telling apart from ordinary chrome, because the seat is real even when its
# stack is not readable.
_HAS_DIGIT = re.compile(r"\d")


@dataclass
class SeatPlate:
    """A handle and the stack rendered directly beneath it.

    `stack` is None when the amount is there but unreadable -- covered by a
    bet-sizing panel, most often. The seat still exists, so it is still
    reported; dropping the plate would lose the player altogether, which is a
    worse answer than an unknown stack.
    """
    name: str
    stack: float
    x: int               # taken from the name row, the wider of the two
    y: int


def find_seat_plates(observations, width, height):
    """Find every name-above-amount pair that looks like a seat plate.

    Works purely off relative geometry, with no per-field pixel boxes, so it
    does not need recalibrating when the layout moves -- which is the whole
    reason this path exists alongside the fixed-region parser.
    """
    gap_min, gap_max = _PLATE_GAP_MIN * height, _PLATE_GAP_MAX * height
    x_tol = _PLATE_X_TOL * width

    trusted = [o for o in observations if o.confidence >= _MIN_CONFIDENCE]
    # Anything numeric-looking in the stack row: a clean amount, or a fragment
    # of one that something on screen is covering.
    stack_row = [
        (o, parse_amount(o.text))
        for o in trusted
        if parse_amount(o.text) is not None or _HAS_DIGIT.search(o.text)
    ]

    plates = []
    for candidate in trusted:
        if parse_amount(candidate.text) is not None:
            continue                      # an amount is never a handle
        if not _is_handle(candidate.text):
            continue
        below = [
            (o, v) for o, v in stack_row
            if gap_min <= (o.y - candidate.y) <= gap_max
            and abs(o.x - candidate.x) <= x_tol
        ]
        if not below:
            continue
        nearest, value = min(below, key=lambda ov: ov[0].y - candidate.y)
        plates.append(SeatPlate(name=candidate.text, stack=value,
                                x=candidate.x, y=candidate.y))
    return plates


def _amount_below(anchor, observations, width, height, gap_max=_PLATE_GAP_MAX):
    """The nearest amount rendered directly beneath `anchor`, or None."""
    gap_min = _PLATE_GAP_MIN * height
    below = []
    for o in observations:
        value = parse_amount(o.text)
        if value is None:
            continue
        if gap_min <= (o.y - anchor.y) <= gap_max * height and \
                abs(o.x - anchor.x) <= _PLATE_X_TOL * width:
            below.append((o.y - anchor.y, value))
    if not below:
        return None
    return min(below)[1]


def find_pot(observations, width, height):
    """The total pot, read from the amount under the "Total Pot" label.

    The label is what makes this unambiguous: several amounts float near the
    middle of the table (each player's committed chips this street), and only
    the one beneath the label is the pot.
    """
    for o in observations:
        if o.text.strip().lower() == "total pot":
            return _amount_below(o, observations, width, height)
    return None


# How far a plate may sit from a seat anchor and still be that seat, as a
# fraction of the frame's diagonal. Generous enough to absorb the plate
# drifting as a seat's decoration changes between hands, tight enough that a
# pairing found somewhere in the table chrome is dropped rather than snapped
# onto whichever seat happens to be least far away.
_ANCHOR_RADIUS = 0.055


def find_handles(observations, width, height):
    """Every observation that could be a player's handle, paired or not."""
    return [
        o for o in observations
        if o.confidence >= _MIN_CONFIDENCE
        and parse_amount(o.text) is None
        and _is_handle(o.text)
    ]


def plates_without_seat(plates, anchors, width, height):
    """Plates that landed too far from every anchor to be assigned.

    A non-empty result means this device's anchor set is missing a seat the
    layout actually has -- iPad `bottom_left`, for instance, is a mirrored
    estimate nobody has sat in yet. Dropping those silently hides the gap and
    shows up later as a player who is simply absent from the output.
    """
    radius = _ANCHOR_RADIUS * ((width ** 2 + height ** 2) ** 0.5)
    orphans = []
    for plate in plates:
        nearest = min(
            (((plate.x - ax) ** 2 + (plate.y - ay) ** 2) ** 0.5
             for ax, ay in anchors.values()),
            default=float("inf"))
        if nearest > radius:
            orphans.append(plate)
    return orphans


def assign_to_seats(plates, anchors, width, height):
    """Map seat plates onto seat ids by proximity to each seat's anchor.

    Returns {seat_id: SeatPlate} for the plates that landed near an anchor.
    Each seat keeps at most one plate -- the closest -- so a spurious pairing
    next to a real one cannot displace it.
    """
    radius = _ANCHOR_RADIUS * ((width ** 2 + height ** 2) ** 0.5)
    best = {}
    for plate in plates:
        for seat_id, (ax, ay) in anchors.items():
            distance = ((plate.x - ax) ** 2 + (plate.y - ay) ** 2) ** 0.5
            if distance > radius:
                continue
            if seat_id not in best or distance < best[seat_id][0]:
                best[seat_id] = (distance, plate)
    return {seat_id: plate for seat_id, (_, plate) in best.items()}


# Where a seat's stack row sits below its handle, as fractions of frame
# height, and how wide to reach either side as a fraction of frame width.
# Same band find_seat_plates pairs on, widened slightly because here we are
# cropping pixels rather than matching an already-located string.
_STACK_ROW = (0.008, 0.042, 0.075)


# A stack digit is about 1% of frame height (25px on the iPad's 2360, 26px on
# the iPhone's 2622). The band is generous either side; its job is only to
# reject crops that plainly hold no digits.
_DIGIT_H_MIN = 0.006
_DIGIT_H_MAX = 0.030


def looks_like_digits(crop, height):
    """Whether a mask crop plausibly contains digits.

    Recovery runs Tesseract at ~56ms a call and is attempted for every seat
    missing a stack, most of which have nothing to read. This is the cheap
    check that keeps the expensive one for crops that could actually pay off.
    """
    if crop is None or crop.size == 0 or not crop.any():
        return False
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        (crop > 0).astype(np.uint8), connectivity=8)
    lo, hi = _DIGIT_H_MIN * height, _DIGIT_H_MAX * height
    for i in range(1, count):
        h = stats[i, cv2.CC_STAT_HEIGHT]
        w = stats[i, cv2.CC_STAT_WIDTH]
        area = stats[i, cv2.CC_STAT_AREA]
        if not (lo <= h <= hi) or h == 0:
            continue
        # A digit is markedly taller than wide and solidly inked. Compression
        # artefacts in the mask clear the height test easily but not these:
        # they come out as sparse blobs of roughly square proportions.
        if not (0.15 <= w / h <= 1.2):
            continue
        if area < 80:
            continue
        return True
    return False


def recover_stack(text_mask, plate, width, height):
    """Re-read a seat's stack from the frame's text mask.

    For the case Vision cannot handle: a one- or two-character stack that its
    text detector does not report at all, even though it reads the handle
    directly above. 19 of the 29 stacks the fixed-region parser found and this
    path missed were single characters, so this is the main accuracy gap.

    Tesseract is used here on purpose, and only here. It is slow per call
    (~56ms, nearly all process overhead) but it does read isolated glyphs,
    which is exactly Vision's weakness -- and it runs on a handful of crops
    per frame rather than 129, so the cost stays small.
    """
    if text_mask is None:
        return None
    top_f, bottom_f, side_f = _STACK_ROW
    y1 = int(plate.y + top_f * height)
    y2 = int(plate.y + bottom_f * height)
    x1 = int(plate.x - side_f * width)
    x2 = int(plate.x + side_f * width)
    y1, x1 = max(y1, 0), max(x1, 0)
    y2, x2 = min(y2, text_mask.shape[0]), min(x2, text_mask.shape[1])
    if y2 <= y1 or x2 <= x1:
        return None
    crop = text_mask[y1:y2, x1:x2]
    if not looks_like_digits(crop, height):
        return None
    # ocr_number expects a colour crop and builds its own mask by keying on
    # white; a white-on-black mask satisfies that unchanged, so the existing
    # decimal-point and digit-confusion recovery all still applies.
    return number_ocr.ocr_number(cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR))


def build_table_state(payload, anchors, board_slots=None, rank_offset=None,
                      hero_cards=None):
    """Turn one Vision payload into a TableState.

    Seats, stacks, the pot and each seat's status come from the recognised
    text. Board cards do not: they are template-matched from the pixel strip
    the helper attaches, because an isolated rank glyph is the one thing
    whole-frame recognition reliably misses. Pass `board_slots` and
    `rank_offset` for the device to enable that; without them, or without a
    strip in the payload, board_cards stays empty.

    Position badges are still unset -- whole-frame recognition picks them up
    only intermittently and they want their own cropped read.
    """
    frame = parse_payload(payload)
    plates = find_seat_plates(frame.observations, frame.width, frame.height)
    assigned = assign_to_seats(plates, anchors, frame.width, frame.height)
    statuses = find_statuses(frame.observations, frame.width, frame.height)
    badges = find_badges(frame.observations, frame.width, frame.height)

    # A handle sitting on a seat anchor with no stack text beneath it is the
    # case Vision cannot see: it reads the name and simply does not report a
    # one-character stack below. Re-read those from the mask. Requiring BOTH
    # anchor proximity and a successful numeric read is what keeps a stray
    # word from becoming a phantom seat.
    if frame.text_mask is not None:
        radius = _ANCHOR_RADIUS * ((frame.width ** 2 + frame.height ** 2) ** 0.5)
        paired = {(p.name, p.x, p.y) for p in plates}
        for handle in find_handles(frame.observations, frame.width, frame.height):
            if (handle.text, handle.x, handle.y) in paired:
                continue
            for seat_id, (ax, ay) in anchors.items():
                if seat_id in assigned:
                    continue
                if ((handle.x - ax) ** 2 + (handle.y - ay) ** 2) ** 0.5 > radius:
                    continue
                candidate = SeatPlate(name=handle.text, stack=None,
                                      x=handle.x, y=handle.y)
                value = recover_stack(frame.text_mask, candidate,
                                      frame.width, frame.height)
                if value is not None:
                    candidate.stack = value
                    assigned[seat_id] = candidate
                break

    bets = find_bets(frame.observations, assigned, frame.width, frame.height,
                     board_slots=board_slots, hero_cards=hero_cards)
    action = find_action(frame.observations, frame.width, frame.height)

    seats = []
    for seat_id, plate in sorted(assigned.items()):
        key = (plate.name, plate.x, plate.y)
        status = statuses.get(key)

        # Whether a seat still holds cards is a visual fact -- they are beside
        # the avatar or they are not -- so it beats the status text, which is
        # read on only 28% of seats. Fall back to the text when no thumbnail
        # came with the frame.
        holding = seat_has_cards(frame.table_thumb, anchors[seat_id],
                                 frame.table_thumb_scale)
        if holding is None:
            holding = is_in_hand(status)
        stack = plate.stack
        if stack is None:
            stack = recover_stack(frame.text_mask, plate, frame.width, frame.height)
        seats.append(Seat(
            seat_id=seat_id,
            name=plate.name,
            stack=stack,
            position=badges.get(key),
            bet=bets.get(seat_id),
            status=status,
            in_hand=holding,
            is_hero=(seat_id == seat_anchors.HERO_SEAT_ID),
        ))

    # One badge is enough to place the whole table: infer_positions rotates
    # the occupied seats in action order from whichever seat has a read badge.
    position_inference.infer_positions(seats)
    board = []
    if board_slots and rank_offset is not None and frame.board_strip is not None \
            and frame.board_strip_rect is not None:
        strip = frame.board_strip
        local = slots_in_strip(board_slots, frame.board_strip_rect,
                               frame_size=(frame.width, frame.height),
                               strip_size=(strip.shape[1], strip.shape[0]))
        board = board_reader.read_board(strip, local, rank_offset)

    return TableState(
        frame_index=frame.frame_index,
        timestamp_sec=frame.timestamp_sec,
        seats=seats,
        board_cards=board,
        pot=find_pot(frame.observations, frame.width, frame.height),
        your_turn=action.your_turn,
        to_call=action.to_call,
    )


# Status text the UI renders near a seat. Two kinds, and the distinction is
# what tells you whether a player is still in the hand: a persistent label
# ("Folded", "Sitting Out", "All In") states a condition, while an action
# bubble ("Check", "Call", ...) reports the move they just made -- which only
# happens if they are still playing.
_PERSISTENT_STATUS = {"folded", "sitting out", "all in"}
_ACTION_STATUS = {"check", "call", "raise", "bet", "fold"}
_STATUS_WORDS = _PERSISTENT_STATUS | _ACTION_STATUS

# Vision reports whatever casing it read -- real frames produced both "Folded"
# and "folded" -- so statuses are reported in one canonical spelling and a
# consumer can compare them directly.
_CANONICAL_STATUS = {w: w.title() for w in _STATUS_WORDS}
_CANONICAL_STATUS["all in"] = "All In"

# How far above its plate a status can sit, as a fraction of frame height.
# Measured on both devices: the persistent label lands at 2.7-2.9% and the
# action bubble at 9.1-9.3% -- the same proportions on the iPad's 2360-high
# frame and the iPhone's 2622, which is why one band covers both. The lower
# bound also excludes the hero's action buttons along the bottom of the
# screen, which sit *below* every plate and otherwise read as a status.
_STATUS_DY_MIN = 0.025
_STATUS_DY_MAX = 0.098
_STATUS_DX_TOL = 0.070   # fraction of frame width


def find_statuses(observations, width, height):
    """Map each status label onto the seat plate it belongs to.

    Returns {(name, x, y): status} keyed by the plate's identity, so the
    caller can match it back to whichever seat that plate was assigned.
    """
    plates = find_seat_plates(observations, width, height)
    dy_min, dy_max = _STATUS_DY_MIN * height, _STATUS_DY_MAX * height
    dx_tol = _STATUS_DX_TOL * width

    out = {}
    for o in observations:
        if o.confidence < _MIN_CONFIDENCE:
            continue
        word = o.text.strip().lower()
        if word not in _STATUS_WORDS:
            continue
        below = [
            p for p in plates
            if dy_min <= (p.y - o.y) <= dy_max and abs(p.x - o.x) <= dx_tol
        ]
        if not below:
            continue
        plate = min(below, key=lambda p: p.y - o.y)
        out[(plate.name, plate.x, plate.y)] = _CANONICAL_STATUS[word]
    return out


# The dealer/blind badges. Note Vision never returns a bare "D" -- a single
# character is too isolated for its text detector -- so in practice only SB
# and BB arrive. That is enough: infer_positions rotates the rest of
# the table from any one badge.
_BADGE_WORDS = {"d": "D", "sb": "SB", "bb": "BB"}

# Cyrillic and Greek letters that render identically to the Latin ones the
# badges use. Vision returned "вB" on a real frame -- a Cyrillic ve for the
# first B -- and one wrong codepoint costs the whole table's positions, since
# a single badge is what anchors the inference. Folding is deliberately
# limited to characters that are visually identical: nothing here can turn a
# digit or an ordinary word into a badge.
_LOOKALIKES = str.maketrans({
    "\u0432": "b", "\u0412": "B",      # Cyrillic ve
    "\u0431": "b", "\u0411": "B",      # Cyrillic be
    "\u0455": "s", "\u0405": "S",      # Cyrillic dze
    "\u0501": "d", "\u0184": "b",      # Cyrillic komi de, Latin-ext
    "\u0392": "B", "\u03f2": "s",      # Greek beta, lunate sigma
})


def _badge_word(text):
    """The badge this string spells, folding lookalike letters, or None."""
    return _BADGE_WORDS.get(text.strip().translate(_LOOKALIKES).lower())

# A badge sits above its plate and off to one side, at 8.7-9.8% of the frame
# diagonal on both devices (measured: 249-282px on the iPhone's 2886px
# diagonal, 253-256px on the iPad's 2874px). Using distance rather than
# separate dx/dy bounds is what makes one number cover both -- the horizontal
# offset is ~100px on each device, which is a different *fraction* of width on
# each, while the diagonal fraction matches.
_BADGE_DIST_MAX = 0.120
_BADGE_DIST_MIN = 0.040


def find_badges(observations, width, height):
    """Map each D/SB/BB badge onto the seat plate it belongs to.

    Returns {(name, x, y): badge} keyed by the plate's identity.
    """
    plates = find_seat_plates(observations, width, height)
    diagonal = (width ** 2 + height ** 2) ** 0.5
    near, far = _BADGE_DIST_MIN * diagonal, _BADGE_DIST_MAX * diagonal

    out = {}
    for o in observations:
        if o.confidence < _MIN_CONFIDENCE:
            continue
        badge = _badge_word(o.text)
        if badge is None:
            continue
        # Every badge renders above its own seat, so a plate above the badge
        # belongs to someone else.
        candidates = [
            (((p.x - o.x) ** 2 + (p.y - o.y) ** 2) ** 0.5, p)
            for p in plates if p.y > o.y
        ]
        candidates = [(d, p) for d, p in candidates if near <= d <= far]
        if not candidates:
            continue
        _, plate = min(candidates, key=lambda dp: dp[0])
        out[(plate.name, plate.x, plate.y)] = badge
    return out


# How far a player's committed chips sit from their seat anchor, as a fraction
# of the frame diagonal. Measured on real frames: bets land 165-270px out on
# the iPad's 2874px diagonal (5.7-9.4%). The floor matters as much as the
# ceiling -- each seat carries a small count 85-98px above its plate, and
# without it every seat would report a bet it never made. The ceiling keeps
# the pot out, which sits ~400px away.
_BET_DIST_MIN = 0.045
_BET_DIST_MAX = 0.110


def find_bets(observations, assigned, width, height, board_slots=None,
              hero_cards=None):
    """Chips each seat has committed this street, as {seat_id: amount}.

    These are the amounts the UI renders on the felt between a seat and the
    pot, so they are read rather than inferred -- no comparing stacks across
    frames, which the single-frame rule rules out anyway. They should sum to
    the pot, which makes a useful self-check: it held exactly on the frames
    this was measured against.
    """
    diagonal = (width ** 2 + height ** 2) ** 0.5
    near, far = _BET_DIST_MIN * diagonal, _BET_DIST_MAX * diagonal

    # A seat's own stack is part of its plate, never a bet.
    plate_rows = {(p.x, p.y) for p in assigned.values()}

    # Both are lists of boxes; regions.HERO_CARDS already is one, so keeping
    # the iPad's the same shape means no special-casing per device.
    excluded = list(board_slots or []) + list(hero_cards or [])

    def on_the_board(o):
        """Board card ranks sit mid-table, inside the bet distance band.

        On a real frame the 5 of spades was attributed to mid_left as a 5.0
        bet, which then made the bets disagree with the pot. The board's
        position is known, so exclude it outright.
        """
        return any(x1 <= o.x <= x2 and y1 <= o.y <= y2
                   for x1, y1, x2, y2 in excluded)

    best = {}
    for o in observations:
        if o.confidence < _MIN_CONFIDENCE:
            continue
        value = parse_amount(o.text)
        if value is None:
            continue
        if any(abs(o.x - px) < 0.06 * width and abs(o.y - py) < 0.05 * height
               for px, py in plate_rows):
            continue
        if on_the_board(o):
            continue
        for seat_id, plate in assigned.items():
            distance = ((o.x - plate.x) ** 2 + (o.y - plate.y) ** 2) ** 0.5
            if not (near <= distance <= far):
                continue
            if seat_id not in best or distance < best[seat_id][0]:
                best[seat_id] = (distance, value)
    return {seat_id: value for seat_id, (_, value) in best.items()}


def is_in_hand(status):
    """Whether a seat showing `status` is still contesting the pot.

    An action bubble means they acted and are still in; "Folded" and the other
    persistent labels mean they are not. Unknown status is not evidence either
    way, so it reports False rather than guessing -- the same conservatism
    parser._is_occupied applies to seat occupancy.
    """
    if status is None:
        return False
    word = status.strip().lower()
    if word in _PERSISTENT_STATUS:
        return False
    return word in _ACTION_STATUS


def slots_in_strip(slots, rect, frame_size, strip_size):
    """Translate board slots from frame coordinates into the strip's.

    The strip is a crop of the frame, possibly resized, so a slot has to be
    shifted by the crop's origin and scaled by the resize. Doing this from the
    rect the helper reports -- rather than from a second copy of the crop
    fractions here -- keeps the two sides from drifting apart.
    """
    frame_w, frame_h = frame_size
    strip_w, strip_h = strip_size
    fx, fy, fw, fh = rect

    origin_x, origin_y = fx * frame_w, fy * frame_h
    scale_x = strip_w / (fw * frame_w)
    scale_y = strip_h / (fh * frame_h)

    # Rounded, not truncated: truncation biases every slot left and up by up
    # to a pixel, which walks the rank crop off the glyph it is meant to frame.
    return [
        (round((x1 - origin_x) * scale_x), round((y1 - origin_y) * scale_y),
         round((x2 - origin_x) * scale_x), round((y2 - origin_y) * scale_y))
        for x1, y1, x2, y2 in slots
    ]


def read_hero_cards(frame, hero_slots):
    """The hero's two hole cards, read from the attached strip.

    Uses card_vision.parse_hero_card unchanged: the pair is fanned, so neither
    card sits at a fixed offset and the rank has to be hunted for -- that
    reader was written for exactly this and is already proven on real hands.
    """
    if frame.hero_strip is None or frame.hero_strip_rect is None or not hero_slots:
        return []
    strip = frame.hero_strip
    local = slots_in_strip(hero_slots, frame.hero_strip_rect,
                           frame_size=(frame.width, frame.height),
                           strip_size=(strip.shape[1], strip.shape[0]))
    cards = []
    for x1, y1, x2, y2 in local:
        x1, y1 = max(x1, 0), max(y1, 0)
        x2 = min(x2, strip.shape[1])
        y2 = min(y2, strip.shape[0])
        if x2 <= x1 or y2 <= y1:
            continue
        card = card_vision.parse_hero_card(strip[y1:y2, x1:x2])
        if card:
            cards.append(card)
    return cards


def hero_signature(frame):
    """A coarse thumbnail of the hero card region, for change detection.

    16x16 grey, the same shape of signal the capture helper uses for its own
    diff gate. Coarse on purpose: it must ignore glow, buttons and animation
    while still separating a new deal, and measured on real frames those sit
    an order of magnitude apart.
    """
    if frame.hero_strip is None:
        return None
    small = cv2.resize(cv2.cvtColor(frame.hero_strip, cv2.COLOR_BGR2GRAY),
                       (16, 16), interpolation=cv2.INTER_AREA)
    return small.flatten().tolist()


# Where a seat's two face-down cards sit relative to its anchor, in full-frame
# pixels. They render up and to the left of the plate on every seat.
_CARD_BACK_BOX = (-140, -100, -20, -20)

# Fraction of that box that is mid-bright. Measured on real frames: a seat
# holding cards reads 7.3-9.5%, a folded one 0.0-2.7%, with nothing between.
_CARD_BACK_LIGHT = 0.05


def seat_has_cards(thumb, anchor, scale):
    """Whether this seat still holds cards, or None if there is no thumbnail.

    This is how a folded player is told from a live one. The alternative is
    the small red "Folded" label, which is read on only 28% of seats -- a
    player who has folded simply has no cards beside them, and that is a
    visual fact rather than an OCR result.
    """
    if thumb is None or not scale:
        return None
    ax, ay = anchor
    dx1, dy1, dx2, dy2 = _CARD_BACK_BOX
    x1, y1 = int((ax + dx1) * scale), int((ay + dy1) * scale)
    x2, y2 = int((ax + dx2) * scale), int((ay + dy2) * scale)
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, thumb.shape[1]), min(y2, thumb.shape[0])
    if x2 <= x1 or y2 <= y1:
        return None
    patch = thumb[y1:y2, x1:x2]
    light = float(((patch > 120) & (patch < 235)).mean())
    return light > _CARD_BACK_LIGHT


# The row of buttons along the bottom. Two different sets live there and they
# mean opposite things: the pre-action checkboxes are shown while you WAIT,
# the action buttons when it is your turn.
_PREACTION = {"call any", "check / fold", "straddle", "fold to any", "check"}
_PREACTION_STRICT = {"call any", "check / fold", "straddle", "fold to any"}
_ACTION = {"fold", "check", "bet", "raise", "all in", "call"}
_CALL_AMOUNT = re.compile(r"(?i)^call\s+([\d.]+)$")


@dataclass
class ActionState:
    """Whether it is the hero's turn, and what a call costs."""
    your_turn: bool = False
    to_call: float = None


def find_action(observations, width, height):
    """Read the bottom button row: is it your turn, and what do you owe?

    A decision needs both and neither was captured before. The subtlety is
    that Vision sometimes splits "Check / Fold" into separate Check and Fold
    tokens, which look exactly like the real buttons -- so any pre-action word
    on the row settles it as "not your turn", whatever else is there.
    """
    row = [o.text.strip() for o in observations
           if o.y > height * 0.93 and o.confidence >= _MIN_CONFIDENCE]
    lowered = [t.lower() for t in row]

    if any(t in _PREACTION_STRICT for t in lowered):
        return ActionState(your_turn=False)

    to_call = None
    for text in row:
        match = _CALL_AMOUNT.match(text)
        if match:
            to_call = parse_amount(match.group(1))
            break

    acting = any(t in _ACTION for t in lowered) or to_call is not None
    return ActionState(your_turn=acting, to_call=to_call)
