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


def numeric_handle_plates(observations, anchors, width, height):
    """Seat plates whose handle is itself a number.

    find_seat_plates refuses any candidate that parses as an amount, because
    an amount above an amount is what the pot and the blind level look like.
    It is also what a player whose username is digits looks like. Live, a
    player called "000707" was missing from the table entirely: float() reads
    that as 707.0, so it stayed eligible as somebody's stack, and an OCR
    fragment above it ("de", "ce", "Ide") paired with it into a phantom seat
    holding 707 chips at a 0.05/0.10 table.

    Digits alone cannot settle it, so the seat anchor does. These plates are
    returned to be judged alongside the ordinary ones, not merged over them:
    assign_to_seats keeps whichever is closest to the anchor, and the real
    handle sits on it -- measured, "000707" was 17px from its anchor while
    the fragment that had been beating it was 68px away. That comparison is
    what fixes the seat; simply offering the plate is enough.

    The anchor test below is therefore a narrowing rather than a guard:
    removing it changes no seat on either corpus, because assign_to_seats
    discards anything too far from an anchor anyway. It is kept so this does
    not manufacture plates for the pot and the blind level on every frame.
    """
    gap_min, gap_max = _PLATE_GAP_MIN * height, _PLATE_GAP_MAX * height
    x_tol = _PLATE_X_TOL * width
    radius = _ANCHOR_RADIUS * ((width ** 2 + height ** 2) ** 0.5)
    trusted = [o for o in observations if o.confidence >= _MIN_CONFIDENCE]

    plates = []
    for candidate in trusted:
        if parse_amount(candidate.text) is None:
            continue                      # this pass is only for numeric handles
        if not _is_handle(candidate.text):
            continue                      # "2x", "100%", "+30s"
        # On a seat anchor, which is where a handle sits -- a stack sits a
        # row below one. See the note above: this narrows what is offered,
        # it is not what decides the seat.
        on_anchor = any(
            abs(candidate.y - ay) <= gap_max
            and ((candidate.x - ax) ** 2 + (candidate.y - ay) ** 2) ** 0.5 <= radius
            for ax, ay in anchors.values()
        )
        if not on_anchor:
            continue
        below = []
        for other in trusted:
            value = parse_amount(other.text)
            if value is None:
                continue
            if not gap_min <= other.y - candidate.y <= gap_max:
                continue
            if abs(other.x - candidate.x) > x_tol:
                continue
            below.append((other, value))
        if not below:
            continue
        _, value = min(below, key=lambda ov: ov[0].y - candidate.y)
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

    Tesseract is used here on purpose, and only here: it reads isolated
    glyphs, which is exactly Vision's weakness.

    It earns its place -- it recovers 170 of the 246 stacks it is asked about.
    It used to be the most expensive thing in the pipeline (~200-245ms a
    call, a subprocess per psm attempt); with Tesseract held in-process by
    ocr_utils it is ~3ms an attempt. See the latency budget in CLAUDE.md.
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
                      hero_cards=None, bet_offsets=None):
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
    # Judged alongside the ordinary plates rather than after them, so the
    # closest-to-anchor rule can prefer a real numeric handle over an OCR
    # fragment that had paired with it. See numeric_handle_plates.
    plates += numeric_handle_plates(frame.observations, anchors,
                                    frame.width, frame.height)
    assigned = assign_to_seats(plates, anchors, frame.width, frame.height)

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

    # After stack recovery, so a seat recovered from the mask -- which is
    # every all-in player, whose stack is "0" -- can still carry a label.
    statuses = find_statuses(frame.observations, frame.width, frame.height,
                             plates=list(assigned.values()))
    badges = find_badges(frame.observations, frame.width, frame.height,
                         plates=list(assigned.values()),
                         mask_badges=find_mask_badges(frame.text_mask))
    bets = find_bets(frame.observations, assigned, frame.width, frame.height,
                     board_slots=board_slots, hero_cards=hero_cards,
                     bet_offsets=bet_offsets)
    # A bet of "1" or "3" is a lone glyph, which Vision does not report --
    # the same gap that loses one-character stacks. Re-read it at the seat's
    # bet spot. Round raise sizes are exactly these, so they mattered.
    if bet_offsets and frame.text_mask is not None:
        for seat_id, plate in assigned.items():
            if seat_id in bets or seat_id not in bet_offsets:
                continue
            dx, dy = bet_offsets[seat_id]
            value = recover_bet(frame, plate.x + dx, plate.y + dy)
            if value is not None:
                bets[seat_id] = value
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
        obscured=table_obscured(frame, plates),
    )


# A PurePoker panel -- chat, settings, a player's profile, the achievement
# drawer -- dims the table under it. Read through that, card backs vanish
# into the dark and every player comes back folded while their action
# bubbles still read: the "folded but checked" rows seen live were all this.
# Measured over 241 saved frames: open tables put the felt at 45-53 and the
# 95th-percentile brightness of a handle at >=175; with a panel up, felt
# <=37 and handles <=164. Both must be dark. Felt alone would trip on a
# different table theme, and a pop-up over the middle (the bomb-pot status
# box) darkens the felt but leaves every seat bright and readable.
_FELT_DIMMED = 41
_HANDLE_DIMMED = 170


def table_obscured(frame, plates):
    """Whether a panel is dimming the table. False when it cannot tell."""
    thumb, scale = frame.table_thumb, frame.table_thumb_scale
    if thumb is None or not scale:
        return False
    h, w = thumb.shape[:2]
    felt = min(float(thumb[int(.52 * h):int(.58 * h), int(.30 * w):int(.40 * w)].mean()),
               float(thumb[int(.52 * h):int(.58 * h), int(.60 * w):int(.70 * w)].mean()))
    if felt >= _FELT_DIMMED:
        return False
    peaks = []
    for plate in plates:
        x, y = int(plate.x * scale), int(plate.y * scale)
        box = thumb[max(y - 5, 0):y + 5, max(x - 20, 0):x + 20]
        if box.size:
            peaks.append(float(np.percentile(box, 95)))
    if not peaks:
        return True
    return float(np.median(peaks)) < _HANDLE_DIMMED


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


# Labels arrive damaged in two regular ways, measured over 58 live frames.
# The red "Folded" under an avatar is half-covered by the stack badge, so
# Vision returns "Folde", "Folder", "Folde100" -- 46 labels dropped by an
# exact match. And "Check" fuses with the SB/BB badge beside it into
# "Checkв", "CheckB", "Check B" -- 10 more. Both are accepted, but only in
# those shapes: a status word followed by anything else is chat or a
# handle ("Betty", "Callum"), and reading those as actions would invent one.
_CLIPPED_FOLDED = re.compile(r"^folde[a-z]?\s?\d*$")
_BADGE_SUFFIX = re.compile(r"^[bsd]{1,2}$")


def _status_word(text):
    """The canonical status a recognised string names, or None."""
    word = text.strip().lower()
    if word in _STATUS_WORDS:
        return _CANONICAL_STATUS[word]
    if _CLIPPED_FOLDED.match(word):
        return "Folded"
    # "Sitting Out" is cut short by the stack badge too ("Sitting C" live).
    # Nothing else on a table starts with "sitting".
    if word.startswith("sitting ") or word == "sitting":
        return "Sitting Out"
    for status in _STATUS_WORDS:
        if not word.startswith(status):
            continue
        # What follows must be the badge beside the bubble, in whatever
        # letters Vision chose for it: "Checkв", "Check B", and live,
        # "Check зв" -- 5 of 15 checks in a 4-minute capture were that shape.
        # A "3" for the S as well ("Check3B") -- allowed only here, after a
        # status word, so a stray "3B" elsewhere never becomes a badge.
        rest = word[len(status):]
        letters = rest.strip().translate(_LOOKALIKES).lower().replace("3", "s")
        if len(rest) <= 3 and _BADGE_SUFFIX.match(letters):
            return _CANONICAL_STATUS[status]
    return None


def find_statuses(observations, width, height, plates=None):
    """Map each status label onto the seat plate it belongs to.

    Returns {(name, x, y): status} keyed by the plate's identity, so the
    caller can match it back to whichever seat that plate was assigned.

    Pass `plates` to match against the seats actually assigned. Without it
    only plates whose stack Vision read are considered, and a player whose
    stack is "0" -- every all-in player, a one-character number Vision does
    not report -- had nowhere for their "All In" label to land.
    """
    if plates is None:
        plates = find_seat_plates(observations, width, height)
    dy_min, dy_max = _STATUS_DY_MIN * height, _STATUS_DY_MAX * height
    dx_tol = _STATUS_DX_TOL * width

    out = {}
    for o in observations:
        if o.confidence < _MIN_CONFIDENCE:
            continue
        status = _status_word(o.text)
        if status is None:
            continue
        below = [
            p for p in plates
            if dy_min <= (p.y - o.y) <= dy_max and abs(p.x - o.x) <= dx_tol
        ]
        if not below:
            continue
        plate = min(below, key=lambda p: p.y - o.y)
        key = (plate.name, plate.x, plate.y)
        # A persistent label states the seat's condition; an action bubble
        # only the last thing they did. If both are read, the label wins.
        if out.get(key, "").lower() in _PERSISTENT_STATUS:
            continue
        out[key] = status
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
    "\u0437": "s", "\u0417": "S",      # Cyrillic ze -- live, "SB" came back "зв"
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


# The badges as drawn, read from the text mask rather than by Vision.
#
# Vision reads "SB"/"BB" only sometimes and the dealer button never -- a
# badge is one or two letters in a tiny disc, the isolated short text it
# misses. Live, two of four hero turns had no badge read anywhere in the
# hand, so no positions at all. In the mask the badges are unmistakable: SB
# and BB are ~40px white discs with the two letters cut out, the button a
# ~42x28 flattened disc with a "D" cut out. Across 259 saved frames this
# found a badge in 238; where Vision also read one (167), the labels never
# disagreed; and 59 of 59 sampled detections checked by eye were right.
#
# Telling them apart: a B has enclosed counters, so its hole in the disc
# holds white islands and an S's does not. Both badges end in B, which is
# also what rejects the "zZ" sitting-out marker -- the same disc style, and
# read as a second SB before the check was added.
_BADGE_DISC = (34, 48)          # SB/BB disc width, px
_BUTTON_ASPECT = (1.3, 1.8)     # the D button is flattened


def find_mask_badges(text_mask):
    """[TextObservation] for each badge drawn in the mask, text "SB"/"BB"/"D"."""
    if text_mask is None:
        return []
    m = (text_mask > 0).astype(np.uint8)
    count, labels, stats, centres = cv2.connectedComponentsWithStats(m, connectivity=8)
    lo, hi = _BADGE_DISC
    found = []
    for k in range(1, count):
        bx, by, bw, bh = (int(v) for v in stats[k, :4])
        if not (lo <= bw <= hi and 22 <= bh <= hi):
            continue
        comp = (labels[by:by + bh, bx:bx + bw] == k).astype(np.uint8)
        contours, hierarchy = cv2.findContours(comp, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            continue
        holes = [contours[i] for i, h in enumerate(hierarchy[0]) if h[3] >= 0]

        def islands(hole):
            # White shapes inside a letter's hole that are not the disc
            # itself: the counters of a B, the inside of a D.
            hx, hy, hw, hh = cv2.boundingRect(hole)
            inside = np.zeros((hh, hw), np.uint8)
            cv2.drawContours(inside, [hole - [hx, hy]], -1, 1, -1)
            region = labels[by + hy:by + hy + hh, bx + hx:bx + hx + hw]
            other = ((region != k) & (region > 0) & (inside > 0)).astype(np.uint8)
            n, _, st, _ = cv2.connectedComponentsWithStats(other, connectivity=8)
            return sum(1 for j in range(1, n) if st[j, cv2.CC_STAT_AREA] >= 6)

        x, y = int(centres[k][0]), int(centres[k][1])
        if abs(bw - bh) <= 6 and len(holes) == 2:
            left, right = sorted(holes, key=lambda h: cv2.boundingRect(h)[0])
            if islands(right) >= 1:
                text = "BB" if islands(left) >= 1 else "SB"
                found.append(TextObservation(text=text, x=x, y=y, w=bw, h=bh, confidence=1.0))
        elif (_BUTTON_ASPECT[0] <= bw / bh <= _BUTTON_ASPECT[1]
              and len(holes) == 1 and islands(holes[0]) == 1):
            found.append(TextObservation(text="D", x=x, y=y, w=bw, h=bh, confidence=1.0))
    return found


def find_badges(observations, width, height, plates=None, mask_badges=()):
    """Map each D/SB/BB badge onto the seat plate it belongs to.

    Returns {(name, x, y): badge} keyed by the plate's identity. `plates`
    should be the assigned seats -- otherwise only plates whose stack Vision
    read are considered, and a badge beside a one-character stack had
    nowhere to land. `mask_badges` adds badges read from the text mask; one
    Vision also read is counted once.
    """
    if plates is None:
        plates = find_seat_plates(observations, width, height)
    seen = [o for o in observations if _badge_word(o.text)]
    observations = list(observations) + [
        b for b in mask_badges
        if not any(abs(b.x - o.x) < 40 and abs(b.y - o.y) < 40 for o in seen)]
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


# How far a bet's label may sit from its seat's measured spot. The spread
# within a seat is under 40px; the nearest two seats' spots are ~300px apart.
_BET_SPOT_TOL = 60


def find_bets(observations, assigned, width, height, board_slots=None,
              hero_cards=None, bet_offsets=None):
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

    # Each amount belongs to the one seat it is nearest, and each seat takes
    # the nearest amount it was given. Letting every seat pick independently
    # handed one pile of chips to two players -- 7 times in 58 live frames,
    # e.g. an all-in 0.35 that also appeared as the bet of the player above.
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
        owner = None
        for seat_id, plate in assigned.items():
            if bet_offsets and seat_id in bet_offsets:
                # Measured geometry: distance from where this seat's bets go.
                dx, dy = bet_offsets[seat_id]
                distance = ((o.x - plate.x - dx) ** 2 + (o.y - plate.y - dy) ** 2) ** 0.5
                fits = distance <= _BET_SPOT_TOL
            else:
                distance = ((o.x - plate.x) ** 2 + (o.y - plate.y) ** 2) ** 0.5
                fits = near <= distance <= far
            if fits and (owner is None or distance < owner[0]):
                owner = (distance, seat_id)
        if owner is None:
            continue
        distance, seat_id = owner
        if seat_id not in best or distance < best[seat_id][0]:
            best[seat_id] = (distance, value)
    return {seat_id: value for seat_id, (_, value) in best.items()}


# Bet amounts are one fixed font size: 23-25px tall in every recovered bet
# checked by eye. A card back flying past mid-deal has a logo that splits
# into two digit-shaped halves, 27-52px tall, which OCR read as "6" and "2".
_BET_GLYPH_H = (20 / 2360, 26 / 2360)

# A bet is always drawn under a chip icon. Measured on the thumbnail, the
# patch just above a real amount averages 74-106; above a card logo, 38-52.
_CHIP_ABOVE_MIN = 65


def _digit_row(window, height, glyph_h=None):
    """Only the glyphs of the one amount nearest the window's centre.

    A bet label shares its neighbourhood with the chip graphic above it,
    sometimes a second chip, and the edge of an avatar. Handed the whole
    window, Tesseract read "0.1" as 0.71 or 9.1 depending on where the crop
    happened to fall. So the digits are picked out by shape -- a row of
    digit-sized blobs sharing a baseline, plus the decimal point between
    them -- and redrawn alone on a blank canvas.
    """
    n, labels, stats, centres = cv2.connectedComponentsWithStats(
        (window > 0).astype(np.uint8), connectivity=8)
    lo, hi = glyph_h or (_DIGIT_H_MIN, _DIGIT_H_MAX)
    lo, hi = lo * height, hi * height
    T, L, W, H, A = (cv2.CC_STAT_TOP, cv2.CC_STAT_LEFT, cv2.CC_STAT_WIDTH,
                     cv2.CC_STAT_HEIGHT, cv2.CC_STAT_AREA)
    digits = [i for i in range(1, n)
              if lo <= stats[i, H] <= hi
              and 0.15 <= stats[i, W] / stats[i, H] <= 1.2
              and stats[i, A] >= 40]
    if not digits:
        return None
    mid_x, mid_y = window.shape[1] / 2, window.shape[0] / 2
    seed = min(digits, key=lambda i: abs(centres[i][1] - mid_y))
    size, base = stats[seed, H], stats[seed, T] + stats[seed, H]
    row = sorted((i for i in digits
                  if abs(stats[i, T] + stats[i, H] - base) <= 0.25 * size
                  and 0.7 * size <= stats[i, H] <= 1.3 * size),
                 key=lambda i: stats[i, L])
    # One amount's digits sit close together; a blob further along the same
    # line is something else, so keep the run nearest the centre.
    runs, run = [], [row[0]]
    for i in row[1:]:
        prev = run[-1]
        if stats[i, L] - (stats[prev, L] + stats[prev, W]) > 1.2 * size:
            runs.append(run)
            run = [i]
        else:
            run.append(i)
    runs.append(run)
    row = min(runs, key=lambda r: min(abs(centres[i][0] - mid_x) for i in r))
    x1 = min(stats[i, L] for i in row)
    x2 = max(stats[i, L] + stats[i, W] for i in row)
    top = min(stats[i, T] for i in row)
    points = [i for i in range(1, n) if i not in row
              and x1 <= centres[i][0] <= x2
              and stats[i, H] <= 0.4 * size
              and abs(stats[i, T] + stats[i, H] - base) <= 0.25 * size]
    canvas = np.zeros_like(window)
    for i in row + points:
        canvas[labels == i] = 255
    return canvas[top:base, x1:x2]


def recover_bet(frame, x, y):
    """Re-read a bet at its seat's spot from the text mask, or None.

    Only where Vision saw no text at all: if it read something there and it
    was not an amount -- a "+37.2" win burst, say -- that is not a bet to
    recover. Only under a chip icon, only glyphs of the bet font's size, and
    believed only if two readings with different padding agree: the reading
    is sensitive to framing in exactly the cases where it is wrong.
    """
    mask, thumb, scale = frame.text_mask, frame.table_thumb, frame.table_thumb_scale
    if mask is None or thumb is None or not scale:
        return None
    if any(abs(o.x - x) < 80 and abs(o.y - y) < 40 for o in frame.observations):
        return None
    chip = thumb[max(int((y - 55) * scale), 0):int((y - 20) * scale),
                 max(int((x - 20) * scale), 0):int((x + 20) * scale)]
    if chip.size == 0 or float(chip.mean()) < _CHIP_ABOVE_MIN:
        return None
    height = frame.height
    window = mask[max(y - 40, 0):y + 40, max(x - 80, 0):x + 80]
    if window.size == 0 or not window.any() or not looks_like_digits(window, height):
        return None
    glyphs = _digit_row(window, height, glyph_h=_BET_GLYPH_H)
    if glyphs is None or glyphs.size == 0:
        return None
    reads = []
    for pad in (8, 16):
        framed = cv2.copyMakeBorder(glyphs, pad, pad, pad, pad,
                                    cv2.BORDER_CONSTANT, value=0)
        reads.append(number_ocr.ocr_number(cv2.cvtColor(framed, cv2.COLOR_GRAY2BGR)))
    if reads[0] is None or reads[0] != reads[1]:
        return None
    return reads[0]


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
    if word == "all in":
        # Persistent, but the opposite of folded: they are in to the end.
        return True
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
    """Whether it is the hero's turn, and what a call costs.

    `to_call` is 0.0 when checking is free and None when the price could not
    be read at all. Those are different answers and a decision needs to tell
    them apart.
    """
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

    # A Check button means there is no outstanding bet, so a call costs
    # nothing. That has to be reported as 0 rather than left None: None is
    # "the price could not be read", and confusing the two makes a free card
    # look like a decision that cannot be taken. The client offers Check or
    # Call, never both, so an explicit amount always wins.
    if to_call is None and "check" in lowered:
        to_call = 0.0

    acting = any(t in _ACTION for t in lowered) or to_call is not None
    return ActionState(your_turn=acting, to_call=to_call)
