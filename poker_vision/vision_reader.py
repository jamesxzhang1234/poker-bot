"""Turns Apple Vision text observations into table state.

The Swift capture helper (tools/ipad_capture) runs one VNRecognizeTextRequest
over each captured frame and emits a JSON payload: every recognised string
with its pixel position and confidence. That is a different shape of input
from the rest of the pipeline, which crops fixed boxes out of a frame and
OCRs each one -- here the text arrives already located, so this module
assigns it to seats by proximity to a seat anchor instead of by calibrated
per-field boxes.
"""
import json
import re
from dataclasses import dataclass
from typing import List

from . import seat_anchors
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


def parse_payload(payload):
    """Parse one JSON payload (str or already-decoded dict) from the helper."""
    data = json.loads(payload) if isinstance(payload, (str, bytes)) else payload
    return VisionFrame(
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
    candidate = text.strip()
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


def _is_handle(text):
    """Whether a recognised string could be a player's handle."""
    cleaned = text.strip().lower()
    if not cleaned or cleaned in _NOT_A_HANDLE:
        return False
    # Raise presets ("2x"), the battery ("100%"), the turn timer ("+30s").
    if re.match(r"^[+-]?\d+(\.\d+)?\s*(x|%|s)$", cleaned):
        return False
    return True


@dataclass
class SeatPlate:
    """A handle and the stack rendered directly beneath it."""
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
    amounts = [(o, parse_amount(o.text)) for o in trusted]
    amounts = [(o, v) for o, v in amounts if v is not None]

    plates = []
    for candidate in trusted:
        if parse_amount(candidate.text) is not None:
            continue                      # an amount is never a handle
        if not _is_handle(candidate.text):
            continue
        below = [
            (o, v) for o, v in amounts
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


def build_table_state(payload, anchors):
    """Turn one Vision payload into a TableState.

    Only fills what a whole-frame text read can establish: who is seated,
    their stacks, and the pot. Board cards and position badges are not set
    here -- Vision reads card *ranks* but not suits, and picks badges up only
    intermittently, so both stay with the pixel-level readers that handle them
    properly (card_vision for suits, the badge crops for D/SB/BB).
    """
    frame = parse_payload(payload)
    plates = find_seat_plates(frame.observations, frame.width, frame.height)
    assigned = assign_to_seats(plates, anchors, frame.width, frame.height)

    seats = [
        Seat(
            seat_id=seat_id,
            name=plate.name,
            stack=plate.stack,
            is_hero=(seat_id == seat_anchors.HERO_SEAT_ID),
        )
        for seat_id, plate in sorted(assigned.items())
    ]
    return TableState(
        frame_index=frame.frame_index,
        timestamp_sec=frame.timestamp_sec,
        seats=seats,
        board_cards=[],
        pot=find_pot(frame.observations, frame.width, frame.height),
    )
