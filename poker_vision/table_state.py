"""Data model for a single parsed frame of table state."""
from dataclasses import dataclass, field, asdict
from typing import List, Optional


@dataclass
class Seat:
    seat_id: str          # e.g. "top_left" (screen position slot, stable across hands)
    name: Optional[str] = None
    stack: Optional[float] = None
    position: Optional[str] = None      # "D" / "SB" / "BB" (badge OCR) or a
                                         # standard name like "UTG" / "HJ" /
                                         # "CO" (inferred, see parser.infer_positions) / None
    status: Optional[str] = None        # "Folded" / "Call" / "Raise" / "Check" / "Sitting Out" / ...
    in_hand: bool = False
    is_hero: bool = False
    hole_cards: List[str] = field(default_factory=list)  # only known for hero, or at showdown


@dataclass
class TableState:
    frame_index: int
    timestamp_sec: float
    seats: List[Seat]
    board_cards: List[str]
    pot: Optional[float] = None

    def to_dict(self):
        return {
            "frame_index": self.frame_index,
            "timestamp_sec": self.timestamp_sec,
            "pot": self.pot,
            "board_cards": self.board_cards,
            "seats": [asdict(s) for s in self.seats],
        }


def is_plausible_table(state, min_seats_with_stack=4):
    """Whether a parsed TableState dict (see TableState.to_dict()) plausibly
    came from a real PurePoker table screenshot, as opposed to some other
    screen on the phone.

    Stack-amount OCR only succeeds on a real table (it needs a specific
    numeric pattern at a specific pixel position per seat), so it almost
    never false-positives on unrelated screen content -- unlike name OCR
    or the card-presence heuristic, which do pick up noise elsewhere.
    Empirically, real table screenshots show 5-6 of 9 seats with a
    detected stack; unrelated screenshots show 0. Default threshold of 4
    leaves margin on both sides.
    """
    seats_with_stack = sum(1 for s in state["seats"] if s["stack"] is not None)
    return seats_with_stack >= min_seats_with_stack
