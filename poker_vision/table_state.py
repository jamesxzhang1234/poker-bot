"""Data model for a single parsed frame of table state."""
from dataclasses import dataclass, field, asdict
from typing import List, Optional


@dataclass
class Seat:
    seat_id: str          # e.g. "top_left" (screen position slot, stable across hands)
    name: Optional[str] = None
    stack: Optional[float] = None
    position: Optional[str] = None      # "D" / "SB" / "BB" / None
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
