"""Action-order direction around this game's seat layout.

Separate from regions.py's pixel calibration (which owns the seat-id
names this depends on) and from positions.py's generic poker-position
names (which this doesn't concern itself with at all) -- this module
answers one question: given regions.SEATS's screen-slot seat-ids, what
order do they act in? parser.infer_positions is the sole consumer.
"""
from . import regions

# Action-order direction around the table (D -> SB -> BB -> UTG -> ... -> back
# to D), i.e. the direction seats are dealt to, not raw screen position.
# Confirmed against a real session: top_right BB implies mid_right is UTG,
# which only holds in this direction (the reverse of a naive top-left-first
# reading of on-screen seat position). Used to fill in a seat's D/SB/BB
# label from another seat's successfully-read badge in the same frame, by
# rotating through the occupied seats -- see parser.infer_positions().
# Confirmed on limited data (a handful of real hands); may need correcting
# if it produces a wrong label on a hand shape not yet seen.
SEAT_ORDER = [
    "top_right",
    "mid_right",
    "lower_mid_right",
    "bottom_right",
    "hero",
    "bottom_left",
    "lower_mid_left",
    "mid_left",
    "top_left",
]

# regions.SEATS is the source of truth for which seat-ids exist; catch a
# renamed/added/removed seat here, loudly, at import time, rather than
# relying on this list and that dict happening to stay in sync by hand.
assert set(SEAT_ORDER) == set(regions.SEATS.keys()), (
    "seat_layout.SEAT_ORDER is out of sync with regions.SEATS's seat-ids"
)
