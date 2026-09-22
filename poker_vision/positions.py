"""Canonical poker position names and their order, dealer through cutoff.

Both parser.py and text_summary.py need this same domain fact -- parser
uses MIDDLE_POSITION_NAMES to name the seats between BB and the button
when resolving an occupied seat's position; text_summary uses
POSITION_ORDER to sort the printed seat list into that order. Both read
from here rather than each declaring their own copy, so there's one place
that says what the position order is.
"""

# Standard poker position names for the seats between BB and the button, in
# action order (closest to BB first), keyed by how many such seats there
# are. Matches common charts (6-max = UTG,HJ,CO; 9-max = UTG,UTG+1,UTG+2,
# LJ,HJ,CO); this game's seat layout maxes out at 9 total seats (8 opponent
# slots + hero), so 6 is the largest key needed (9 - D - SB - BB).
MIDDLE_POSITION_NAMES = {
    0: [],
    1: ["UTG"],
    2: ["UTG", "CO"],
    3: ["UTG", "HJ", "CO"],
    4: ["UTG", "UTG+1", "HJ", "CO"],
    5: ["UTG", "UTG+1", "LJ", "HJ", "CO"],
    6: ["UTG", "UTG+1", "UTG+2", "LJ", "HJ", "CO"],
}

# Full action-order sequence from dealer through the largest (9-max)
# table's cutoff. Derived from MIDDLE_POSITION_NAMES's largest entry
# rather than typed out separately, so the two can't silently drift.
POSITION_ORDER = ["D", "SB", "BB"] + MIDDLE_POSITION_NAMES[6]
