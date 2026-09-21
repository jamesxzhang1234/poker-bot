"""
Screen-region calibration for the PurePoker mobile client (portrait recordings).

All boxes were measured on a 1206x2622 reference frame extracted from a real
gameplay recording (see frames/crops/grid_*.png, produced with
tools/grid_overlay.py). At runtime, boxes are rescaled by the ratio between
the actual frame size and this reference size, so the pipeline still works if
the recording is a different resolution (as long as the aspect ratio and UI
layout match).

Box format is always (x1, y1, x2, y2) in pixels at the reference resolution.
"""

REF_W = 1206
REF_H = 2622

# One entry per seat around the table (8 opponent seats + hero).
# "cards" = where face-down card backs render when a seat is still in the hand.
# "badge" = generous zone for the D/SB/BB position badge near the avatar.
# "status" = generous zone spanning the action bubble (Check/Call/Raise/Bet)
#            above the avatar and the persistent status text (Folded/Sitting
#            Out/All-in) near the avatar's lower edge.
SEATS = {
    "top_left": {
        "avatar": (177, 415, 347, 585),
        "cards": (120, 490, 240, 580),
        "name_row": (150, 602, 372, 660),
        "stack_row": (150, 660, 372, 718),
        "badge": (405, 585, 464, 627),
        "status": (80, 340, 400, 600),
    },
    "top_right": {
        "avatar": (877, 415, 1012, 585),
        "cards": (818, 505, 918, 585),
        "name_row": (832, 602, 1055, 660),
        "stack_row": (832, 660, 1055, 718),
        # Mirrored from top_left across the frame's horizontal center;
        # no seat was ever occupied here in the calibration screenshots,
        # so this is an unverified estimate. Recheck if a player sits
        # here and their D/SB/BB badge isn't detected.
        "badge": (742, 585, 801, 627),
        "status": (800, 340, 1120, 600),
    },
    "mid_left": {
        "avatar": (75, 735, 220, 905),
        "cards": (0, 860, 100, 945),
        "name_row": (28, 955, 258, 1022),
        "stack_row": (28, 1022, 258, 1090),
        "badge": (235, 741, 288, 794),
        "status": (0, 700, 280, 960),
    },
    "mid_right": {
        "avatar": (945, 790, 1085, 960),
        "cards": (945, 860, 1040, 945),
        "name_row": (945, 955, 1180, 1022),
        "stack_row": (945, 1022, 1180, 1090),
        # Mirrored from mid_left -- unverified estimate, see top_right's note.
        "badge": (918, 741, 971, 794),
        "status": (900, 700, 1206, 960),
    },
    "lower_mid_left": {
        "avatar": (0, 1310, 195, 1460),
        "cards": (0, 1400, 90, 1480),
        "name_row": (0, 1500, 255, 1557),
        "stack_row": (0, 1557, 255, 1615),
        # Mirrored from lower_mid_right -- unverified estimate, see top_right's note.
        "badge": (218, 1290, 285, 1346),
        "status": (0, 1150, 300, 1500),
    },
    "lower_mid_right": {
        "avatar": (945, 1290, 1150, 1490),
        "cards": (945, 1400, 1040, 1490),
        "name_row": (945, 1490, 1180, 1558),
        "stack_row": (945, 1552, 1180, 1618),
        "badge": (915, 1290, 988, 1346),
        "status": (900, 1140, 1206, 1500),
    },
    "bottom_left": {
        "avatar": (50, 1705, 235, 1875),
        "cards": (0, 1785, 140, 1875),
        "name_row": (68, 1885, 300, 1935),
        "stack_row": (68, 1935, 300, 1985),
        "badge": (54, 1661, 108, 1712),
        "status": (0, 1545, 320, 1855),
    },
    "bottom_right": {
        "avatar": (955, 1700, 1150, 1870),
        "cards": (955, 1775, 1050, 1865),
        "name_row": (903, 1885, 1150, 1935),
        "stack_row": (903, 1935, 1150, 1985),
        # Mirrored from bottom_left -- unverified estimate, see top_right's note.
        "badge": (1098, 1661, 1152, 1712),
        "status": (880, 1545, 1206, 1850),
    },
    "hero": {
        "avatar": (420, 1690, 790, 2095),
        "cards": None,  # hero cards are shown face-up separately, see HERO_CARDS
        "name_row": (480, 2100, 722, 2150),
        "stack_row": (480, 2155, 722, 2210),
        "badge": (460, 1845, 528, 1907),
        "status": (380, 1900, 820, 2100),
    },
}

# Hero's own hole cards, shown face-up at bottom-right regardless of seat.
HERO_CARDS = [
    (748, 2085, 862, 2245),
    (866, 2085, 978, 2245),
]

# Five community card slots, left to right.
BOARD_SLOTS = [
    (250, 1095, 372, 1265),
    (397, 1095, 517, 1265),
    (542, 1095, 662, 1265),
    (687, 1095, 807, 1265),
    (832, 1095, 950, 1265),
]

# Rank glyph sits in the top-left corner of each card slot.
RANK_OFFSET = (15, 8, 60, 60)  # (dx1, dy1, dx2, dy2) relative to slot's (x1, y1)

POT_BOX = (470, 955, 735, 1040)

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


def status_sub_zones(seat_cfg):
    """Split a seat's generous 'status' zone into two tighter sub-zones:

    - bubble_zone: strictly above the avatar, where transient action bubbles
      (Check/Call/Raise/Bet) render. Avoids avatar-color interference.
    - label_zone: a band straddling the avatar's bottom edge, where
      persistent state text (Folded/Sitting Out/All-in) renders.

    Either may be None if the seat's geometry doesn't leave room for it.
    """
    sx1, sy1, sx2, sy2 = seat_cfg["status"]
    ax1, ay1, ax2, ay2 = seat_cfg["avatar"]

    bubble_zone = None
    if ay1 - sy1 > 15:
        bubble_zone = (sx1, sy1, sx2, ay1)

    label_zone = None
    label_top = max(sy1, ay2 - 70)
    label_bottom = min(sy2, ay2 + 40)
    if label_bottom - label_top > 15:
        label_zone = (sx1, label_top, sx2, label_bottom)

    return bubble_zone, label_zone


def scale_box(box, frame_w, frame_h):
    """Rescale a reference-resolution box to the actual frame size."""
    sx = frame_w / REF_W
    sy = frame_h / REF_H
    x1, y1, x2, y2 = box
    return (int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy))
