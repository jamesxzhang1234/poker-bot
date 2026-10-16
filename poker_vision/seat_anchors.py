"""Per-device layout constants for the Vision capture path.

Holds the seat anchors used by vision_reader and the board geometry used by
board_reader -- everything that differs between an iPhone frame and an iPad
frame, in one place, so adding a device is a single edit here.

An anchor is a single (x, y) point near where a seat's nameplate renders --
not a box around a field. The Vision path locates text by recognition and then
asks which seat it belongs to, so it only needs to know roughly where each
seat lives; contrast regions.py, which must know the exact pixel rectangle of
every field because it crops before it reads. That difference is why adding a
device here costs a handful of coordinates rather than a full recalibration.

Coordinates are in the reference frame's pixels, top-left origin.
"""

# Derived from 30 real frames captured off a USB-attached iPad at 1640x2360:
# every name-above-stack pairing was clustered and each cluster's centroid
# taken. Sighting counts below are how many of the 30 frames contributed, and
# are a direct measure of how well-evidenced each anchor is.
IPAD_PORTRAIT = {
    "top_left":         (511, 565),    # 30 sightings
    "top_right":        (1124, 565),   #  8
    "mid_left":         (404, 895),    # 30
    "mid_right":        (1232, 897),   # 11
    "lower_mid_left":   (393, 1392),   # 30
    "lower_mid_right":  (1228, 1391),  # 26
    "bottom_right":     (1195, 1721),  # 24
    "hero":             (814, 1920),   # 29
    # Never occupied in the sample, so this is bottom_right mirrored across
    # the frame's horizontal centre (x = 820) -- an estimate, not a
    # measurement. regions.py marks its unverified mirrored seats the same
    # way. Recheck if a player sits here and isn't picked up.
    "bottom_left":      (445, 1721),   #  0 -- mirrored, unverified
}

IPAD_PORTRAIT_SIZE = (1640, 2360)

# Where each seat's committed chips are labelled, as an offset from its plate
# (the handle's centre). Measured from every bet Vision read across 241 saved
# iPad frames: each seat has one spot, and the spread inside it is under 40px.
# Bets are accepted only near their seat's spot. The distance band used
# before took anything 129-316px from a plate, which admitted face-up card
# ranks at showdown ("6", "5" ~160px above a seat) and strays such as "86".
# bottom_left now has 27 sightings, so its anchor above is no longer unused.
IPAD_BET_OFFSETS = {
    "top_left":        (142, 131),    #  9 sightings
    "top_right":       (-140, 135),   # 44
    "mid_left":        (170, 12),     # 23
    "mid_right":       (-170, 9),     # 10
    "lower_mid_left":  (180, 0),      # 30
    "lower_mid_right": (-170, 0),     # 32
    "bottom_left":     (180, -104),   # 26
    "bottom_right":    (-178, -104),  # 30
    "hero":            (4, -269),     # 40
}

# Which seat id is the player running the capture. The hero plate is the one
# at the bottom centre of the screen on every layout this project handles.
HERO_SEAT_ID = "hero"


# Derived the same way from 47 real iPhone frames at 1206x2622. Worth knowing:
# every one of these landed within 4-23px of the corresponding name_row centre
# that regions.py records from hand measurement, so the clustering and the
# original calibration independently agree. That is the evidence that the
# anchor approach is sound, not just cheaper.
IPHONE_PORTRAIT = {
    "top_left":         (258, 628),
    "top_right":        (938, 628),
    "mid_left":         (143, 993),
    "mid_right":        (1057, 994),
    "lower_mid_left":   (128, 1546),
    "lower_mid_right":  (1057, 1546),
    "bottom_left":      (180, 1913),
    "bottom_right":     (1017, 1913),
    "hero":             (595, 2133),
}

IPHONE_PORTRAIT_SIZE = (1206, 2622)


# Board geometry per device, for board_reader. The community cards are dealt
# left to right into a fixed five-slot band -- they are NOT centred as the
# count grows, which is what lets one set of slots serve every street.
#
# iPad slots measured off seven real five-card boards by finding each card's
# edges against the felt: the cards start at x=494, repeat every 132px and
# are 125px wide.
#
# The earlier figures -- pitch 136, width 118 -- were derived rather than
# measured, and the 4px pitch error accumulated left to right until the fifth
# slot sat 17px right of its card. That clips the left of the rank glyph,
# which a single character survives and a "10" does not: it loses its "1" and
# the bare "0" matched Q, so a live river read Qc for a ten. Re-measuring took
# the labelled boards in the live corpus from 5/16 to 16/16.
#
# The band is cropped 12px above the cards (973 rather than 985). The glyph
# starts 13px below the card's top edge, and without that margin it touches
# the crop boundary and the isolator clips it.
IPAD_BOARD_SLOTS = [(494 + i * 132, 973, 494 + i * 132 + 125, 1127) for i in range(5)]

# Rank glyph in a card's top-left corner, relative to the slot's (x1, y1).
# Taller than the iPhone's because the iPad renders the card larger; it stops
# short of the suit pip, which would otherwise merge with the rank and give
# the same rank several different templates.
IPAD_RANK_OFFSET = (8, 4, 70, 48)


# The hero's own two cards, which render fanned beside their seat -- inside
# the distance band where bets live, so their rank glyphs would otherwise be
# read as chips the hero had committed.
# A list of boxes, matching regions.HERO_CARDS on the iPhone side, so both
# devices hand find_bets the same shape.
IPAD_HERO_CARDS = [(940, 1850, 1180, 2070)]

# The hero's two cards as separate slots, for reading them. Deliberately
# generous and overlapping, the way regions.HERO_CARDS is on the iPhone: the
# pair is fanned, so neither card sits at an exact fixed position and the left
# one is partly behind the right. Chosen by testing three candidate pairs
# against hands read off the screen by eye; this one got A-d K-d right on
# every frame it appeared in.
IPAD_HERO_CARD_SLOTS = [
    (945, 1830, 1060, 2000),
    (1055, 1840, 1175, 2020),
]
