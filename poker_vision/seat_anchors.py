"""Seat anchor points per device layout, for the Vision capture path.

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
