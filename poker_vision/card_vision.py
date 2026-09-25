"""Card recognition: suit by background color, rank by OCR on the corner glyph."""
import cv2
import numpy as np

from .ocr_utils import _cascade, _ocr

# Suits are colour-coded by this client, so they are classified from the
# card's background colour rather than from per-suit pip templates. See
# classify_suit for how the four colours are told apart.
_RANK_WHITELIST = "23456789TJQKA"


# Spades render on a dark navy background, not truly black, so the only suit
# they can be confused with is the other blue one, diamonds.
#
# This used to be a brightness test -- anything dimmer than 80 in its
# brightest channel was a spade -- on the grounds that spades topped out at
# 55 and the dimmest non-spade started at 110. That gap is real for cards at
# full brightness and absent once the client dims them, which it does to the
# losing hands at showdown. A dimmed club measured 79 and a dimmed diamond
# 78, both inside the spade range, and a real board read "As 7c Jc 6s 5s"
# when it was Ac 7c Jc 6s 8c.
#
# Most of that damage is undone by asking the dominant-channel questions
# first: a club stays green-dominant and a heart stays red-dominant however
# far they are dimmed, and under the old order the brightness test answered
# for them before either question was put.
#
# That leaves spades against diamonds, the two blue suits, which does need a
# number. The gap between them is how far blue runs ahead of red -- a
# difference, not a ratio. A ratio separates the iPad cleanly (navy 1.70-2.38
# against 4.00-5.94) but not the iPhone, whose blue sits at 2.61 against a
# navy of 2.28. The difference separates both, because dimming scales the
# channels together and leaves navy's near-grey balance intact:
#
#     navy (spades)   b-r  7 -  25    across both devices
#     diamonds        b-r 59 - 113    dimmed ones included
#
# so the cutoff sits in the empty band between 25 and 59.
_SPADE_MAX_BLUE_OVER_RED = 40


def classify_suit(bgr_mean):
    b, g, r = bgr_mean
    if g >= b and g >= r:
        return "c"  # clubs: green
    if r > b and r > g:
        return "h"  # hearts: red
    # Both remaining suits are blue. Navy is near-grey; a diamond is not.
    return "s" if b - r < _SPADE_MAX_BLUE_OVER_RED else "d"


# Hero cards need a different test from the board's. Their suit is sampled
# from the exposed left edge of a fanned card rather than a clean corner, and
# the client dims them when a hand ends -- a dimmed diamond measured
# (60,40,23), inside the navy brightness range, so the board's brightness test
# calls it a spade. Ratios survive dimming where absolute levels do not: that
# card's blue is 2.6x its red, matching the bright diamond's 2.8x, while hero
# navy sits at 1.47-1.54.
#
# The board rule cannot simply adopt this: sampled from a corner, board navy
# runs 2.25-2.5 and board blue 2.61, which ratios cannot separate -- which is
# why classify_suit settles the same pair on a difference instead. Two
# sampling regions, two rules.
_HERO_DIAMOND_BLUE_RATIO = 2.0


def classify_hero_suit(bgr_mean):
    b, g, r = bgr_mean
    if r >= b and r >= g:
        return "h"
    if g >= b and g >= r:
        return "c"
    return "d" if b >= _HERO_DIAMOND_BLUE_RATIO * max(r, 1.0) else "s"


def sample_card_color(card_crop):
    """Median colour of a card's face, ignoring its white ink.

    Used for board cards in place of the corner patch. The slots are wider
    than the cards, so the rightmost one overhangs onto the blue table and a
    corner sample measures felt -- on a real frame that read the ace of spades
    as a diamond, which the backtest caught as the same card appearing in both
    the board and the hero's hand. A median ignores a minority of felt for the
    same reason it ignores the pips: neither is most of the crop.
    """
    pixels = card_crop.reshape(-1, 3)
    ink = (pixels[:, 0] > 150) & (pixels[:, 1] > 150) & (pixels[:, 2] > 150)
    face = pixels[~ink]
    if len(face) < 20:
        face = pixels
    return np.median(face, axis=0)


def _sample_bg_color(card_crop):
    h, w = card_crop.shape[:2]
    patch = card_crop[int(h * 0.06):int(h * 0.16), w - int(w * 0.16):w - int(w * 0.06)]
    if patch.size == 0:
        patch = card_crop
    return patch.reshape(-1, 3).mean(axis=0)


def is_card_present(crop, edge_density_threshold=0.02):
    """Heuristic: an empty felt background is smooth; a card (face-up or
    face-down back design) has much higher edge density."""
    if crop is None or crop.size == 0:
        return False
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 160)
    density = np.count_nonzero(edges) / edges.size
    return density > edge_density_threshold


def _extract_rank(text):
    """Pull a valid rank character out of a raw OCR string, or None."""
    cleaned = text.strip().upper().replace("O", "0")
    # A ten renders as the two characters "10", and neither is a rank. Map it
    # rather than strip it: dropping the zero leaves "1", which is not in the
    # whitelist either, so the read failed and the psm cascade kept going
    # until it produced some other valid rank -- a ten came back as an ace,
    # and no board card in the whole corpus ever read as T.
    if cleaned.startswith("10") or cleaned.startswith("1"):
        return "T"
    cleaned = cleaned.replace("0", "").strip()
    if not cleaned:
        return None
    rank = cleaned[0]
    return rank if rank in _RANK_WHITELIST else None


def _read_rank(mask, psms):
    """Upscale a black-text-on-white rank mask and try it under each psm in
    `psms` in turn (ocr_utils._cascade), taking the first valid rank read.
    Shared by _ocr_rank and _ocr_rank_bright, which differ only in how the
    mask itself gets built and in which psm order works best for that mask.
    """
    upscaled = cv2.resize(mask, (mask.shape[1] * 4, mask.shape[0] * 4), interpolation=cv2.INTER_CUBIC)
    return _cascade(psms, lambda psm: _extract_rank(_ocr(upscaled, whitelist=_RANK_WHITELIST, psm=psm)))


def _ocr_rank(region_bgr):
    if region_bgr.size == 0:
        return None
    gray = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2GRAY)
    # Rank glyph is bold white/light text on a darker background; compression
    # softens it by varying amounts (board cards vs. the hero's own card
    # overlay render at different sharpness), so an adaptive Otsu threshold
    # is more robust here than one fixed brightness cutoff.
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # psm 10 ("treat as a single character") is wrong surprisingly often on
    # this font -- confirmed on real hands: a clean, legible "5" and "6"
    # both came back empty at psm 10 whitelisted, while psm 8/13 read them
    # correctly.
    return _read_rank(mask, (10, 8, 13, 7, 6))


def read_rank(card_crop, rank_offset):
    """Read a flat card's rank from the corner box at `rank_offset`.

    The offset is passed in rather than read from a module constant so this
    file does not depend on any one device's pixel calibration -- that
    coupling is what kept the iPhone's regions.py loaded by the capture path.
    """
    dx1, dy1, dx2, dy2 = rank_offset
    corner = card_crop[dy1:dy2, dx1:dx2]
    return _ocr_rank(corner)


def _ocr_rank_bright(region_bgr, thresh=100):
    """Like _ocr_rank, but masks by a fixed brightness cutoff instead of
    adaptive Otsu thresholding. Used for hero's rotated cards (see
    read_rank_from_top): Otsu is a single threshold over the *whole* crop,
    which is more sensitive to a background gradient and other shapes
    (a suit pip, the neighboring overlapping card's edge) bleeding into a
    looser, position-tolerant crop than adaptive thresholding can absorb --
    confirmed on a real hand where Otsu misread a clearly-legible "K" as
    "7", but a fixed brightness mask read it correctly. psm 6 tried first:
    empirically the most reliable of the cascade on this glyph style,
    unlike flat/corner-cropped cards where psm 10 usually wins.
    """
    if region_bgr.size == 0:
        return None
    b = region_bgr[:, :, 0].astype(int)
    g = region_bgr[:, :, 1].astype(int)
    r = region_bgr[:, :, 2].astype(int)
    bright = (r > thresh) & (g > thresh) & (b > thresh)
    mask = (~bright).astype(np.uint8) * 255
    return _read_rank(mask, (6, 10, 8, 13, 7))


# Characters a "1" is read as when the whitelist is off, and characters a "0"
# is read as. A ten is the only two-character rank, so this shape is unique to
# it -- checked against every other card in the corpus, none of which produced
# anything matching.
_TEN_FIRST = set("1470lIitT|")
_TEN_LAST = set("0OoDQ")


def looks_like_ten(text):
    """Whether an unwhitelisted read of a rank glyph is the two-character ten."""
    cleaned = text.strip()
    if len(cleaned) != 2:
        return False
    return cleaned[0] in _TEN_FIRST and cleaned[1] in _TEN_LAST


def _read_ten(card_crop, fraction=0.62):
    """One unwhitelisted read at a window wide enough to show a whole "10".

    The whitelist is what hides tens: constrained to single rank characters,
    Tesseract can never report the two it actually sees, so it picks the
    closest single one and the cascade accepts it. Reading once without the
    whitelist is the only way to see the shape.
    """
    h = card_crop.shape[0]
    region = card_crop[0:int(h * fraction), :]
    if region.size == 0:
        return False
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    upscaled = cv2.resize(cv2.bitwise_not(binary), None, fx=3, fy=3,
                          interpolation=cv2.INTER_CUBIC)
    return looks_like_ten(_ocr(upscaled, psm=10))


def isolate_rank_glyph(card_crop):
    """A tight binary crop of just the rank glyph, or None.

    Handing OCR a window of the card's top gives it the rank, the suit pips
    and a lot of empty felt -- and the pips are several times larger than the
    glyph, so they dominate. On a real frame a 5 came back as a 3 that way.
    Cutting the glyph out first removes everything it was getting confused by.

    The brightness cutoff is derived from the crop rather than fixed, because
    a card face runs from navy through green to dimmed blue and a single
    threshold finds the glyph on some and nothing on others.
    """
    found = _rank_components(card_crop)
    if found is None:
        return None
    mask, parts = found
    # The rank sits above and left of everything else on the card.
    x, y, bw, bh = min(parts, key=lambda p: p[1] + p[0] * 0.5)
    return mask[y:y + bh, x:x + bw] * 255


def _rank_components(card_crop):
    """(mask, parts) for the glyph-sized blobs in the card's top-left corner.

    Each part is (x, y, w, h) into `mask`. Split out of isolate_rank_glyph so
    that isolate_rank_pair can reuse the same blobs rather than re-deriving
    them from a second, possibly different, threshold.
    """
    h, w = card_crop.shape[:2]
    top = card_crop[0:int(h * 0.55), 0:int(w * 0.70)]
    if top.size == 0:
        return None
    gray = cv2.cvtColor(top, cv2.COLOR_BGR2GRAY)
    threshold = max(gray.mean() + 1.6 * gray.std(), 60)
    mask = (gray > threshold).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    parts = []
    for i in range(1, count):
        x, y = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
        bw, bh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        if not (top.shape[0] * 0.20 <= bh <= top.shape[0] * 0.85):
            continue
        if bw < 5 or area < 60 or area < 0.12 * bw * bh:
            continue
        parts.append((x, y, bw, bh))
    return None if not parts else (mask, parts)


# How far apart two characters of one rank may sit, as a fraction of the left
# character's width; how much of their height must overlap; and how different
# their heights may be. A "10" is one glyph drawn as two characters, so its
# halves are near-touching, share a baseline and are the same height. The
# bounds are deliberately tight: the only thing to the right of a rank that
# could be mistaken for its other half is the neighbouring card's rank, which
# on a fanned hand sits much further away.
_PAIR_MAX_GAP = 0.6
_PAIR_MIN_OVERLAP = 0.6
_PAIR_HEIGHT_RATIO = (0.7, 1.4)


def isolate_rank_pair(card_crop):
    """The rank as TWO adjacent characters -- i.e. a "10" -- or None.

    isolate_rank_glyph returns one connected component, which is right for
    twelve of the thirteen ranks and wrong for the ten: it hands back the "1"
    alone, and a lone "1" scores as a J. A live showdown opened Td Tc and the
    board was lost from its first slot that way.

    This is offered as an extra candidate rather than a replacement. When the
    card is not a ten there is no second character to find and this returns
    None; when the pairing is spurious the merged shape matches no rank well,
    and the caller keeps whichever candidate scores highest.
    """
    found = _rank_components(card_crop)
    if found is None:
        return None
    mask, parts = found
    if len(parts) < 2:
        return None

    left, *rest = sorted(parts, key=lambda p: p[0])
    lx, ly, lw, lh = left
    for rx, ry, rw, rh in rest:
        gap = rx - (lx + lw)
        overlap = min(ly + lh, ry + rh) - max(ly, ry)
        low, high = _PAIR_HEIGHT_RATIO
        if not 0 <= gap <= _PAIR_MAX_GAP * lw:
            continue
        if overlap < _PAIR_MIN_OVERLAP * min(lh, rh):
            continue
        if not low <= rh / lh <= high:
            continue
        x1, y1 = min(lx, rx), min(ly, ry)
        x2, y2 = max(lx + lw, rx + rw), max(ly + lh, ry + rh)
        return mask[y1:y2, x1:x2] * 255
    return None


def _read_isolated(glyph):
    upscaled = cv2.resize(cv2.bitwise_not(glyph), None, fx=4, fy=4,
                          interpolation=cv2.INTER_CUBIC)
    upscaled = cv2.copyMakeBorder(upscaled, 20, 20, 20, 20,
                                  cv2.BORDER_CONSTANT, value=255)
    return _cascade((10, 7, 8), lambda psm: _extract_rank(
        _ocr(upscaled, whitelist=_RANK_WHITELIST, psm=psm)))


def read_rank_from_top(card_crop, top_fractions=(0.4, 0.45, 0.5, 0.55, 0.6, 0.65)):
    """Like read_rank, but for a card whose rank doesn't sit at a reliable
    fixed corner offset (a rotated/fanned card -- see HERO_CARDS): searches
    a top portion of the whole crop instead of one tiny fixed sub-box.

    Tries a cascade of window sizes rather than one fixed fraction: the
    card's vertical position within its (deliberately generous) box isn't
    fixed frame to frame -- confirmed on two real hands where the rank sat
    at different depths. A single window big enough to reach the lower
    position also drags a suit pip into the crop for the higher position,
    confusing the OCR into misreading a rank that was already legible on
    its own.

    Known remaining gap: a dim/faded render state (seen on a post-showdown
    frame) can still misread even with this cascade -- not yet solved.
    """
    # Tens first: the cascade below cannot find them, because its smaller
    # windows crop the two-character glyph and return a confident wrong
    # single letter before a larger window gets a chance.
    if _read_ten(card_crop):
        return "T"

    # Cut the glyph out and read that alone -- see isolate_rank_glyph.
    #
    # Read with Tesseract, not the templates, and this was measured rather
    # than assumed. On a fanned card the templates separate badly: their top
    # score sits at 0.62-0.66 with a margin of 0.01-0.03 over the runner-up,
    # so the right answer and two wrong ones are indistinguishable. Across
    # hands checked by eye Tesseract reads 20/21 against the templates' 16.
    # A live 8 of spades that Tesseract called a 3 is the one it loses, and
    # preferring templates to fix it broke three other cards.
    #
    # Note this is the opposite of the BOARD, where templates win outright --
    # those cards are flat and unobstructed, so the same glyph shape scores
    # 0.85-0.99 with room to spare.
    glyph = isolate_rank_glyph(card_crop)
    if glyph is not None:
        # A confident fanned-exemplar match beats OCR; below that bar OCR
        # decides. Neither reads every hand alone -- the templates get an 8
        # of spades that Tesseract calls a 3, Tesseract gets a K the
        # templates call an A -- and their failures are separable because a
        # template mistake scores low while its successes score near 1.
        from . import rank_templates
        matched, score = rank_templates.match_fanned(glyph)
        if matched is not None and score >= rank_templates.FANNED_MIN_SCORE:
            return matched
        rank = _read_isolated(glyph)
        if rank is not None:
            return rank

    # Isolation can come up empty on a card this has not seen; the original
    # window cascade still stands behind it.
    h = card_crop.shape[0]
    for fraction in top_fractions:
        rank = _ocr_rank_bright(card_crop[0:int(h * fraction), :])
        if rank is not None:
            return rank
    return None


def parse_card(card_crop, rank_offset):
    """Given a cropped card slot image, return e.g. '7c', 'Th', or None if
    the slot is empty / unreadable."""
    if not is_card_present(card_crop):
        return None
    suit = classify_suit(_sample_bg_color(card_crop))
    rank = read_rank(card_crop, rank_offset)
    if rank is None:
        return None
    return f"{rank}{suit}"


def _sample_center_color(card_crop):
    """Suit-background sample for a rotated card (see parse_hero_card):
    _sample_bg_color's fixed top-right corner assumes an upright card and
    can land off the actual (rotated) card surface entirely -- confirmed
    on a real hand, where it sampled empty background and misclassified a
    diamond as a spade. The crop's center is reliably on the card
    regardless of rotation direction, as long as the crop reasonably
    bounds the card.
    """
    h, w = card_crop.shape[:2]
    patch = card_crop[int(h * 0.35):int(h * 0.55), int(w * 0.35):int(w * 0.65)]
    if patch.size == 0:
        patch = card_crop
    return patch.reshape(-1, 3).mean(axis=0)


def sample_face_color(card_crop):
    """Average colour of a fanned card's own face.

    Samples the left portion rather than the centre. The hero's two cards
    overlap, so the one behind has its middle covered by the one in front and
    a centre sample averages the two together -- that read a 9 of spades as a
    diamond, and read dimmed red queens as spades. The left edge is the part
    of a fanned card that stays exposed.

    Swept over all 122 hero cards in the real screenshots: 13 suits changed,
    and every one checked by eye was a correction.
    """
    h, w = card_crop.shape[:2]
    patch = card_crop[int(h * 0.35):int(h * 0.75), int(w * 0.05):int(w * 0.35)]
    if patch.size == 0:
        patch = card_crop
    return patch.reshape(-1, 3).mean(axis=0)


def parse_hero_card(card_crop):
    """Like parse_card, but for one of hero's own (fanned/rotated) cards --
    see HERO_CARDS. HERO_CARDS' boxes are deliberately generous (rotated
    cards don't sit at one exact fixed position frame to frame -- confirmed
    on two real hands with a ~65px vertical difference), so both the rank
    search and the suit sample use rotation/position-tolerant strategies
    instead of parse_card's fixed corner offsets.
    """
    if not is_card_present(card_crop):
        return None
    suit = classify_hero_suit(sample_face_color(card_crop))
    rank = read_rank_from_top(card_crop)
    if rank is None:
        return None
    return f"{rank}{suit}"
