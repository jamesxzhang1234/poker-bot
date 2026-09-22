"""Turns a raw table screenshot/frame into a TableState."""
from . import card_vision, number_ocr, ocr_utils, regions
from .table_state import Seat, TableState


def _crop(frame, box_ref):
    if box_ref is None:
        return None
    frame_h, frame_w = frame.shape[:2]
    x1, y1, x2, y2 = regions.scale_box(box_ref, frame_w, frame_h)
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, frame_w), min(y2, frame_h)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def _read_badge(frame, badge_boxes):
    """OCR a seat's badge from its list of candidate boxes (see
    regions.SEATS's "badge" docstring -- some seats render the badge in
    more than one position depending on whether the player is currently
    holding cards); tries each in order and returns the first recognized
    D/SB/BB, or None if none of them read.
    """
    for box in badge_boxes:
        position = ocr_utils.ocr_badge(_crop(frame, box))
        if position is not None:
            return position
    return None


_RANK_ROLES = {"D": 0, "SB": 1, "BB": 2}


def _is_occupied(seat):
    # Deliberately excludes in_hand: the card-presence heuristic it comes
    # from (edge density in the card-slot crop) can false-trigger on a
    # background artifact in a genuinely empty seat -- confirmed against a
    # real screenshot where an "Invite" (empty) seat showed in_hand=True
    # with no name or stack, and got pulled into the rotation as a result.
    # A real player always shows at least a name or a stack.
    return bool(seat.name) or seat.stack is not None



# Standard poker position names for the seats between BB and the button, in
# action order (closest to BB first), keyed by how many such seats there
# are. Matches common charts (6-max = UTG,HJ,CO; 9-max = UTG,UTG+1,UTG+2,
# LJ,HJ,CO); this game's seat layout maxes out at 9 total seats (8 opponent
# slots + hero), so 6 is the largest key needed (9 - D - SB - BB).
_MIDDLE_POSITION_NAMES = {
    0: [],
    1: ["UTG"],
    2: ["UTG", "CO"],
    3: ["UTG", "HJ", "CO"],
    4: ["UTG", "UTG+1", "HJ", "CO"],
    5: ["UTG", "UTG+1", "LJ", "HJ", "CO"],
    6: ["UTG", "UTG+1", "UTG+2", "LJ", "HJ", "CO"],
}


def _middle_position_names(seats_between):
    if seats_between in _MIDDLE_POSITION_NAMES:
        return _MIDDLE_POSITION_NAMES[seats_between]
    # Beyond this game's 9-seat max (shouldn't happen) -- extrapolate with
    # plain UTG+k numbering rather than guessing at further named slots.
    return ["UTG"] + [f"UTG+{i}" for i in range(1, seats_between)]


def infer_positions(seats):
    """Fill in each occupied seat's poker position (D/SB/BB, then
    UTG/.../HJ/CO for the rest) from another seat's badge in the same
    frame, when that seat's own badge OCR came back empty.

    Single-frame only (no history across screenshots): rotates through the
    occupied seats in regions.SEAT_ORDER's action-order direction, anchored
    on whichever seat had ANY position successfully read by OCR. Never
    overwrites a position OCR already found, and never assigns one to a
    seat with no player in it. Heads-up (exactly 2 occupied seats, where
    the dealer also posts the small blind) is out of scope -- unverified,
    revisit if it comes up.
    """
    by_id = {s.seat_id: s for s in seats}
    occupied_order = [sid for sid in regions.SEAT_ORDER if sid in by_id and _is_occupied(by_id[sid])]
    n = len(occupied_order)
    if n < 3:
        return seats

    anchor_id = anchor_role = None
    for sid in occupied_order:
        pos = by_id[sid].position
        if pos in _RANK_ROLES:
            anchor_id, anchor_role = sid, pos
            break
    if anchor_id is None:
        return seats

    # A role already held by some other seat's own OCR read means the
    # rotation model doesn't fit this hand's occupied-seat set -- drop that
    # inferred assignment rather than duplicate the role onto a second seat.
    roles_taken = {s.position for s in seats if s.position in _RANK_ROLES}

    names_by_rank = {rank: role for role, rank in _RANK_ROLES.items()}
    for i, name in enumerate(_middle_position_names(n - 3)):
        names_by_rank[3 + i] = name

    anchor_idx = occupied_order.index(anchor_id)
    anchor_rank = _RANK_ROLES[anchor_role]
    for rank, role in names_by_rank.items():
        if role in roles_taken:
            continue
        sid = occupied_order[(anchor_idx + (rank - anchor_rank)) % n]
        seat = by_id[sid]
        if seat.position is None:
            seat.position = role
    return seats


class TableParser:
    def __init__(self):
        self.seat_defs = regions.SEATS

    def parse_seat(self, frame, seat_id):
        cfg = self.seat_defs[seat_id]
        name = ocr_utils.ocr_name(_crop(frame, cfg["name_row"])) or None
        stack = number_ocr.ocr_number(_crop(frame, cfg["stack_row"]))
        position = _read_badge(frame, cfg["badge"])

        bubble_box, label_box = regions.status_sub_zones(cfg)
        status = ocr_utils.ocr_bubble_action(_crop(frame, bubble_box))
        if status is None:
            status = ocr_utils.ocr_persistent_label(_crop(frame, label_box))

        cards_crop = _crop(frame, cfg.get("cards"))
        in_hand = card_vision.is_card_present(cards_crop) if cards_crop is not None else False
        if status == "Folded":
            in_hand = False

        hole_cards = []
        is_hero = seat_id == "hero"
        if is_hero:
            hero_cards = [card_vision.parse_hero_card(_crop(frame, box)) for box in regions.HERO_CARDS]
            hole_cards = [c for c in hero_cards if c]
            in_hand = len(hole_cards) > 0 or in_hand

        return Seat(
            seat_id=seat_id,
            name=name,
            stack=stack,
            position=position,
            status=status,
            in_hand=in_hand,
            is_hero=is_hero,
            hole_cards=hole_cards,
        )

    def parse_board(self, frame):
        cards = []
        for slot in regions.BOARD_SLOTS:
            crop = _crop(frame, slot)
            card = card_vision.parse_card(crop) if crop is not None else None
            if card:
                cards.append(card)
        return cards

    def parse_pot(self, frame):
        return number_ocr.ocr_number(_crop(frame, regions.POT_BOX))

    def parse(self, frame, frame_index=0, timestamp_sec=0.0):
        seats = [self.parse_seat(frame, seat_id) for seat_id in self.seat_defs]
        seats = infer_positions(seats)
        board = self.parse_board(frame)
        pot = self.parse_pot(frame)
        return TableState(
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
            seats=seats,
            board_cards=board,
            pot=pot,
        )
