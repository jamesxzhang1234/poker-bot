"""Fills in each seat's poker position from one badge read in the same frame.

Lifted out of parser.py so the capture path does not have to import the
fixed-region parser -- and through it the iPhone pixel calibration -- just to
name positions. Nothing here touches pixels; it is pure poker: given which
seats are occupied and one seat whose D/SB/BB badge was read, work out what
everyone else is.
"""
from . import positions, seat_layout


_RANK_ROLES = {"D": 0, "SB": 1, "BB": 2}


def _is_occupied(seat):
    # Deliberately excludes in_hand: the card-presence heuristic it comes
    # from (edge density in the card-slot crop) can false-trigger on a
    # background artifact in a genuinely empty seat -- confirmed against a
    # real screenshot where an "Invite" (empty) seat showed in_hand=True
    # with no name or stack, and got pulled into the rotation as a result.
    # A real player always shows at least a name or a stack.
    return bool(seat.name) or seat.stack is not None



def _dealt_in(order, by_id):
    """The occupied seats that are actually in the rotation this hand.

    A seat can hold a name and a stack and still not be dealt in: sitting
    out, or just sat down and waiting for the big blind. Counting it shifts
    every inferred position after it -- live, a waiting player was named the
    dealer. Two signals remove one, both read in this frame:

      * the seat says "Sitting Out";
      * it lies between two blinds/button that were read directly. D, SB and
        BB go to consecutive dealt-in seats, so anyone the rotation passes
        between them is not playing. Once the badges came from the text mask
        as well as Vision, D/SB/BB were read together on most frames, and the
        17 frames where they did not line up all had exactly such a player.
    """
    order = [sid for sid in order
             if (by_id[sid].status or "").lower() != "sitting out"]
    read = {by_id[sid].position: sid for sid in order
            if by_id[sid].position in _RANK_ROLES}
    skip = set()
    for first, second in (("D", "SB"), ("SB", "BB")):
        if first in read and second in read:
            i, j = order.index(read[first]), order.index(read[second])
            k = (i + 1) % len(order)
            while k != j:
                skip.add(order[k])
                k = (k + 1) % len(order)
    return [sid for sid in order if sid not in skip]


def _middle_position_names(seats_between):
    if seats_between in positions.MIDDLE_POSITION_NAMES:
        return positions.MIDDLE_POSITION_NAMES[seats_between]
    # Beyond this game's 9-seat max (shouldn't happen) -- extrapolate with
    # plain UTG+k numbering rather than guessing at further named slots.
    return ["UTG"] + [f"UTG+{i}" for i in range(1, seats_between)]


def infer_positions(seats):
    """Fill in each occupied seat's poker position (D/SB/BB, then
    UTG/.../HJ/CO for the rest) from another seat's badge in the same
    frame, when that seat's own badge OCR came back empty.

    Single-frame only (no history across screenshots): rotates through the
    occupied seats in seat_layout.SEAT_ORDER's action-order direction, anchored
    on whichever seat had ANY position successfully read by OCR. Never
    overwrites a position OCR already found, and never assigns one to a
    seat with no player in it. Heads-up (exactly 2 occupied seats, where
    the dealer also posts the small blind) is out of scope -- unverified,
    revisit if it comes up.
    """
    by_id = {s.seat_id: s for s in seats}
    occupied_order = [sid for sid in seat_layout.SEAT_ORDER if sid in by_id and _is_occupied(by_id[sid])]
    occupied_order = _dealt_in(occupied_order, by_id)
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
