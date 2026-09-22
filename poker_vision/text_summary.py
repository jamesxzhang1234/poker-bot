"""Formats a parsed TableState dict (see TableState.to_dict()) as a compact
human-readable summary for the server's stdout output.
"""

# Action-order reference for sorting the printed seat list (D first, then
# SB, BB, ... around to CO) -- matches parser.infer_positions's naming.
_POSITION_ORDER = ["D", "SB", "BB", "UTG", "UTG+1", "UTG+2", "LJ", "HJ", "CO"]

# Suit letter -> symbol, matching card_vision.classify_suit's codes.
_SUIT_SYMBOLS = {"c": "♣", "d": "♦", "h": "♥", "s": "♠"}

# Spelled out for readability as the leading label; other positions
# (SB, BB, UTG, ...) are already short enough to print as-is.
_POSITION_LABELS = {"D": "Dealer"}


def _format_card(card):
    if card and card[-1] in _SUIT_SYMBOLS:
        return card[:-1] + _SUIT_SYMBOLS[card[-1]]
    return card


def _position_rank(position):
    if position in _POSITION_ORDER:
        return _POSITION_ORDER.index(position)
    if position and position.startswith("UTG+"):
        try:
            return 3 + int(position.split("+", 1)[1])
        except ValueError:
            pass
    return len(_POSITION_ORDER)  # unknown/unset position sorts last


def format_summary(state):
    lines = [f"--- frame {state['frame_index']} @ {state['timestamp_sec']:.1f}s ---"]

    pot = state.get("pot")
    lines.append(f"Pot: {pot if pot is not None else '?'}")

    board = state.get("board_cards") or []
    lines.append(f"Board: {' '.join(_format_card(c) for c in board) if board else '(none)'}")

    seats = sorted(state.get("seats", []), key=lambda s: _position_rank(s.get("position")))
    for seat in seats:
        position = seat.get("position")
        is_hero = seat.get("is_hero")
        # Internal screen-slot names (e.g. "lower_mid_right") never appear
        # in the output -- the resolved poker position is the identifier.
        # Hero still needs the bracket since "HERO" itself doesn't convey
        # position; every other seat's tag already IS its position, so
        # repeating it in a bracket would just be redundant.
        if is_hero:
            tag = "HERO"
        else:
            tag = _POSITION_LABELS.get(position, position) or "?"
        parts = [tag]
        if seat.get("name"):
            parts.append(seat["name"])
        if is_hero and position:
            parts.append(f"[{position}]")
        if seat.get("stack") is not None:
            parts.append(f"${seat['stack']}")
        if seat.get("status"):
            parts.append(f"({seat['status']})")
        if seat.get("hole_cards"):
            parts.append(" ".join(_format_card(c) for c in seat["hole_cards"]))
        lines.append("  " + " ".join(parts))

    return "\n".join(lines)
