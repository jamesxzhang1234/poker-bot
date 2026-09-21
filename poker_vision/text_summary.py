"""Formats a parsed TableState dict (see TableState.to_dict()) as a compact
human-readable summary for the server's stdout output.
"""


def format_summary(state):
    lines = [f"--- frame {state['frame_index']} @ {state['timestamp_sec']:.1f}s ---"]

    pot = state.get("pot")
    lines.append(f"Pot: {pot if pot is not None else '?'}")

    board = state.get("board_cards") or []
    lines.append(f"Board: {' '.join(board) if board else '(none)'}")

    for seat in state.get("seats", []):
        tag = "HERO" if seat.get("is_hero") else seat.get("seat_id", "?")
        parts = [tag]
        if seat.get("name"):
            parts.append(seat["name"])
        if seat.get("position"):
            parts.append(f"[{seat['position']}]")
        if seat.get("stack") is not None:
            parts.append(f"${seat['stack']}")
        if seat.get("status"):
            parts.append(f"({seat['status']})")
        if seat.get("hole_cards"):
            parts.append(" ".join(seat["hole_cards"]))
        lines.append("  " + " ".join(parts))

    return "\n".join(lines)
