"""In-place live dashboard for the Vision capture path.

Renders a TableState as a fixed block that gets rewritten each frame, so the
terminal shows the table as it stands rather than a scrolling history.

Rendering is pure: state in, string out. The caller owns the terminal.

On the single-frame rule: this module is handed the *previous* state purely to
mark what moved. That memory lives in the display and goes nowhere near
parsing -- nothing here is inferred across frames or fed back into a
TableState, which is what the rule in CLAUDE.md is protecting against.
"""
from .text_summary import _format_card, _position_rank

# How far past the pot the felt chips must total before it counts as a fault.
# Chips land on the felt a frame before the pot total absorbs them, so during
# the animation the bets legitimately run a blind or so ahead -- seen on 2 of
# 30 real frames. A genuine misread is not subtle by comparison: the hero's
# own hole card being counted as a bet put 9.05 against a pot of 0.15. A
# factor of two separates the two cases cleanly.
_POT_OVERSHOOT_FACTOR = 2.0

_WIDTH = 52


def _amount(value):
    return "?" if value is None else f"{value:g}"


def _action(seat):
    """The action bubble plus what it cost, e.g. "(Raise : 1.7)"."""
    if not seat.status:
        return ""
    if seat.bet is None:
        return f"({seat.status})"
    return f"({seat.status} : {seat.bet:g})"


def _changed(seat, previous_by_seat):
    """Which of this seat's fields moved since the last frame.

    Keyed on seat_id, not the handle: Vision returns "jxz" and "jXZ" for the
    same player between frames, and keying on the name made a case flip look
    like a different player, so that seat never showed a change at all.
    """
    if previous_by_seat is None:
        return set()
    before = previous_by_seat.get(seat.seat_id)
    if before is None:
        return set()
    moved = set()
    for field in ("stack", "status", "bet", "position"):
        if getattr(before, field) != getattr(seat, field):
            moved.add(field)
    return moved


def _pot_mismatch(state):
    """True when the chips on the felt total MORE than the pot.

    Only this direction is a fault. The pot is cumulative across streets while
    the felt shows only the current one, so from the flop onwards the bets are
    supposed to total less -- checking for equality fires on nearly every
    post-flop frame. Exceeding the pot is impossible though, since the pot
    already contains this street's chips, so it means something was misread:
    it is what caught the hero's own hole card being counted as a 9.0 bet.
    """
    if not state.pot:
        return False
    bets = [s.bet for s in state.seats if s.bet is not None]
    if not bets:
        return False
    return sum(bets) > _POT_OVERSHOOT_FACTOR * state.pot


def render(state, previous=None, frames=None, latency_ms=None):
    """The dashboard block for one frame."""
    previous_by_seat = ({s.seat_id: s for s in previous.seats} if previous else None)

    board = " ".join(_format_card(c) for c in state.board_cards) or "(none)"
    pot = _amount(state.pot)
    flag = "  ! bets > pot" if _pot_mismatch(state) else ""

    lines = [f"  Pot {pot:<12}Board  {board}{flag}", "  " + "-" * _WIDTH]

    for seat in sorted(state.seats, key=lambda s: _position_rank(s.position)):
        moved = _changed(seat, previous_by_seat)
        mark = "*" if moved else " "
        position = (seat.position or "--")
        hero = "<" if seat.is_hero else " "
        lines.append(
            f"  {position:>5}{hero} {str(seat.name or '?'):<16}"
            f"{_amount(seat.stack):>8} {mark}  {_action(seat)}")

    lines.append("  " + "-" * _WIDTH)
    footer = f"  frame {state.frame_index}"
    if frames is not None:
        footer += f"   {frames} captured"
    if latency_ms is not None:
        footer += f"   {latency_ms:.0f}ms"
    footer += "   * = changed    < = you"
    lines.append(footer)
    return "\n".join(lines)


def render_waiting(status=None):
    """The panel shown before any frame has arrived.

    The live feed not being up is an ordinary state, not an error: the iPad
    locks, QuickTime gets opened, the cable gets knocked. Say what is wrong
    and what fixes it, and keep waiting.
    """
    lines = ["  " + "-" * _WIDTH,
             "  waiting for the table...",
             ""]
    if status:
        lines.append(f"  {status}")
    else:
        lines.append("  starting capture")
    lines += ["",
              "  the iPad must be unlocked, plugged in over USB,",
              "  and QuickTime Player must not be running",
              "  " + "-" * _WIDTH,
              "  ctrl-c to stop"]
    return "\n".join(lines)


CLEAR = "\033[H\033[J"


def draw_to_terminal(text):
    """Rewrite the block in place."""
    import sys
    sys.stdout.write(CLEAR + text + "\n")
    sys.stdout.flush()
