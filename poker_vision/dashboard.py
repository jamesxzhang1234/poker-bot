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


_AGGRESSIVE = ("Bet", "Raise", "All In")


def betting_line(state):
    """Who has put pressure on this street, in order, e.g.
    "prolapsedvagina raise 0.4 -> borpa re-raise 1.2", or "" when nobody has.

    Read from one frame, no memory. Within a betting round each player's
    bubble stays up and their chips on the felt total what they have put in,
    so a raise and the re-raise over it are both on screen at once: ordered
    by amount, the smaller is the opener. The limit is honest and visible --
    once the opener calls the re-raise, their bubble says Call and their
    chips match, so the original 0.4 is no longer on screen and is not shown.
    """
    moves = sorted((s for s in state.seats
                    if s.status in _AGGRESSIVE and s.bet is not None),
                   key=lambda s: s.bet)
    parts, raised = [], False
    for seat in moves:
        if seat.status == "All In":
            word = "all-in"
        elif seat.status == "Bet" and not raised:
            word = "bet"
        else:
            word = "re-raise" if raised else "raise"
        raised = raised or seat.status in ("Bet", "Raise")
        parts.append(f"{seat.name or '?'} {word} {_amount(seat.bet)}")
    return " -> ".join(parts)


def render(state, previous=None, frames=None, latency_ms=None, status=None,
           banner=None):
    """The dashboard block for one frame.

    `status` marks the display as HELD: the table shown is the last one that
    arrived, not a live read. That happens when the iPad locks (the capture
    interface is withdrawn entirely, so no frames come at all) or when the
    screen is showing something that is not a table. Saying so matters more
    than it sounds -- a frozen table that looks live is the one failure that
    could actually be acted on.

    `banner` is a first line saying what kind of snapshot this is -- player
    mode uses it, because a table that only redraws on your turn must never
    pass for a live one.
    """
    previous_by_seat = ({s.seat_id: s for s in previous.seats} if previous else None)

    board = " ".join(_format_card(c) for c in state.board_cards) or "(none)"
    pot = _amount(state.pot)
    flag = "  ! bets > pot" if _pot_mismatch(state) else ""

    live = sum(1 for s in state.seats if s.in_hand)
    header = f"  Pot {pot:<12}Board  {board}{flag}"
    if status:
        header = f"  [HELD] {status}\n" + header
    lines = [header]
    action = betting_line(state)
    if action:
        lines.append(f"  Action  {action}")
    if banner:
        lines.insert(0, f"  {banner}")
    if state.your_turn:
        # 0 and None mean different things here (see find_action): nothing
        # to call, versus a price that could not be read. "to call 0" states
        # the first accurately and unhelpfully.
        if state.to_call is None:
            owed = ""
        elif state.to_call == 0:
            owed = "   check is free"
        else:
            owed = f"   to call {state.to_call:g}"
        lines.append(f"  >>> YOUR TURN{owed}")
    lines += [f"  {live} in hand", "  " + "-" * _WIDTH]

    for seat in sorted(state.seats, key=lambda s: _position_rank(s.position)):
        moved = _changed(seat, previous_by_seat)
        mark = "*" if moved else " "
        position = (seat.position or "--")
        hero = "<" if seat.is_hero else " "
        # A seat that has folded is dimmed to a marker, so the players still
        # contesting the pot read at a glance -- that count drives almost
        # every decision.
        out = " " if seat.in_hand else "."
        cards = ""
        if seat.hole_cards:
            cards = "[" + " ".join(_format_card(c) for c in seat.hole_cards) + "] "
        lines.append(
            f" {out}{position:>5}{hero} {str(seat.name or '?'):<16}"
            f"{_amount(seat.stack):>8} {mark}  {cards}{_action(seat)}".rstrip())

    lines.append("  " + "-" * _WIDTH)
    footer = f"  frame {state.frame_index}"
    if frames is not None:
        footer += f"   {frames} captured"
    if latency_ms is not None:
        footer += f"   {latency_ms:.0f}ms behind"
    footer += "   * changed   < you   . folded"
    lines.append(footer)
    return "\n".join(lines)


def render_waiting_for_turn():
    """Player mode, with a table on screen but no turn of yours seen yet.

    Without this the panel would keep saying "waiting for a poker table" while
    a table is plainly up, which reads as a broken capture.
    """
    lines = ["  " + "-" * _WIDTH,
             "  PLAYER MODE - table found, waiting for your turn",
             "",
             "  the table is being read live; it is shown here",
             "  only when it is your turn to act",
             "  " + "-" * _WIDTH, "  ctrl-c to stop"]
    return "\n".join(lines)


def render_waiting(status=None, connected=False):
    """The panel shown before any table has been seen.

    Two different situations, and conflating them sends the user after the
    wrong problem. Not connected: the device is locked, unplugged, or held by
    QuickTime, and the setup hints are what they need. Connected but no table:
    they are sitting in the lobby, nothing is broken, and the thing to do is
    open a table.
    """
    lines = ["  " + "-" * _WIDTH]
    if connected:
        lines += ["  connected - waiting for a poker table",
                  "",
                  f"  {status}" if status else "  capturing",
                  "",
                  "  open a table in PurePoker and it will appear here"]
    else:
        lines += ["  waiting for the iPad...",
                  "",
                  f"  {status}" if status else "  starting capture",
                  "",
                  "  it must be unlocked and plugged in over USB,",
                  "  and QuickTime Player must not be running"]
    lines += ["  " + "-" * _WIDTH, "  ctrl-c to stop"]
    return "\n".join(lines)


CLEAR = "\033[H\033[J"


def draw_to_terminal(text):
    """Rewrite the block in place."""
    import sys
    sys.stdout.write(CLEAR + text + "\n")
    sys.stdout.flush()
