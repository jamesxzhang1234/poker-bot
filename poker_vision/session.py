"""Operator-entered session facts, shared by every capture path.

The blind sizes are typed in at launch and written out so a later
decision component can read the stakes without re-parsing a command line.
Nothing here is specific to how frames arrive, which is why it does not live
with either the archived push server or the live capture runner.
"""
import json
import re
from datetime import datetime
from pathlib import Path

DEFAULT_SESSION_FILE = Path("frames/session.json")

_BLINDS_PATTERN = re.compile(r"^(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)$")


def parse_blinds(text):
    """Parse a 'SB/BB' blind-size string (e.g. '1/2') into (sb, bb) floats.
    Raises ValueError with a clear message on malformed input."""
    match = _BLINDS_PATTERN.match(text.strip())
    if not match:
        raise ValueError(f"blinds must be in SB/BB format like '1/2', got {text!r}")
    return float(match.group(1)), float(match.group(2))


def write_session_info(path, blinds_raw, small_blind, big_blind):
    """Persist the operator-provided blind sizes for this capture session
    so a later consumer (e.g. a future decision/solver component) can read
    them without re-parsing the command line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "blinds": blinds_raw,
        "small_blind": small_blind,
        "big_blind": big_blind,
        "started_at": datetime.now().isoformat(),
    }, indent=2))


