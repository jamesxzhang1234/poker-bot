"""Renders the live debug view: the captured frame with region boxes drawn
on it, plus a sidebar of the latest parsed table state.
"""
import time

import cv2

from . import regions

SIDEBAR_WIDTH = 420
BG = (30, 24, 20)
FG = (230, 230, 230)
DIM = (140, 140, 140)
ACCENT = (80, 220, 255)
GOOD = (120, 220, 120)


def draw_regions(frame, state):
    """Draw calibrated region boxes on a copy of frame, labeled with the
    most recently parsed value for each, so you can see what the pipeline
    is reading."""
    out = frame.copy()
    h, w = out.shape[:2]

    seat_by_id = {}
    if state:
        seat_by_id = {s["seat_id"]: s for s in state.get("seats", [])}

    for seat_id, cfg in regions.SEATS.items():
        seat = seat_by_id.get(seat_id)
        color = GOOD if (seat and seat.get("in_hand")) else DIM

        for key in ("name_row", "stack_row", "cards", "badge"):
            box = cfg.get(key)
            if box is None:
                continue
            x1, y1, x2, y2 = regions.scale_box(box, w, h)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 1)

        if seat:
            ax1, ay1, ax2, ay2 = regions.scale_box(cfg["avatar"], w, h)
            label = seat.get("name") or seat_id
            if seat.get("stack") is not None:
                label += f"  {seat['stack']:g}"
            if seat.get("position"):
                label += f"  [{seat['position']}]"
            cv2.putText(out, label, (ax1, max(ay1 - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, ACCENT, 1, cv2.LINE_AA)
            if seat.get("status"):
                cv2.putText(out, seat["status"], (ax1, ay2 + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1, cv2.LINE_AA)

    for slot in regions.BOARD_SLOTS:
        x1, y1, x2, y2 = regions.scale_box(slot, w, h)
        cv2.rectangle(out, (x1, y1), (x2, y2), ACCENT, 1)

    for box in regions.HERO_CARDS:
        x1, y1, x2, y2 = regions.scale_box(box, w, h)
        cv2.rectangle(out, (x1, y1), (x2, y2), ACCENT, 1)

    x1, y1, x2, y2 = regions.scale_box(regions.POT_BOX, w, h)
    cv2.rectangle(out, (x1, y1), (x2, y2), ACCENT, 1)

    return out


def _put_line(img, y, text, color=FG, scale=0.55, thickness=1):
    cv2.putText(img, text, (14, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
    return y + int(24 * scale / 0.55)


def build_sidebar(height, state, last_parse_ts, parsing):
    import numpy as np
    panel = np.full((height, SIDEBAR_WIDTH, 3), BG, dtype="uint8")
    y = 30
    y = _put_line(panel, y, "POKER VISION - LIVE", ACCENT, 0.7, 2)
    y += 10

    if parsing:
        y = _put_line(panel, y, "analyzing...", (0, 200, 255))
    elif last_parse_ts:
        age = time.time() - last_parse_ts
        y = _put_line(panel, y, f"last update: {age:0.1f}s ago", DIM)
    y += 10

    if not state:
        _put_line(panel, y, "waiting for first parse...", DIM)
        return panel

    y = _put_line(panel, y, f"Pot: {state.get('pot')}", GOOD, 0.6)
    board = " ".join(state.get("board_cards") or []) or "-"
    y = _put_line(panel, y, f"Board: {board}", GOOD, 0.6)
    y += 14
    y = _put_line(panel, y, f"{'SEAT':<17}{'NAME':<12}{'STACK':<7}{'POS':<4}", DIM, 0.45)

    for seat in state.get("seats", []):
        name = (seat.get("name") or "-")[:11]
        stack = seat.get("stack")
        stack_s = f"{stack:g}" if stack is not None else "-"
        pos = seat.get("position") or ""
        color = FG if seat.get("in_hand") else DIM
        line = f"{seat['seat_id']:<17}{name:<12}{stack_s:<7}{pos:<4}"
        y = _put_line(panel, y, line, color, 0.45)
        if seat.get("status"):
            y = _put_line(panel, y, f"    -> {seat['status']}", (100, 100, 255), 0.45)
        if seat.get("hole_cards"):
            y = _put_line(panel, y, f"    cards: {' '.join(seat['hole_cards'])}", ACCENT, 0.45)

    return panel


def compose(frame, state, last_parse_ts, parsing, display_height=900):
    h, w = frame.shape[:2]
    scale = display_height / h
    disp = cv2.resize(frame, (int(w * scale), display_height))
    annotated = draw_regions(disp, state)
    sidebar = build_sidebar(display_height, state, last_parse_ts, parsing)
    return cv2.hconcat([annotated, sidebar])
