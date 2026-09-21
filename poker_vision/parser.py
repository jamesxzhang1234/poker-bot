"""Turns a raw table screenshot/frame into a TableState."""
from . import card_vision, ocr_utils, regions
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


class TableParser:
    def __init__(self):
        self.seat_defs = regions.SEATS

    def parse_seat(self, frame, seat_id):
        cfg = self.seat_defs[seat_id]
        name = ocr_utils.ocr_name(_crop(frame, cfg["name_row"])) or None
        stack = ocr_utils.ocr_number(_crop(frame, cfg["stack_row"]))
        position = ocr_utils.ocr_badge(_crop(frame, cfg["badge"]))

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
            hero_cards = [card_vision.parse_card(_crop(frame, box)) for box in regions.HERO_CARDS]
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
        return ocr_utils.ocr_number(_crop(frame, regions.POT_BOX))

    def parse(self, frame, frame_index=0, timestamp_sec=0.0):
        seats = [self.parse_seat(frame, seat_id) for seat_id in self.seat_defs]
        board = self.parse_board(frame)
        pot = self.parse_pot(frame)
        return TableState(
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
            seats=seats,
            board_cards=board,
            pot=pot,
        )
