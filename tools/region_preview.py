"""Draw all calibrated regions on a frame so they can be visually verified.

Usage: python tools/region_preview.py <frame.png> <out.png>
"""
import os
import sys

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from poker_vision import regions  # noqa: E402


def main(src, dst):
    img = cv2.imread(src)
    h, w = img.shape[:2]

    def box(b, color, label=None):
        if b is None:
            return
        x1, y1, x2, y2 = regions.scale_box(b, w, h)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        if label:
            cv2.putText(img, label, (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 1, cv2.LINE_AA)

    for seat_id, cfg in regions.SEATS.items():
        box(cfg["avatar"], (255, 255, 0))
        box(cfg.get("cards"), (0, 255, 255), "cards")
        box(cfg["name_row"], (0, 255, 0), "name")
        box(cfg["stack_row"], (0, 200, 0), "stack")
        box(cfg["badge"], (255, 0, 255), "badge")
        box(cfg["status"], (0, 128, 255), "status")
        cv2.putText(img, seat_id, (regions.scale_box(cfg["avatar"], w, h)[0], regions.scale_box(cfg["avatar"], w, h)[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    for slot in regions.BOARD_SLOTS:
        box(slot, (0, 0, 255), "board")

    for c in regions.HERO_CARDS:
        box(c, (255, 128, 0), "hero_card")

    box(regions.POT_BOX, (255, 255, 255), "pot")

    cv2.imwrite(dst, img)
    print(f"wrote {dst}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
