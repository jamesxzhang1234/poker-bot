"""Interactive helper: take a full-screen screenshot, drag a box around the
phone-mirror window, and print the region string to pass to live_app.py.

Usage: python tools/select_region.py
"""
import mss
import numpy as np
import cv2


def main():
    with mss.mss() as sct:
        monitor = sct.monitors[1]  # primary display
        shot = sct.grab(monitor)
    img = cv2.cvtColor(np.array(shot), cv2.COLOR_BGRA2BGR)

    print("Drag a box around the phone-mirror window, then press ENTER or SPACE.")
    print("Press 'c' to cancel.")
    r = cv2.selectROI("Select phone screen region - ENTER to confirm", img, showCrosshair=True)
    cv2.destroyAllWindows()

    x, y, w, h = r
    if w == 0 or h == 0:
        print("No region selected.")
        return
    print(f"\nRegion selected: x={x}, y={y}, w={w}, h={h}")
    print(f"\nUse it with:\n  python -m poker_vision.live_app --source region:{x},{y},{w},{h}")


if __name__ == "__main__":
    main()
