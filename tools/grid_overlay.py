"""Draw a pixel grid over a frame to help calibrate region coordinates."""
import cv2
import sys

def main(src, dst, step=100):
    img = cv2.imread(src)
    h, w = img.shape[:2]
    for x in range(0, w, step):
        cv2.line(img, (x, 0), (x, h), (0, 255, 0), 1)
        cv2.putText(img, str(x), (x + 2, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    for y in range(0, h, step):
        cv2.line(img, (0, y), (w, y), (0, 255, 255), 1)
        cv2.putText(img, str(y), (2, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    cv2.imwrite(dst, img)
    print(f"wrote {dst} ({w}x{h})")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 100)
