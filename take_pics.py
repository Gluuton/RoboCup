import cv2
import glob

while True:
    print("pp")

    if (cv2.waitKey(1) & 0xFF == ord('q') or
        cv2.getWindowProperty("Frame", cv2.WND_PROP_VISIBLE) < 1):
        break

cv2.destroyAllWindows()