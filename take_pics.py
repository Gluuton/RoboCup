import cv2
import glob

outputDir = "YOLOpics"
showImg = True
saveImg = False

cap = cv2.VideoCapture(0)

existing_frames = glob.glob(f"{outputDir}/*.png")

frameNum = len(existing_frames)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    cv2.flip(frame,     0)
    
    if (showImg):
        cv2.imshow("Frame", frame)
    if (saveImg):
        cv2.imwrite(f"{outputDir}/frame{frameNum}.png", frame)
        frameNum += 1
    
    if (cv2.waitKey(1) &0xFF == ord(' ')):
        cv2.imwrite(f"{outputDir}/frame{frameNum}.png", frame)
        frameNum += 1
    
    if (cv2.waitKey(1) & 0xFF == ord('q') or
        cv2.getWindowProperty("Frame", cv2.WND_PROP_VISIBLE) < 1):
        break

cap.release()
cv2.destroyAllWindows()