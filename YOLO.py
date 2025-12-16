from ultralytics import YOLO
import cv2

modelDir = "YOLOmodels"

model = YOLO(f"{modelDir}/yolov8n-face-lindevs.pt") # våran egna tränade modell


cap = cv2.VideoCapture(0)
# frame = getFrame() isch

# frame = cv2.imread("test.jpg")
# height, width, channels = frame.shape

# print(f"{height}, {width}")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    cv2.flip(frame, 0)
    
    height, width, channels = frame.shape

    results = model(frame, verbose=False)[0]
    
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        
        ncx = cx / width * 2 - 1
        ncy = cy / height * -2 + 1
        

        cls = int(box.cls[0])
        name = results.names[cls]
        conf = float(box.conf[0])
        
        # Draw bounding box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)

        # Draw label and confidence
        label = f"{name} {conf:.2f}"
        cv2.putText(frame, label, (int(x1), int(y1)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        coords = f"x: {ncx}, y: {ncy}"
        cv2.putText(frame, coords, (int(x1), int(y1) - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Optional: draw center point
        cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

    cv2.imshow("YOLO", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()