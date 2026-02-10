from ultralytics import YOLO
import cv2
import numpy
import time
import copy

MODEL_DIR = "YOLOmodels"
CONFIDENCE_THRESH = 0.8

ALIVE_COL = (0, 255, 255)
DEAD_COL = (200, 0, 200)


model = YOLO(f"{MODEL_DIR}/yolov8n-balls-1-5_ncnn_model")

cap = cv2.VideoCapture(0)

current_time = time.time()
last_time = 0
delta_time = 0

alive = 2
dead = 1

while True:
    ret, frame = cap.read()
    if (not ret):
        print("VideoCapture returned false")
        break
    
    cv2.flip(frame, 0)
    
    im_height, im_width, _ = frame.shape
    
    center = (im_width//2, im_height//2)
    
    results = model(frame, verbose=False)[0]
    
    move_vec = (0, 0)
    
    
    for box in results.boxes:
        cls = int(box.cls[0])
        name = results.names[cls]
        conf = box.conf[0]
        
        if (conf < CONFIDENCE_THRESH):
            continue
        
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        
        ncx = cx / im_width * 2 - 1
        ncy = cy / im_height * -2 + 1
        
        color = (0, 0, 0)
            
        if (name == "alive"):
            color = ALIVE_COL
            
            if (alive > 0):
                move_vec = (ncx, ncy)
                
                cv2.line(frame, center, (cx, cy), (0, 0, 255), 2)
            
        elif (name == "dead"):
            color = DEAD_COL
            
            if (alive <= 0):
                move_vec = (ncx, ncy)
                
                cv2.line(frame, center, (cx, cy), (0, 0, 255), 2)
        
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)
                
    # fps
    last_time = copy.deepcopy(current_time)
    current_time = time.time()
    delta_time = current_time - last_time
    
    fps = int(1 / delta_time)
    
    cv2.putText(frame, f"{fps}", (8, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # alive dead
    cv2.putText(frame, f"alive: {alive}", (im_width - 64, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, f"dead: {dead}", (im_width - 64, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    cv2.imshow("Frame", frame)
    
    
    if (cv2.waitKey(1) & 0xFF == ord('a')):
        alive -= 1
    if (cv2.waitKey(1) & 0xFF == ord('d')):
        dead -= 1
    if (cv2.waitKey(1) & 0xFF == ord('q') or 
        cv2.getWindowProperty("Frame", cv2.WND_PROP_VISIBLE) < 1):
        break

cap.release()
cv2.destroyAllWindows()