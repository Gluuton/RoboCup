import ncnn
import cv2
import numpy as np
import time
import math
from ultralytics import YOLO

# ----------------------------
# Config
# ----------------------------
PARAM_PATH = "YOLOmodels/yolov8n-balls-1-5-ncnn.param"
BIN_PATH   = "YOLOmodels/yolov8n-balls-1-5-ncnn.bin"

IMG_SIZE = 512
CONFIDENCE_THRESH = 0.6
NMS_THRESH = 0.45
NUM_THREADS = 4

pt_model = YOLO("YOLOmodels/yolov8n-balls-1-5.pt")
CLASSES = pt_model.names

# ----------------------------
# Load NCNN model
# ----------------------------
net = ncnn.Net()
net.opt.num_threads = NUM_THREADS
net.opt.use_fp16_packed = True
net.opt.use_fp16_storage = True
net.opt.use_fp16_arithmetic = True

net.load_param(PARAM_PATH)
net.load_model(BIN_PATH)

# ----------------------------
# Camera
# ----------------------------
cap = cv2.VideoCapture(0)

prev_time = time.time()

# ----------------------------
# Utils
# ----------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def nms(boxes, scores, thresh):
    indices = cv2.dnn.NMSBoxes(boxes, scores, CONFIDENCE_THRESH, thresh)
    return indices.flatten() if len(indices) else []


# ----------------------------
# Main loop
# ----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h0, w0 = frame.shape[:2]

    # ----------------------------
    # Letterbox
    # ----------------------------
    scale = min(IMG_SIZE / w0, IMG_SIZE / h0)
    nw, nh = int(w0 * scale), int(h0 * scale)

    resized = cv2.resize(frame, (nw, nh))
    padded = np.full((IMG_SIZE, IMG_SIZE, 3), 114, dtype=np.uint8)
    padded[:nh, :nw] = resized

    padded = padded[:, :, ::-1]  # BGR → RGB
    padded = padded.astype(np.float32) / 255.0

    mat = ncnn.Mat.from_pixels(
        padded,
        ncnn.Mat.PixelType.PIXEL_RGB,
        IMG_SIZE,
        IMG_SIZE
    )

    # ----------------------------
    # Inference
    # ----------------------------
    with net.create_extractor() as ex:
        ex.input("in0", mat)
        _, out = ex.extract("out0")

    preds = np.array(out)  # (N, 4 + num_classes)

    boxes = []
    scores = []
    class_ids = []

    for p in preds:
        obj_conf = p[4]
        if obj_conf < CONFIDENCE_THRESH:
            continue

        class_scores = p[5:]
        cls = int(np.argmax(class_scores))
        conf = obj_conf * class_scores[cls]

        if conf < CONFIDENCE_THRESH or cls >= len(CLASSES):
            continue

        cx, cy, w, h = p[:4]

        x = (cx - w / 2) * IMG_SIZE
        y = (cy - h / 2) * IMG_SIZE
        w *= IMG_SIZE
        h *= IMG_SIZE

        x /= scale
        y /= scale
        w /= scale
        h /= scale

        boxes.append([int(x), int(y), int(w), int(h)])
        scores.append(float(conf))
        class_ids.append(cls)


    # ----------------------------
    # NMS
    # ----------------------------
    indices = nms(boxes, scores, NMS_THRESH)

    for i in indices:
        x, y, w, h = boxes[i]
        cls = class_ids[i]
        conf = scores[i]

        cx = int(x + w / 2)
        cy = int(y + h / 2)

        ncx = cx / w0 * 2 - 1
        ncy = cy / h0 * -2 + 1

        color = (255, 255, 0) if cls == 0 else (255, 0, 0)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)

        label = f"{CLASSES[cls]} {conf:.2f}"
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        coords = f"x: {ncx:.3f}, y: {ncy:.3f}"
        cv2.putText(frame, coords, (x, y - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

    # ----------------------------
    # FPS
    # ----------------------------
    now = time.time()
    fps = int(1 / (now - prev_time))
    prev_time = now

    cv2.putText(frame, f"{fps}", (8, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("YOLOv8 NCNN", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
