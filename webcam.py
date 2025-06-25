import cv2
import pandas as pd
from ultralytics import YOLO
import torch

# === GPU Detection Check (Optional) ===
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    print("⚠ GPU not available, switching to CPU.")

# === SETTINGS ===
MODEL_PATH = 'runs/best.pt'   # Path to your trained YOLOv8 model
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # Use GPU if available
VIDEO_SRC = 0                    # 0 = Default webcam
CSV_FILE = 'detections.csv'      # File to save detections

# === LOAD YOLOv8 MODEL ===
model = YOLO(MODEL_PATH)

# === INIT VIDEO CAPTURE ===
cap = cv2.VideoCapture(VIDEO_SRC)
df = pd.DataFrame(columns=['frame', 'class', 'conf', 'xmin', 'ymin', 'xmax', 'ymax'])
frame_id = 0

print("🔍 Starting detection. Press ESC to stop...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Webcam not accessible.")
        break

    frame_id += 1
    results = model.predict(frame, device=DEVICE, conf=0.25, iou=0.45, verbose=False)

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            df.loc[len(df)] = [
                frame_id,
                model.names[cls_id],
                round(conf, 2),
                x1, y1, x2, y2
            ]

        # Show detections
        annotated = r.plot()
        cv2.imshow("YOLOv8 PPE Detection", annotated)

    # Exit if ESC key is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        print("🛑 Detection stopped by user.")
        break

cap.release()
cv2.destroyAllWindows()

# === SAVE DETECTIONS TO CSV ===
df.to_csv(CSV_FILE, index=False)
print(f"✅ Detections saved to: {CSV_FILE}")