[1]
```python
!pip install ultralytics
```
**Explanation**: This installs the Ultralytics YOLO library, used for object detection.

---

[2]
```python
!pip install roboflow
from roboflow import Roboflow
rf = Roboflow(api_key="EHnyu3HZPFbhaV3l3ztl")
project = rf.workspace("ankurts").project("ball_detection-tkmlj")
version = project.version(2)
dataset = version.download("yolov11")
```
**Explanation**: Installs Roboflow, authenticates using the API key, and downloads the YOLOv11 compatible dataset.

---

[3]
```python
from google.colab import drive
drive.mount('/content/drive')
```
**Explanation**: Mounts Google Drive to access or store project files.

---

[4]
```python
!mv /content/Ball_Detection-2 /content/drive/MyDrive/IML_Project
```
**Explanation**: Moves the dataset to your project directory in Google Drive.

---

[5]
```python
from ultralytics import YOLO
model = YOLO("/content/drive/MyDrive/IML_Project/Ball_Detection-2/yolo11s.pt")
results = model.train(data="/content/drive/MyDrive/IML_Project/Ball_Detection-2/data.yaml", epochs=15, imgsz=640)
```
**Explanation**: Loads YOLOv11 model and starts training using dataset config for 15 epochs.

---

[6]
```python
!mv /content/runs /content/drive/MyDrive/IML_Project
```
**Explanation**: Saves training run outputs to your Google Drive.

---

[7]
```python
from ultralytics import YOLO
import cv2
model = YOLO("/content/drive/MyDrive/IML_Project/runs/detect/train/weights/best.pt")
video_path = "/content/drive/MyDrive/IML_Project/videos/videoplayback.mp4"
cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))
output_path = "/content/drive/MyDrive/IML_Project/videos/result_fullvideo_videoplayback.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            if cls==1:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"Class player: {conf:.2f}"
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (3, 219, 252), 2)
                label = f"Class ball: {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (3, 219, 252), 2)
    out.write(frame)
cap.release()
out.release()
print(f"Processed video saved at: {output_path}")
```
**Explanation**: Runs inference on the video, draws boxes, saves the output.

---

[8]
```python
metrics = model.val(data="/content/drive/MyDrive/IML_Project/Ball_Detection-2/data.yaml", split="test")
```
**Explanation**: Evaluates the trained model on the test set.

---

[9]
```python
!mv /content/runs/detect/val /content/drive/MyDrive/IML_Project/runs/detect
```
**Explanation**: Moves validation results to project directory.

---

[10]
```python
print(metrics)
```
**Explanation**: Displays evaluation metrics.

---

[11]
```python
(metrics.results_dict)
```
**Explanation**: Prints metrics as dictionary.

---

[12]
```python
!pip install --upgrade ultralytics
```
**Explanation**: Updates the ultralytics library.

---

[13]
```python
!git clone https://github.com/ultralytics/ultralytics.git
%cd ultralytics
!pip install -e .
```
**Explanation**: Clones ultralytics repo and installs in editable mode.

---

[14]
```python
!wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12m.pt
```
**Explanation**: Downloads pretrained YOLOv12m model.

---

[15]
```python
from google.colab import drive
drive.mount('/content/drive')
```
**Explanation**: Remounting drive for accessing files.

---

[16]
```python
from ultralytics import YOLO
model = YOLO("/content/drive/MyDrive/IML_Project/Ball_Detection-2/yolo12m.pt")
results = model.train(data="/content/drive/MyDrive/IML_Project/Ball_Detection-2/data.yaml", epochs=15, imgsz=640)
```
**Explanation**: Trains YOLOv12 model.

---

[17]
```python
from google.colab import drive
drive.mount('/content/drive')
```
**Explanation**: Remount drive again.

---

[18]
```python
!mv /content/ultralytics/runs/detect/train /content/drive/MyDrive/IML_Project/Ball_Detection-2/YOLOv12m
```
**Explanation**: Moves YOLOv12 training results to project directory.

---

[19]
```python
model_11 = YOLO("/content/drive/MyDrive/IML_Project/runs/detect/train/weights/best.pt")
metrics_v11 = model_11.val(data="/content/drive/MyDrive/IML_Project/Ball_Detection-2/data.yaml", split="test")
```
**Explanation**: Evaluates YOLOv11.

---

[20]
```python
print("Baseline mAP@0.5:", metrics_v11.box.map50)
```
**Explanation**: Prints YOLOv11 mAP@0.5 → 0.7405947229090659

---

[21]
```python
metrics = model.val(data="/content/drive/MyDrive/IML_Project/Ball_Detection-2/data.yaml", split="test")
```
**Explanation**: Evaluates YOLOv12.

---

[22]
```python
print("Baseline mAP@0.5:", metrics.box.map50)
```
**Explanation**: Prints YOLOv12 mAP@0.5 → 0.7803895895873909

---

[23]
```python
from ultralytics import YOLO

model = YOLO("yolo12m.pt")  # or your own model path
results = model.train(data="/content/drive/MyDrive/IML_Project/Ball_Detection-2/data.yaml", epochs=20, imgsz=640)
```

**Explanation:**  
This imports the `YOLO` class from Ultralytics and starts training the YOLOv12 model (`yolo12m.pt`) using the custom dataset defined in `data.yaml`.  
- `epochs=20` means it'll train for 20 rounds.
- `imgsz=640` sets the input image size to 640x640.

---

[24]
```bash
!pip install mediapipe opencv-python
```

**Explanation:**  
This installs the required libraries:
- `mediapipe` for pose detection (used later to detect hand direction).
- `opencv-python` for image and video processing.

---

[25]
```bash
!pip install ultralytics
!pip install mediapipe opencv-python ultralytics
```

**Explanation:**  
First line installs Ultralytics (YOLO library).  
Second line installs all three again just to ensure no dependency is missed. You can consider this a redundancy step to avoid errors.

---

[26]
```bash
!pip uninstall -y ultralytics
```

**Explanation:**  
This **uninstalls** the ultralytics package without asking for confirmation (`-y`). Sometimes we do this to fix version issues or clear a broken installation.

---

[27]
```bash
!pip install ultralytics
```

**Explanation:**  
Reinstalls `ultralytics` fresh after uninstalling it in [26]. This ensures the latest working version is being used, avoiding version mismatch bugs.

---

[28]
```python
import ultralytics
print(ultralytics.__version__)
```

**Explanation:**  
This simply prints the currently installed version of the Ultralytics package. Useful to debug or confirm what version you’re working with.

---

[29]
```python
from google.colab import drive
drive.mount('/content/drive')
```

**Explanation:**  
Mounts your Google Drive into Colab so that your model, videos, and dataset paths (like `/content/drive/MyDrive/IML_Project/...`) can be accessed.

---

[30]
```python
import cv2
import math
import mediapipe as mp
from ultralytics import YOLO

# Load trained model
model = YOLO("/content/drive/MyDrive/IML_Project/Ball_Detection-2/YOLOv12m/weights/best.pt")

# Load video
cap = cv2.VideoCapture("/content/drive/MyDrive/IML_Project/videos/videoplayback-VEED.mp4")
frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Output writer
out = cv2.VideoWriter("/content/drive/MyDrive/IML_Project/videos/result_tactical_analysis_final.mp4",
                      cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# Initialize mediapipe pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)

# Stats and memory
player_stats = {
    "top": {"Forehand": 0, "Backhand": 0, "color": (0, 255, 255)},
    "bottom": {"Forehand": 0, "Backhand": 0, "color": (255, 0, 255)}
}
player_memory = {"top": False, "bottom": False}
DISTANCE_THRESHOLD = 150

def classify_shot_direction(landmarks):
    try:
        r_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        r_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        l_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        l_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]

        right_diff = r_wrist.x - r_shoulder.x
        left_diff = l_wrist.x - l_shoulder.x

        if abs(right_diff) > abs(left_diff):
            return "Forehand" if right_diff > 0 else "Backhand"
        else:
            return "Backhand" if left_diff > 0 else "Forehand"
    except:
        return None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    player_boxes = []
    ball_boxes = []

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            if cls == 0:  # ball
                ball_boxes.append((x1, y1, x2, y2, cx, cy))
            elif cls == 1:  # player
                player_boxes.append((x1, y1, x2, y2, cx, cy))

    for px1, py1, px2, py2, pcx, pcy in player_boxes:
        player_pos = "top" if pcy < frame_height // 2 else "bottom"
        color = player_stats[player_pos]["color"]

        # Find closest ball
        min_dist = float('inf')
        closest_ball = None
        for bx1, by1, bx2, by2, bcx, bcy in ball_boxes:
            dist = math.hypot(pcx - bcx, pcy - bcy)
            if dist < min_dist:
                min_dist = dist
                closest_ball = (bcx, bcy)

        # Extract player ROI
        player_crop = frame[py1:py2, px1:px2]
        if player_crop.size == 0:
            continue

        # Run mediapipe pose
        rgb_crop = cv2.cvtColor(player_crop, cv2.COLOR_BGR2RGB)
        mp_results = pose.process(rgb_crop)

        # Determine arm direction if ball is near
        if mp_results.pose_landmarks and closest_ball and min_dist < DISTANCE_THRESHOLD:
            shot_type = classify_shot_direction(mp_results.pose_landmarks.landmark)

            if shot_type and not player_memory[player_pos]:
                player_stats[player_pos][shot_type] += 1
                player_memory[player_pos] = True

                cv2.putText(frame, f"{shot_type}", (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
            player_memory[player_pos] = False

        # Draw player bounding box
        cv2.rectangle(frame, (px1, py1), (px2, py2), color, 2)

    # Draw ball boxes
    for bx1, by1, bx2, by2, _, _ in ball_boxes:
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
        cv2.putText(frame, "Ball", (bx1, by1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Draw stats
    x0 = frame_width - 240
    y = 30
    for pos in ["top", "bottom"]:
        stats = player_stats[pos]
        cv2.putText(frame, f"{pos.capitalize()} Player", (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, stats["color"], 2)
        y += 25
        cv2.putText(frame, f"Forehand : {stats['Forehand']}", (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, stats["color"], 1)
        y += 20
        cv2.putText(frame, f"Backhand : {stats['Backhand']}", (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, stats["color"], 1)
        y += 30

    out.write(frame)

cap.release()
out.release()
pose.close()

print("Analysis complete and video saved.")
print("Final stats:", player_stats)
```

**Explanation:**  
This is the **main tactical analysis pipeline**:
- Loads the **trained YOLOv12 model**.
- Opens and reads the input match video.
- Detects **players and balls** frame-by-frame.
- Uses **MediaPipe** to detect **poses**.
- Analyzes **arm direction** to classify shots as **Forehand** or **Backhand**.
- Draws bounding boxes, overlays stats, and writes a final analyzed video.
- At the end, prints out total counts of shots per player (top/bottom).
