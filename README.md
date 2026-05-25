# ⚽ Football Goal Analysis — YOLOv8 Player & Referee Tracker

A computer vision pipeline that detects and tracks **players**, **referees**, and the **ball** in football match footage using a custom-trained YOLOv8 model and ByteTrack multi-object tracking.

---

## 📽️ Demo

### Before Annotation (Raw Input)

> 📌 **Replace the placeholder below with your raw input video**

<!-- To embed a video in GitHub Markdown, upload it via the GitHub UI (drag & drop into an Issue or PR), copy the generated URL, and paste it here -->

**Input Video Preview:**

https://github.com/user-attachments/assets/REPLACE-WITH-YOUR-INPUT-VIDEO-URL

---

### After Annotation (Model Output)

> 📌 **Replace the placeholder below with your annotated output video**

**Output Video Preview:**

https://github.com/user-attachments/assets/REPLACE-WITH-YOUR-OUTPUT-VIDEO-URL

> 💡 **How to attach videos in GitHub:** Go to any GitHub Issue or Pull Request → drag & drop your `.mp4` or `.avi` file → GitHub generates a URL → paste it above.
> Alternatively, convert your `.avi` to `.mp4` using `ffmpeg -i output.avi output.mp4` for better browser compatibility.

---

## 📁 Project Structure

```
Football Goal Analysis/
│
├── main.py                    # Entry point — runs the full pipeline
├── yolov8m.pt                 # YOLOv8 medium pretrained weights
├── yolov8s.pt                 # YOLOv8 small pretrained weights
│
├── input_videos/
│   └── 08fd33_4.mp4           # Source football match video
│
├── models/
│   ├── best.pt                # Custom-trained model (best checkpoint)
│   └── last.pt                # Custom-trained model (last checkpoint)
│
├── output_videos/
│   └── output.avi             # Final annotated video output
│
├── stubs/
│   └── tracks.pkl             # Cached tracking data (skip re-inference)
│
├── tracker/
│   ├── tracker.py             # Core Tracker class (detect + track + annotate)
│   └── __init__.py
│
├── utils/
│   ├── video_utils.py         # read_video / save_video helpers
│   ├── bbox_utils.py          # Bounding box utility functions
│   └── __init__.py
│
└── Training/
    ├── football_training_yolo_v5.ipynb   # Training notebook
    └── football-players-detection-1/     # Roboflow dataset (train/valid/test)
```

---

## 🧠 How It Works

### 1. Video Reading
The input match video is read frame-by-frame using OpenCV via `read_video()` in `utils/video_utils.py`.

### 2. Object Detection (YOLOv8)
A custom YOLOv8 model (`models/best.pt`) — fine-tuned on a football players dataset from Roboflow — runs inference on batches of 20 frames at a time with a confidence threshold of `0.2`.

The model detects three classes:
- `player`
- `goalkeeper` (remapped → `player`)
- `referee`
- `ball`

### 3. Multi-Object Tracking (ByteTrack)
Detections are passed to **ByteTrack** (via the `supervision` library) which assigns persistent track IDs to each detected object across frames, even through occlusions.

### 4. Annotation & Visualization
Each frame is annotated with:
- **Ellipses** drawn at the feet of players and referees (mimicking a shadow/ground effect)
- **Track ID labels** in a small rectangle beneath each player
- **Color coding:** Players → Red `(0,0,255)` · Referees → Yellow `(0,255,255)`

### 5. Stub Caching
To avoid re-running inference every time, tracking results are serialized to `stubs/tracks.pkl`. Set `read_from_stub=True` in `main.py` to load cached results instantly.

### 6. Video Export
Annotated frames are saved back as a video using `save_video()`.

---

## 🔄 Pipeline Flow

```
Input Video
    │
    ▼
Read Frames (OpenCV)
    │
    ▼
YOLOv8 Detection (batch=20, conf=0.2)
    │
    ▼
Goalkeeper → Player remapping
    │
    ▼
ByteTrack Multi-Object Tracking
    │
    ▼
Draw Ellipses + Track ID Labels
    │
    ▼
Output Annotated Video (.avi)
```

---

## 🗂️ Dataset

The model was trained on the **Football Players Detection** dataset sourced from [Roboflow](https://roboflow.com), organized as:

| Split | Images |
|-------|--------|
| Train | ~280   |
| Valid | ~39    |
| Test  | ~13    |

Training was done using a YOLOv5/v8 notebook (`Training/football_training_yolo_v5.ipynb`).

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.11+
- CUDA-compatible GPU (recommended)

### Install Dependencies

```bash
pip install ultralytics supervision opencv-python
```

### Run the Pipeline

```bash
python main.py
```

To re-run inference (ignoring cached stubs), set `read_from_stub=False` in `main.py`:

```python
tracks = tracker.get_object_tracks(video_frames, read_from_stub=False)
```

---

## 📌 Key Files Explained

### `main.py`
The entry point. Reads the video, initializes the tracker with the custom model, retrieves object tracks (from stub or fresh inference), draws annotations, and saves the output.

### `tracker/tracker.py`
The core `Tracker` class containing:
- `detect_frames()` — batched YOLOv8 inference
- `get_object_tracks()` — converts detections to tracked objects with persistent IDs
- `draw_ellipse()` — draws the ellipse + ID badge for each tracked entity
- `draw_annotations()` — applies annotations across all frames

### `utils/bbox_utils.py`
Helper functions: `get_centre_of_bbox()` and `get_width_bbox()` used to compute ellipse placement from bounding box coordinates.

---

## 🚀 Future Improvements

- Ball tracking (currently detected but not annotated in output)
- Team assignment via jersey color clustering (K-Means)
- Speed & distance estimation per player
- Heatmap generation for player positioning
- Camera movement compensation for accurate field mapping

---

## 🙏 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Roboflow Football Players Detection Dataset](https://roboflow.com)
- [Supervision Library by Roboflow](https://github.com/roboflow/supervision)
- [ByteTrack](https://github.com/ifzhang/ByteTrack)
