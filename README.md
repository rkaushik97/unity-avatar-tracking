# Unity Avatar Tracking

Real-time human pose estimation and avatar tracking system. A Python Flask backend uses MediaPipe to detect and track up to 2 people from a Unity camera stream, returning 33-point pose landmarks that drive 3D avatar animations in a simulated pedestrian environment.

> **Presentation slides:** [SWE Seminar – Unity Avatar Tracking](https://docs.google.com/presentation/d/1QopaIY7cjwDFBW2ifQ08Iu7SYa-eLOOu/edit?usp=sharing&ouid=101375655957008376887&rtpof=true&sd=true)

---

## Architecture

```
Unity Client (C#)
  └─ StreamToBackend.cs   →  POST /stream  (base64 JPEG frames @ 30 FPS)
  └─ PoseReceiver.cs      ←  GET  /pose   (JSON pose landmarks @ 30 FPS)

Python Backend (Flask + MediaPipe)
  └─ pose_estimation_runner.py
       ├─ PoseLandmarker  →  33 landmarks per person, up to 2 people
       ├─ PoseTracker     →  consistent IDs across frames
       └─ OpenCV window   →  live annotated preview
```

## Project Structure

```
unity-avatar-tracking/
├── python_backend/
│   ├── pose_estimation_runner.py          # Main Flask server (entrypoint)
│   ├── pose_estimation_with_detection.py  # Alternative with person detector pre-step
│   ├── pipeline_analysis.ipynb            # Pipeline comparison notebook
│   ├── scenarios/                         # Source test images
│   ├── pipeline_1/                        # Annotated results – pipeline 1
│   └── pipeline_2/                        # Annotated results – pipeline 2
│
└── unity-project/                         # Unity 2022.3 LTS project
    └── Assets/
        ├── Scripts/
        │   ├── StreamToBackend.cs          # Captures & streams camera frames
        │   ├── PoseReceiver.cs             # Polls backend for pose JSON
        │   ├── AvatarAnimatorWalker.cs     # NavMesh-based avatar walking
        │   ├── WalkBetweenPoints.cs        # Path-based walking with turn animations
        │   └── CrosswalkMover.cs           # Crosswalk-specific movement controller
        ├── Scenes/
        │   ├── ParkingLot_WithPose.unity   # Main demo scene
        │   ├── Pose Landmark Detection.unity
        │   └── SingleImageTest.unity
        ├── Ready Player Me/                # 3D avatar SDK
        └── ModularLowpolyStreetsFree/      # Street / parking lot environment
```

## Requirements

### Python Backend

- Python 3.9+
- Flask
- flask-cors
- mediapipe
- opencv-python
- numpy

Install dependencies:

```bash
pip install flask flask-cors mediapipe opencv-python numpy
```

Download the MediaPipe pose model and place it in `python_backend/`:

```bash
wget -q https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task \
     -O python_backend/pose_landmarker_full.task
```

### Unity Client

- Unity **2022.3.62f3** (LTS)
- Packages (auto-installed via `manifest.json`):
  - AI Navigation (NavMesh)
  - Ready Player Me Core SDK
  - Newtonsoft JSON 3.2.1
  - TextMeshPro 3.0.7

## Getting Started

### 1. Start the Python backend

```bash
cd python_backend
python pose_estimation_runner.py
```

The server starts on `http://localhost:5000`. An OpenCV window shows the live annotated stream. Press `q` or `ESC` to quit.

### 2. Open the Unity project

Open `unity-project/` in Unity Hub (Unity 2022.3 LTS). Open the **ParkingLot_WithPose** scene.

### 3. Run the scene

Press **Play** in Unity, then press **SPACE** to start streaming camera frames to the backend. The status overlay (top-left) shows streaming state and FPS.

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/stream` | POST | Receive a base64-encoded JPEG frame from Unity |
| `/pose`   | GET  | Return the latest pose data as JSON |
| `/health` | GET  | Server health check |

### Pose response format

```json
[
  {
    "id": 0,
    "landmarks": [
      { "x": 0.51, "y": 0.23, "z": -0.04 },
      ...
    ]
  }
]
```

Each person has 33 landmarks (MediaPipe convention, normalized 0–1). Landmark 0 is the nose / head position used for tracking.

## Key Configuration

| Parameter | Location | Default | Description |
|-----------|----------|---------|-------------|
| `num_poses` | `pose_estimation_runner.py:92` | `2` | Max people tracked simultaneously |
| `max_distance` | `PoseTracker.__init__` | `0.05` | Max normalized distance for ID re-association |
| `max_missed` | `PoseTracker.__init__` | `35` | Frames before a lost track is dropped |
| `backendURL` | `StreamToBackend.cs` | `http://localhost:5000/stream` | Flask server address |
| `targetFPS` | `StreamToBackend.cs` | `30` | Streaming frame rate |
| `imageWidth/Height` | `StreamToBackend.cs` | `1280×720` | Capture resolution |

## Pipeline Analysis

`python_backend/pipeline_analysis.ipynb` compares two pose estimation approaches on the six test images in `scenarios/`. Annotated output images are saved to `pipeline_1/` and `pipeline_2/`.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Pose estimation | MediaPipe PoseLandmarker (full model) |
| Backend server | Python · Flask · OpenCV |
| Multi-person tracking | Custom centroid-based PoseTracker |
| 3D environment | Unity 2022.3 LTS |
| Avatars | Ready Player Me SDK |
| Pathfinding | Unity AI Navigation (NavMesh) |
| Frame transport | HTTP · base64 JPEG |
