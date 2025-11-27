import cv2
import numpy as np
import base64
import time
from threading import Thread, Lock
from flask import Flask, request, jsonify
from flask_cors import CORS
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ---------------- Config / Helpers ----------------
MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red

def crop_from_bbox(image, bbox):
    """
    image: numpy array (H,W,3) BGR
    bbox: MediaPipe BoundingBox object (origin_x, origin_y, width, height)
    returns: cropped image (BGR) or None if invalid
    """
    x1 = int(bbox.origin_x)
    y1 = int(bbox.origin_y)
    x2 = int(bbox.origin_x + bbox.width)
    y2 = int(bbox.origin_y + bbox.height)

    H, W = image.shape[:2]
    x1 = max(0, min(W - 1, x1))
    y1 = max(0, min(H - 1, y1))
    x2 = max(0, min(W, x2))
    y2 = max(0, min(H, y2))

    if x2 <= x1 or y2 <= y1:
        return None

    return image[y1:y2, x1:x2]

def draw_landmarks_on_image(bgr_image, pose_landmarks_list, person_ids):
    """
    bgr_image: full-frame BGR image (will be copied)
    pose_landmarks_list: list of lists of landmark_pb2.NormalizedLandmark (x,y,z normalized to full frame)
    person_ids: list of ids aligned with pose_landmarks_list
    """
    annotated_image = np.copy(bgr_image)
    for idx, lm_list in enumerate(pose_landmarks_list):
        # convert to NormalizedLandmarkList proto for drawing
        proto = landmark_pb2.NormalizedLandmarkList()
        proto.landmark.extend([landmark_pb2.NormalizedLandmark(x=lmk.x, y=lmk.y, z=lmk.z)
                               for lmk in lm_list])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style()
        )

        # Draw ID above the head if nose exists
        if len(lm_list) > 0:
            nose = lm_list[0]
            px = int(nose.x * annotated_image.shape[1])
            py = int(nose.y * annotated_image.shape[0]) - 10
            cv2.putText(annotated_image,
                        f"ID {person_ids[idx] if idx < len(person_ids) else -1}",
                        (px, py),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2)
    return annotated_image

# ---------------- Flask Setup ----------------
app = Flask(__name__)
CORS(app)

current_frame = None
frame_lock = Lock()
running = True

# to store pose data
latest_pose_data = []
pose_data_lock = Lock()

# ---------------- Tracking ----------------
class PoseTracker:
    def __init__(self, max_distance=0.05, max_missed=35):
        self.tracked_people = {}   # id → (x, y, missed_frames)
        self.next_id = 0
        self.max_distance = max_distance
        self.max_missed = max_missed

    def update(self, landmarks):
        updated_ids = []
        current_centers = []

        # calculate centers
        for idx, lm_list in enumerate(landmarks):
            if lm_list and len(lm_list) > 0:
                x = lm_list[0].x
                y = lm_list[0].y
                current_centers.append((idx, (x, y)))

        assigned = set()

        # match existing tracks
        for person_id, (tx, ty, missed) in list(self.tracked_people.items()):
            best_dist = float('inf')
            best_idx = None
            best_center = None

            for idx, center in current_centers:
                if idx in assigned:
                    continue
                x, y = center
                dist = (x - tx)**2 + (y - ty)**2
                if dist < best_dist and dist < self.max_distance:
                    best_dist = dist
                    best_idx = idx
                    best_center = center

            if best_idx is not None:
                assigned.add(best_idx)
                self.tracked_people[person_id] = (*best_center, 0)
                updated_ids.append(person_id)
            else:
                missed += 1
                if missed > self.max_missed:
                    del self.tracked_people[person_id]
                else:
                    self.tracked_people[person_id] = (tx, ty, missed)

        # create new tracks
        for idx, center in current_centers:
            if idx not in assigned:
                self.tracked_people[self.next_id] = (*center, 0)
                updated_ids.append(self.next_id)
                self.next_id += 1

        return updated_ids

pose_tracker = PoseTracker()

# ---------------- PoseLandmarker Setup ----------------
base_options = python.BaseOptions(model_asset_path='pose_landmarker_full.task')
pose_options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=False,
    num_poses=2,
)
pose_detector = vision.PoseLandmarker.create_from_options(pose_options)

# ---------------- Object Detector Setup (NEW) ----------------
# Uses efficientdet_lite2.tflite (place model in same folder or update path)
det_base = python.BaseOptions(model_asset_path='efficientdet_lite2.tflite')
det_options = vision.ObjectDetectorOptions(
    base_options=det_base,
    score_threshold=0.6,   # detection threshold
    max_results=8
)
object_detector = vision.ObjectDetector.create_from_options(det_options)

# ---------------- Flask Routes (unchanged) ----------------
@app.route('/stream', methods=['POST'])
def receive_frame():
    global current_frame
    try:
        data = request.json
        if 'frame' not in data:
            return jsonify({'error': 'No frame data'}), 400

        img_data = base64.b64decode(data['frame'])
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # BGR

        if frame is None:
            return jsonify({'error': 'Failed to decode frame'}), 400

        with frame_lock:
            current_frame = frame

        return jsonify({'status': 'success'}), 200
    except Exception as e:
        print(f"Error receiving frame: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'running', 'has_frame': current_frame is not None}), 200

@app.route('/pose', methods=['GET'])
def get_pose():
    with pose_data_lock:
        return jsonify(latest_pose_data), 200

# ---------------- Display Loop (MAIN CHANGE) ----------------
def display_frames():
    global running, pose_tracker
    cv2.namedWindow('Unity Pose Stream', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Unity Pose Stream', 1280, 720)

    last_time = time.time()
    frame_count = 0
    fps = 0

    while running:
        with frame_lock:
            frame_to_process = None if current_frame is None else current_frame.copy()

        if frame_to_process is None:
            display_frame = np.zeros((360, 640, 3), dtype=np.uint8)
            cv2.putText(display_frame, 'Waiting for Unity stream...', (50, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            try:
                H_full, W_full = frame_to_process.shape[:2]

                # Convert BGR -> RGB (MediaPipe expects RGB data)
                rgb_frame = cv2.cvtColor(frame_to_process, cv2.COLOR_BGR2RGB)
                mp_full_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                # 1) Run object detector on full image
                det_result = object_detector.detect(mp_full_image)

                # 2) For each person detection: crop & run pose on crop
                pose_landmarks_global = []  # will store list-of-landmark-lists in full-frame normalized coords
                debug_bboxes = []          # store bboxes for visualization if needed

                if det_result and det_result.detections:
                    for det in det_result.detections:
                        # ensure category exists
                        if not det.categories:
                            continue
                        cat = det.categories[0].category_name
                        score = det.categories[0].score if det.categories[0].score is not None else 0.0
                        if cat != 'person' or score < 0.4:
                            continue

                        bbox = det.bounding_box
                        crop_bgr = crop_from_bbox(frame_to_process, bbox)
                        if crop_bgr is None:
                            continue
                        crop_h, crop_w = crop_bgr.shape[:2]

                        # Convert crop to RGB and create mp.Image
                        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                        mp_crop_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)

                        # Run pose detector on the crop
                        try:
                            crop_pose_res = pose_detector.detect(mp_crop_image)
                        except Exception as e:
                            print("Pose detector on crop failed:", e)
                            crop_pose_res = None

                        if crop_pose_res and crop_pose_res.pose_landmarks:
                            # Note: crop_pose_res.pose_landmarks is a list (one per detected pose on crop)
                            # We'll take each pose (usually one) and transform landmark coords to full-frame normalized coords
                            for pose_landmarks in crop_pose_res.pose_landmarks:
                                transformed = []
                                for lmk in pose_landmarks:
                                    # lmk.x/lmk.y are normalized to the crop (0..1). transform to full-frame normalized coords
                                    global_x = (bbox.origin_x + (lmk.x * bbox.width)) / float(W_full)
                                    global_y = (bbox.origin_y + (lmk.y * bbox.height)) / float(H_full)
                                    global_z = lmk.z  # z is relative; keep as-is (optionally scale)
                                    transformed.append(landmark_pb2.NormalizedLandmark(
                                        x=float(global_x), y=float(global_y), z=float(global_z)
                                    ))
                                pose_landmarks_global.append(transformed)
                                debug_bboxes.append((int(bbox.origin_x), int(bbox.origin_y),
                                                    int(bbox.origin_x + bbox.width), int(bbox.origin_y + bbox.height)))

                # 3) If we found any poses from cropped detections, run tracking and update pose data
                person_ids = []
                if pose_landmarks_global:
                    person_ids = pose_tracker.update(pose_landmarks_global)

                    # Prepare JSON-serializable pose info
                    pose_list = []
                    for idx, lm_list in enumerate(pose_landmarks_global):
                        pid = person_ids[idx] if idx < len(person_ids) else -1
                        person_dict = {
                            "id": pid,
                            "landmarks": [{"x": lmk.x, "y": lmk.y, "z": lmk.z} for lmk in lm_list]
                        }
                        pose_list.append(person_dict)

                    with pose_data_lock:
                        latest_pose_data.clear()
                        latest_pose_data.extend(pose_list)

                    print("Pose data:", pose_list)

                # 4) Visualization - draw detected bboxes and pose landmarks on the original full-frame
                if pose_landmarks_global:
                    display_frame = draw_landmarks_on_image(frame_to_process, pose_landmarks_global, person_ids)
                    # optionally draw boxes
                    for (x1, y1, x2, y2) in debug_bboxes:
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 128, 255), 2)
                else:
                    # Fallback: show original frame (optionally run pose on full frame if you still want)
                    display_frame = frame_to_process

                # FPS calculation
                frame_count += 1
                if time.time() - last_time >= 1.0:
                    fps = frame_count
                    frame_count = 0
                    last_time = time.time()

            except Exception as e:
                print("Pose estimation error:", e)
                display_frame = frame_to_process  # Fallback to original frame on error

        # Add FPS counter to the display frame
        cv2.putText(display_frame, f'FPS: {fps}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the processed frame
        cv2.imshow('Unity Pose Stream', display_frame)
        
        # Check for exit key
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 'q' or ESC
            running = False
            break

    cv2.destroyAllWindows()

# ---------------- Main ----------------
if __name__ == '__main__':
    print("Starting Unity Pose Estimation Server...")
    print("Press 'q' or ESC to exit")
    
    # Start Flask server in background thread
    flask_thread = Thread(target=lambda: app.run(
        host='0.0.0.0', port=5000, debug=False, threaded=True, use_reloader=False
    ), daemon=True)
    flask_thread.start()

    # Display loop must run on the main thread
    display_frames()
    print("Server stopped.")
