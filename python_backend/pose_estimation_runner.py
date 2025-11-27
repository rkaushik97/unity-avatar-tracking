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

# ---------------- Flask Setup ----------------
app = Flask(__name__)
CORS(app)

current_frame = None
frame_lock = Lock()
running = True

# to store pose data
latest_pose_data = []
pose_data_lock = Lock()

# Tracking 
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
                # match found → reset missed counter
                assigned.add(best_idx)
                self.tracked_people[person_id] = (*best_center, 0)
                updated_ids.append(person_id)
            else:
                # no match → increment missed frames
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
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=False,
    num_poses=2,
)
pose_detector = vision.PoseLandmarker.create_from_options(options)

# ---------------- Drawing Helper ----------------
def draw_landmarks_on_image(bgr_image, detection_result, person_ids):
    annotated_image = np.copy(bgr_image)
    for idx, pose_landmarks in enumerate(detection_result.pose_landmarks):
        proto = landmark_pb2.NormalizedLandmarkList()
        proto.landmark.extend([landmark_pb2.NormalizedLandmark(x=lmk.x, y=lmk.y, z=lmk.z)
                               for lmk in pose_landmarks])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style()
        )

        # Draw ID above the head
        nose = pose_landmarks[0]
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

# ---------------- Flask Routes ----------------
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

# ---------------- Display Loop ----------------
def display_frames():
    global running, pose_tracker
    cv2.namedWindow('Unity Pose Stream', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Unity Pose Stream', 1280, 720)

    last_time = time.time()
    frame_count = 0
    fps = 0

    while running:
        with frame_lock:
            frame_to_process = None if current_frame is None else current_frame

        if frame_to_process is None:
            # Display waiting screen
            display_frame = np.zeros((360, 640, 3), dtype=np.uint8)
            cv2.putText(display_frame, 'Waiting for Unity stream...', (300, 360),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            try:
                # Convert BGR → RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame_to_process,
                                         cv2.COLOR_BGR2RGB)
                #print(rgb_frame.shape)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                
                # Detect pose
                result = pose_detector.detect(mp_image)
                person_ids = []

                if result.pose_landmarks:
                    person_ids = pose_tracker.update(result.pose_landmarks)

                    # Prepare serializable data
                    pose_list = []
                    for idx, lm_list in enumerate(result.pose_landmarks):
                        person_dict = {
                            "id": person_ids[idx],
                            "landmarks": [{"x": lmk.x, "y": lmk.y, "z": lmk.z} for lmk in lm_list]
                        }
                        pose_list.append(person_dict)
                    
                    with pose_data_lock:
                        latest_pose_data.clear()
                        latest_pose_data.extend(pose_list)
                    
                    # Check and print to console
                    print("Pose data: ", pose_list)

                # Draw landmarks on the ORIGINAL frame (not the converted one)
                if result.pose_landmarks:
                    display_frame = draw_landmarks_on_image(frame_to_process, result, person_ids)
                else:
                    display_frame = frame_to_process  # No poses detected, use original

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