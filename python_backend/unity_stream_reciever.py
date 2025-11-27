import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
from threading import Thread, Lock
import time

app = Flask(__name__)
CORS(app)

current_frame = None
frame_lock = Lock()
running = True

@app.route('/stream', methods=['POST'])
def receive_frame():
    global current_frame
    try:
        data = request.json
        if 'frame' not in data:
            return jsonify({'error': 'No frame data'}), 400

        img_data = base64.b64decode(data['frame'])
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({'error': 'Failed to decode frame'}), 400

        with frame_lock:
            current_frame = frame

        return jsonify({'status': 'success', 'timestamp': time.time()}), 200

    except Exception as e:
        print(f"Error receiving frame: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'running', 'has_frame': current_frame is not None}), 200


def run_flask():
    print("Starting Unity Stream Receiver on http://localhost:5000")
    print("Waiting for frames from Unity...")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)


def display_frames():
    global running
    print("Display window opened. Press 'q' to quit.")
    cv2.namedWindow('Unity Stream', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Unity Stream', 1280, 720)

    while running:
        with frame_lock:
            frame = current_frame.copy() if current_frame is not None else np.zeros((720, 1280, 3), dtype=np.uint8)
        if current_frame is None:
            cv2.putText(frame, 'Waiting for Unity stream...',
                        (400, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('Unity Stream', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            break
        time.sleep(0.01)

    cv2.destroyAllWindows()
    print("Display window closed.")


if __name__ == '__main__':
    # Start Flask in background, GUI in main thread
    flask_thread = Thread(target=run_flask, daemon=True)
    flask_thread.start()

    # GUI must run on the main thread on macOS
    try:
        display_frames()
    except cv2.error as e:
        print("❌ OpenCV GUI failed to initialize:")
        print(e)
    except Exception as e:
        print("Unexpected error:", e)
