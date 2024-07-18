import cv2
import numpy as np
import os
import pickle
from insightface.app import FaceAnalysis
import time
import threading
from queue import Queue, Empty

# Direct parameters
DATA_DIR = "./data/"  # directory of known faces images
CTX_ID = 0  # context id, <0 means using CPU
DET_SIZE = 640  # detection size
VIDEO_SOURCE = "output.mp4"  # video source, specify the video file name
FPS = 15  # frame rate of the source video

# Initialize FaceAnalysis application
app = FaceAnalysis()
app.prepare(ctx_id=CTX_ID, det_size=(DET_SIZE, DET_SIZE))

# Check if GPU is being used
device_info = app.models["recognition"].session.get_providers()
print(f"Using device: {device_info}")

# Build known face embeddings and names database
known_images = [
    os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".png")
]
known_names = [
    os.path.splitext(f)[0] for f in os.listdir(DATA_DIR) if f.endswith(".png")
]

known_embeddings = []

for img_path in known_images:
    img = cv2.imread(img_path)
    faces = app.get(img)
    if faces:
        known_embeddings.append(faces[0].normed_embedding)

# Save known face embeddings and names
with open("known_faces.pkl", "wb") as f:
    pickle.dump((known_embeddings, known_names), f)

# Load known face embeddings and names
with open("known_faces.pkl", "rb") as f:
    known_embeddings, known_names = pickle.load(f)

# Open video source
cap = cv2.VideoCapture(VIDEO_SOURCE)


def find_matching_name(embedding, known_embeddings, known_names):
    sims = np.dot(known_embeddings, embedding)
    best_match_idx = np.argmax(sims)
    return known_names[best_match_idx], sims[best_match_idx]


# Use Queues for multithreaded processing
frame_queue = Queue(maxsize=5)
result_queue = Queue(maxsize=5)
stop_event = threading.Event()

# Store last detection results
last_detection_results = []


def capture_frames():
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            stop_event.set()
            break
        frame_queue.put(frame)
    cap.release()


def process_frames():
    global last_detection_results
    while not stop_event.is_set() or not frame_queue.empty():
        try:
            frame = frame_queue.get(timeout=1)
        except Empty:
            continue
        last_detection_results = []
        faces = app.get(frame)
        if faces:
            for face in faces:
                matching_name, similarity = find_matching_name(
                    face.normed_embedding, known_embeddings, known_names
                )
                # print(f"Detected {matching_name} - Age: {face.age}, Gender: {'Male' if face.gender == 1 else 'Female'}, "
                #       f"BoundingBox: {face.bbox}, Similarity: {similarity:.2f}")

                # Store last detection results
                last_detection_results.append((face.bbox, matching_name, similarity))

        # Draw detection results and names on the frame
        for bbox, name, similarity in last_detection_results:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{name} ({similarity:.2f})",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (36, 255, 12),
                2,
            )

        result_queue.put(frame)


def display_frames():
    fps_count = 0
    fps_start_time = time.time()
    while not stop_event.is_set() or not result_queue.empty():
        try:
            frame = result_queue.get(timeout=1)
        except Empty:
            continue
        # Display the frame using OpenCV
        cv2.imshow("Video", frame)

        # Calculate FPS
        fps_count += 1
        if fps_count >= 10:
            fps_end_time = time.time()
            fps = fps_count / (fps_end_time - fps_start_time)
            print(f"FPS: {fps:.2f}")
            fps_count = 0
            fps_start_time = fps_end_time

        # Exit loop on 'q' key press
        if cv2.waitKey(1000 // FPS) & 0xFF == ord("q"):
            stop_event.set()
            break
    cv2.destroyAllWindows()
    stop_event.set()  # Ensure to set the event to stop other threads


# Create and start threads
capture_thread = threading.Thread(target=capture_frames)
process_thread = threading.Thread(target=process_frames)
display_thread = threading.Thread(target=display_frames)

capture_thread.start()
process_thread.start()
display_thread.start()

# Wait for all threads to complete
capture_thread.join()
process_thread.join()
display_thread.join()

# Release resources and close all OpenCV windows
cv2.destroyAllWindows()
