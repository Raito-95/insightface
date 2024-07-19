import cv2
import numpy as np
import os
import pickle
from insightface.app import FaceAnalysis
import time
import threading
import queue
import atexit

# Direct parameters
DATA_DIR = "./data/"  # directory of known faces images
RESULT_DIR = "./result/"  # directory to save result videos
CTX_ID = 0  # context id, <0 means using CPU
FPS = 10  # frame rate
VIDEO_SOURCE = 0  # video source (0 for webcam)
RETRY_INTERVAL = 5  # interval in seconds to retry opening the camera
SHOW_VIDEO = False  # flag to show video

# Ensure result directory exists
os.makedirs(RESULT_DIR, exist_ok=True)

# Initialize FaceAnalysis application
app = FaceAnalysis()
app.prepare(ctx_id=CTX_ID)

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

def find_matching_name(embedding, known_embeddings, known_names, threshold=0.5):
    sims = np.dot(known_embeddings, embedding)
    best_match_idx = np.argmax(sims)
    if sims[best_match_idx] >= threshold:
        return known_names[best_match_idx], sims[best_match_idx]
    else:
        return None, None

stop_event = threading.Event()
start_recording_event = threading.Event()

# Variables to manage video recording
recording = False
current_video_filename = None
recorded_frame_count = 0
face_detected = False
files_to_delete = []  # List to store filenames that need to be deleted

# Define condition variable and file usage status
condition = threading.Condition()
file_in_use = False

def print_camera_specs(cap):
    specs = {
        'Frame Width': cap.get(cv2.CAP_PROP_FRAME_WIDTH),
        'Frame Height': cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
        'FPS': cap.get(cv2.CAP_PROP_FPS),
    }
    for spec, value in specs.items():
        print(f"{spec}: {value}")

def capture_frames():
    global recording, current_video_filename, recorded_frame_count, face_detected, file_in_use

    while not stop_event.is_set():
        cap = cv2.VideoCapture(VIDEO_SOURCE)
        if not cap.isOpened():
            print(f"Failed to open camera. Retrying in {RETRY_INTERVAL} seconds...")
            time.sleep(RETRY_INTERVAL)
            continue

        cap.set(cv2.CAP_PROP_FPS, FPS)
        print_camera_specs(cap)

        video_writer = None
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame. Retrying...")
                break

            frame_queue.put(frame)  # Add frame to queue for processing

            if SHOW_VIDEO:
                resized_frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
                cv2.imshow("Video", resized_frame)
                if cv2.waitKey(1000 // FPS) & 0xFF == ord("q"):
                    stop_event.set()
                    break

            with condition:
                if start_recording_event.is_set() and not recording:
                    recording = True
                    recorded_frame_count = 0
                    current_video_filename = os.path.join(RESULT_DIR, f"recording_{int(time.time())}.mp4")
                    video_writer = cv2.VideoWriter(
                        current_video_filename,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        FPS,
                        (int(frame.shape[1]), int(frame.shape[0]))
                    )
                    print(f"Started recording: {current_video_filename}")
                    file_in_use = True  # File is in use
                    files_to_delete.append(current_video_filename)  # Add filename to deletion list

                if recording and video_writer is not None:
                    video_writer.write(frame)
                    recorded_frame_count += 1

                    if recorded_frame_count >= 10 * FPS:
                        video_writer.release()
                        video_writer = None
                        file_in_use = False  # File is no longer in use
                        condition.notify_all()  # Notify all waiting threads
                        if not face_detected:
                            print(f"Deleting {current_video_filename} due to no face detected")
                        else:
                            print(f"Saved {current_video_filename}")
                            if current_video_filename in files_to_delete:
                                files_to_delete.remove(current_video_filename)  # Remove filename from deletion list
                        recording = False
                        start_recording_event.clear()
                        recorded_frame_count = 0
                        face_detected = False
                        delete_pending_files()  # Delete pending files immediately

                if recording and not start_recording_event.is_set():
                    if video_writer is not None:
                        video_writer.release()
                        video_writer = None
                        file_in_use = False  # File is no longer in use
                        condition.notify_all()  # Notify all waiting threads
                    print(f"Marking {current_video_filename} for deletion due to no face detected")
                    recording = False
                    if current_video_filename is not None:
                        files_to_delete.append(current_video_filename)  # Add filename to deletion list
                    recorded_frame_count = 0
                    face_detected = False
                    delete_pending_files()  # Delete pending files immediately

        cap.release()
        if SHOW_VIDEO:
            cv2.destroyAllWindows()

def process_frames():
    global face_detected, recording, file_in_use, recorded_frame_count
    while not stop_event.is_set():
        try:
            while not frame_queue.empty():
                frame_queue.get_nowait()  # Clear old frames
            frame = frame_queue.get()  # Block until new frame is available
        except queue.Empty:
            continue

        faces = app.get(frame)
        face_present = False
        face_detected = False  # Reset face_detected for each frame processing

        if faces:
            for face in faces:
                matching_name, similarity = find_matching_name(
                    face.normed_embedding, known_embeddings, known_names
                )

                if matching_name is not None and similarity is not None:
                    face_present = True
                    face_detected = True
                    print(f"Detected known face: {matching_name} with similarity {similarity}")
                    break

        if face_present:
            start_recording_event.set()
        else:
            if recording:
                print(f"Marking {current_video_filename} for deletion due to no face detected")
                with condition:
                    recording = False
                    while file_in_use:
                        condition.wait()  # Wait until file is no longer in use
                    if current_video_filename is not None:
                        files_to_delete.append(current_video_filename)  # Add filename to deletion list
                recorded_frame_count = 0
                face_detected = False
                delete_pending_files()  # Delete pending files immediately
            start_recording_event.clear()

def delete_pending_files():
    global files_to_delete
    with condition:
        files_to_delete_copy = files_to_delete.copy()
        for filename in files_to_delete_copy:
            if not file_in_use and os.path.exists(filename):
                try:
                    os.remove(filename)
                    print(f"Deleted {filename}")
                    files_to_delete.remove(filename)
                except PermissionError:
                    print(f"Unable to delete {filename}. File is in use.")

frame_queue = queue.Queue()

capture_thread = threading.Thread(target=capture_frames)
process_thread = threading.Thread(target=process_frames)

def cleanup():
    print("Cleaning up...")
    stop_event.set()
    capture_thread.join()
    process_thread.join()
    delete_pending_files()  # Ensure all pending files are deleted
    if SHOW_VIDEO:
        cv2.destroyAllWindows()
    print("Cleanup complete.")

# Register the cleanup function to be called on exit
atexit.register(cleanup)

capture_thread.start()
process_thread.start()

# Wait for threads to finish
capture_thread.join()
process_thread.join()
