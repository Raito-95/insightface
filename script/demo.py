import cv2
import numpy as np
import os
import pickle
import json
import logging
from insightface.app import FaceAnalysis
import time
import threading
import queue
import atexit

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s:%(message)s")


class FaceRecognitionSystem:
    def __init__(self, config_path="config.json"):
        self.load_config(config_path)
        self.init_directories()
        self.app = FaceAnalysis()
        self.app.prepare(ctx_id=self.CTX_ID)
        self.load_known_faces()

        self.stop_event = threading.Event()
        self.start_recording_event = threading.Event()

        self.recording = False
        self.current_video_filename: str | None = None
        self.recorded_frame_count = 0
        self.face_detected = False
        self.files_to_delete = queue.Queue()

        self.condition = threading.Condition()
        self.file_in_use = False

        self.frame_queue = queue.Queue(maxsize=5)
        self.video_writer = None

    def load_config(self, config_path):
        with open(config_path) as config_file:
            config = json.load(config_file)
        self.DATA_DIR = config["data_dir"]
        self.RESULT_DIR = config["result_dir"]
        self.CTX_ID = config["ctx_id"]
        self.VIDEO_SOURCE = config["video_source"]
        self.FPS = config["fps"]
        self.VIDEO_LENGTH_SECONDS = config["video_length_seconds"]
        self.SHOW_VIDEO = config["show_video"]

    def init_directories(self):
        os.makedirs(self.RESULT_DIR, exist_ok=True)

    def load_known_faces(self):
        known_images = [
            os.path.join(self.DATA_DIR, f)
            for f in os.listdir(self.DATA_DIR)
            if f.endswith(".png")
        ]
        known_names = [
            os.path.splitext(f)[0]
            for f in os.listdir(self.DATA_DIR)
            if f.endswith(".png")
        ]

        known_embeddings = []

        for img_path in known_images:
            img = cv2.imread(img_path)
            faces = self.app.get(img)
            if faces:
                known_embeddings.append(faces[0].normed_embedding)

        with open("known_faces.pkl", "wb") as f:
            pickle.dump((known_embeddings, known_names), f)

        with open("known_faces.pkl", "rb") as f:
            self.known_embeddings, self.known_names = pickle.load(f)

    def find_matching_name(self, embedding, threshold=0.5):
        sims = np.dot(self.known_embeddings, embedding)
        best_match_idx = np.argmax(sims)
        if sims[best_match_idx] >= threshold:
            return self.known_names[best_match_idx], sims[best_match_idx]
        else:
            return None, None

    def print_camera_specs(self, cap):
        specs = {
            "Frame Width": cap.get(cv2.CAP_PROP_FRAME_WIDTH),
            "Frame Height": cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
            "FPS": cap.get(cv2.CAP_PROP_FPS),
        }
        for spec, value in specs.items():
            logging.info(f"{spec}: {value}")

    def capture_frames(self):
        while not self.stop_event.is_set():
            try:
                cap = cv2.VideoCapture(self.VIDEO_SOURCE)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 10)

                if not cap.isOpened():
                    logging.error(f"Failed to open camera. Retrying in 5 seconds...")
                    time.sleep(5)
                    continue

                cap.set(cv2.CAP_PROP_FPS, self.FPS)
                self.print_camera_specs(cap)

                while not self.stop_event.is_set():
                    ret, frame = cap.read()
                    if not ret:
                        logging.error("Failed to read frame. Retrying...")
                        self.handle_recording_failure()
                        break

                    self.record_frame(frame)

                    try:
                        self.frame_queue.put_nowait(frame)
                    except queue.Full:
                        pass

                cap.release()
            except Exception as e:
                logging.exception("Error during frame capture")
                self.cleanup()

    def handle_recording_failure(self):
        if self.recording and self.video_writer is not None:
            self.video_writer.release()
            if self.current_video_filename:
                self.files_to_delete.put(self.current_video_filename)
                logging.info(
                    f"Queued {self.current_video_filename} for deletion due to read frame failure"
                )
            self.reset_recording_state()

    def record_frame(self, frame):
        if self.start_recording_event.is_set() and not self.recording:
            self.start_new_recording(frame)

        if self.recording and self.video_writer is not None:
            self.video_writer.write(frame)
            self.recorded_frame_count += 1

            if self.recorded_frame_count >= self.VIDEO_LENGTH_SECONDS * self.FPS:
                self.finalize_recording()

    def start_new_recording(self, frame):
        self.recording = True
        self.recorded_frame_count = 0
        self.current_video_filename = os.path.join(
            self.RESULT_DIR, f"recording_{int(time.time())}.mp4"
        )
        logging.info(f"Started recording: {self.current_video_filename}")
        self.file_in_use = True
        self.video_writer = cv2.VideoWriter(
            self.current_video_filename,
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.FPS,
            (int(frame.shape[1]), int(frame.shape[0])),
        )
        self.files_to_delete.put(self.current_video_filename)

    def finalize_recording(self):
        if self.video_writer is not None:
            self.video_writer.release()
        self.file_in_use = False
        with self.condition:
            self.condition.notify_all()
        if not self.face_detected and self.current_video_filename:
            logging.info(
                f"Queued {self.current_video_filename} for deletion due to no face detected"
            )
        else:
            if self.current_video_filename:
                logging.info(f"Saved {self.current_video_filename}")
        self.reset_recording_state()

    def mark_file_for_deletion(self):
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            self.file_in_use = False
            with self.condition:
                self.condition.notify_all()
        if self.current_video_filename:
            logging.info(
                f"Queued {self.current_video_filename} for deletion due to no face detected"
            )
            self.files_to_delete.put(self.current_video_filename)
        self.reset_recording_state()

    def reset_recording_state(self):
        self.recording = False
        self.current_video_filename = None
        self.recorded_frame_count = 0
        self.face_detected = False
        self.file_in_use = False
        self.video_writer = None

    def process_frames(self):
        while not self.stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=1)
            except queue.Empty:
                continue

            faces = self.app.get(frame)
            face_present = False
            self.face_detected = False

            if faces:
                for face in faces:
                    matching_name, similarity = self.find_matching_name(
                        face.normed_embedding
                    )
                    if matching_name is not None and similarity is not None:
                        face_present = True
                        self.face_detected = True
                        bbox = face.bbox.astype(int)
                        cv2.rectangle(
                            frame,
                            (bbox[0], bbox[1]),
                            (bbox[2], bbox[3]),
                            (0, 255, 0),
                            2,
                        )
                        cv2.putText(
                            frame,
                            f"{matching_name} ({similarity:.2f})",
                            (bbox[0], bbox[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (0, 255, 0),
                            2,
                        )
                        logging.info(
                            f"Detected known face: {matching_name} with similarity {similarity}"
                        )
                        break

            if self.SHOW_VIDEO:
                resized_frame = cv2.resize(
                    frame, (frame.shape[1] // 2, frame.shape[0] // 2)
                )
                cv2.imshow("Frame", resized_frame)
                if cv2.waitKey(1000 // self.FPS) & 0xFF == ord("q"):
                    self.stop_event.set()
                    self.cleanup()
                    break

            if face_present:
                self.start_recording_event.set()
            else:
                self.handle_no_face_detected()

    def handle_no_face_detected(self):
        if self.recording:
            logging.info(
                f"Marking {self.current_video_filename} for deletion due to no face detected"
            )
            self.mark_file_for_deletion()
        self.start_recording_event.clear()

    def delete_pending_files(self):
        while not self.stop_event.is_set() or not self.files_to_delete.empty():
            try:
                filename = self.files_to_delete.get(timeout=1)
                if filename and os.path.exists(filename):
                    try:
                        os.remove(filename)
                        logging.info(f"Deleted {filename}")
                    except PermissionError:
                        logging.warning(f"Unable to delete {filename}. File is in use.")
                self.files_to_delete.task_done()
            except queue.Empty:
                continue

    def cleanup(self):
        logging.info("Cleaning up...")
        self.stop_event.set()
        if self.capture_thread is not threading.current_thread():
            self.capture_thread.join()
        if self.process_thread is not threading.current_thread():
            self.process_thread.join()
        if self.delete_thread is not threading.current_thread():
            self.delete_thread.join()
        if self.SHOW_VIDEO:
            cv2.destroyAllWindows()
        logging.info("Cleanup complete.")

    def run(self):
        self.capture_thread = threading.Thread(target=self.capture_frames)
        self.process_thread = threading.Thread(target=self.process_frames)
        self.delete_thread = threading.Thread(target=self.delete_pending_files)

        atexit.register(self.cleanup)

        self.capture_thread.start()
        self.process_thread.start()
        self.delete_thread.start()

        try:
            while (
                self.capture_thread.is_alive()
                and self.process_thread.is_alive()
                and self.delete_thread.is_alive()
            ):
                self.capture_thread.join(timeout=1)
                self.process_thread.join(timeout=1)
                self.delete_thread.join(timeout=1)
        except KeyboardInterrupt:
            self.cleanup()


if __name__ == "__main__":
    system = FaceRecognitionSystem()
    system.run()
