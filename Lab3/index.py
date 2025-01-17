import cv2
import time
import numpy as np
from typing import Tuple, List

class FaceDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        self.smile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_smile.xml'
        )
        self.cap = cv2.VideoCapture(0)
        self.prev_time = 0
        self.fps = 0

    def detect_faces(self, gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
        return self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5
        )

    def detect_features(self, roi_gray: np.ndarray, roi_color: np.ndarray) -> Tuple[bool, bool]:
        smile_detected = False
        eyes_detected = False

        smiles = self.smile_cascade.detectMultiScale(
            roi_gray, scaleFactor=1.8, minNeighbors=20
        )
        if len(smiles) > 0:
            smile_detected = True
            for (sx, sy, sw, sh) in smiles:
                cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 2)

        eyes = self.eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            eyes_detected = True
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)

        return smile_detected, eyes_detected

    def update_fps(self):
        curr_time = time.time()
        time_diff = curr_time - self.prev_time
        self.fps = 1 / time_diff if time_diff > 0 else 0
        self.prev_time = curr_time

    def draw_status(self, frame: np.ndarray, smile_detected: bool, eyes_detected: bool):
        if not smile_detected:
            cv2.putText(frame, "Smile!", (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if not eyes_detected:
            cv2.putText(frame, "Open eyes!", (10, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"FPS: {int(self.fps)}", (10, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    def process_frame(self, frame: np.ndarray):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detect_faces(gray)

        smile_detected = False
        eyes_detected = False

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]
            smile_detected, eyes_detected = self.detect_features(roi_gray, roi_color)

        self.update_fps()
        self.draw_status(frame, smile_detected, eyes_detected)
        return frame

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            processed_frame = self.process_frame(frame)
            cv2.imshow('Face Detection', processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def __del__(self):
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    detector = FaceDetector()
    detector.run()

if __name__ == "__main__":
    main()