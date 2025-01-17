import cv2
import numpy as np
from typing import Optional, Tuple

class BallTracker:
    def __init__(self, video_path: str):
        self.cap = cv2.VideoCapture(video_path)
        self.lower_yellow = np.array([5, 100, 100])
        self.upper_yellow = np.array([60, 255, 255])

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[tuple]]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)
        center = self._find_ball_center(mask)
        annotated_frame = self._draw_annotations(frame, mask, center)
        return annotated_frame, center

    def _find_ball_center(self, mask: np.ndarray) -> Optional[tuple]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)

        if M["m00"] == 0:
            return None

        return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

    def _draw_annotations(self, frame: np.ndarray, mask: np.ndarray,
                        center: Optional[tuple]) -> np.ndarray:
        annotated = frame.copy()

        if center:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            largest_contour = max(contours, key=cv2.contourArea)

            cv2.drawContours(annotated, [largest_contour], -1, (0, 255, 0), 2)
            cv2.circle(annotated, center, 5, (0, 0, 255), -1)
            text = f"Center: ({center[0]}, {center[1]})"
        else:
            text = "Object not found"

        cv2.putText(annotated, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return annotated

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            annotated_frame, _ = self.process_frame(frame)
            cv2.imshow('Frame', annotated_frame)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

    def __del__(self):
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    video_path = 'C:\KFU\M.K.11-209-Robotics\Lab2\catball.mp4'
    tracker = BallTracker(video_path)
    tracker.run()

if __name__ == "__main__":
    main()