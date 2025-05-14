import math
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
import time
from typing import List, Optional, Tuple
from .mouse import MouseController

RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]  # MediaPipe landmarks for right eye
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]  # MediaPipe landmarks for left eye
RIGHT_EYE = [
    33,
    7,
    163,
    144,
    145,
    153,
    154,
    155,
    133,
    173,
    157,
    158,
    159,
    160,
    161,
    246,
]
LEFT_EYE = [
    362,
    382,
    381,
    380,
    374,
    373,
    390,
    249,
    263,
    466,
    388,
    387,
    386,
    385,
    384,
    398,
]
BLINK_RATIO_THRESHOLD = 18
BLINK_DETECTION_INTERVAL = 0.5
GREEN_COLOR = (86, 241, 13)
RED_COLOR = (30, 46, 209)


def on_trackbar(val):
    pass  # We'll update this later


class FaceVisualizer:
    """Handles visualization of face landmarks"""

    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        self.drawing_spec_tesselate = self.mp_drawing.DrawingSpec(
            color=(160, 160, 160), thickness=1, circle_radius=1
        )
        self.drawing_styles = mp.solutions.drawing_styles
        self.face_landmarks = []
        self.right_blink = False
        self.left_blink = False
        self.right_ear = 0.0
        self.left_ear = 0.0
        self.blink_ratio_threshold = int(BLINK_RATIO_THRESHOLD)
        cv2.namedWindow("Parameters")
        cv2.createTrackbar(
            "blink", "Parameters", self.blink_ratio_threshold, 100, on_trackbar
        )

    def draw(self, image: np.ndarray, face_landmarks: np.ndarray) -> np.ndarray:
        """Draw face landmarks on the image"""
        self.face_landmarks = face_landmarks
        self.blink_ratio_threshold = float(
            cv2.getTrackbarPos("blink", "Parameters") / 100.0
        )
        self.mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=self.mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.drawing_spec_tesselate,
        )
        # self.mp_drawing.draw_landmarks(
        #     image=image,
        #     landmark_list=self.face_landmarks,
        #     connections=self.mp_face_mesh.FACEMESH_CONTOURS,
        #     landmark_drawing_spec=None,
        #     connection_drawing_spec=self.drawing_styles.get_default_face_mesh_contours_style(),
        # )
        # self.mp_drawing.draw_landmarks(
        #   image=image,
        #   landmark_list=face_landmarks,
        #   connections=self.mp_face_mesh.FACEMESH_IRISES,
        #   landmark_drawing_spec=None,
        #   connection_drawing_spec=self.drawing_styles.get_default_face_mesh_iris_connections_style())

        cv2.putText(
            image,
            f"Right EAR: {self.right_ear:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
        cv2.putText(
            image,
            f"Left EAR: {self.left_ear:.2f}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

        self._draw_eye(RIGHT_EYE, image, self.face_landmarks)
        self._draw_eye(LEFT_EYE, image, self.face_landmarks)
        # self.draw_eye_landmarks(image, face_landmarks, RIGHT_EYE, GREEN_COLOR)
        # if self.right_blink:
        #     self._draw_irish(RIGHT_EYE_INDICES, image, self.face_landmarks)
        if self.left_blink:
            self._draw_irish(LEFT_EYE_INDICES, image, self.face_landmarks)
        return image


    def _draw_eye(self, idxx, image: np.ndarray, face_landmarks: np.ndarray):
        h, w = image.shape[:2]

        eye_points = [
            (self.face_landmarks.landmark[i].x, self.face_landmarks.landmark[i].y)
            for i in idxx
        ]
        center_x = sum([p[0] for p in eye_points]) / len(eye_points)
        center_y = sum([p[1] for p in eye_points]) / len(eye_points)
        zoom = 1
        for p in eye_points:
            cv2.circle(
                image,
                (
                    int(w * (center_x + (p[0] - center_x) * zoom)),
                    int(h * (center_y + (p[1] - center_y) * zoom)),
                ),
                1,
                (0, 255, 255),
                -1,
            )

    def _draw_irish(self, idxx, image: np.ndarray, face_landmarks: np.ndarray):
        h, w = image.shape[:2]

        eye_points = [
            (self.face_landmarks.landmark[i].x, self.face_landmarks.landmark[i].y)
            for i in idxx
        ]

        center_x = sum([p[0] for p in eye_points]) / len(eye_points)
        center_y = sum([p[1] for p in eye_points]) / len(eye_points)
        cv2.circle(image, (int(center_x * w), int(center_y * h)), 10, (0, 255, 0), -1)

    def show(self, image, zoom=2.0):
        cv2.imshow("Parameters", cv2.resize(image, (0, 0), fx=zoom, fy=zoom))

        # Exit on 'q' key
        if cv2.waitKey(5) & 0xFF == ord("q"):
            return False
        return True


class FaceDetector(FaceVisualizer):
    """Handles face detection using MediaPipe"""

    def __init__(self):
        super().__init__()
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.last_blink_time = 0
        self.w = 0
        self.h = 0

    def detect(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Detect face landmarks in the image"""
        self.w, self.h = image.shape[1], image.shape[0]
        results = self.face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return None
        self.face_landmarks = results.multi_face_landmarks[0]
        self._detect_blinks(self.face_landmarks)
        return self.face_landmarks

    def _detect_blinks(self, face_landmarks):
        right_eye_landmarks = [
            (face_landmarks.landmark[i].x, face_landmarks.landmark[i].y)
            for i in RIGHT_EYE_INDICES
        ]
        left_eye_landmarks = [
            (face_landmarks.landmark[i].x, face_landmarks.landmark[i].y)
            for i in LEFT_EYE_INDICES
        ]
        self.right_ear = self._calculate_eye_aspect_ratio(right_eye_landmarks)
        self.left_ear = self._calculate_eye_aspect_ratio(left_eye_landmarks)

        # Detect blinks
        self.right_blink = self.right_ear < self.blink_ratio_threshold
        if self.left_ear < self.blink_ratio_threshold:
            if (time.time() - self.last_blink_time) > BLINK_DETECTION_INTERVAL:
                self.left_blink = True
        else:
            self.last_blink_time = time.time()
            self.left_blink = False
        # self.left_blink = self.left_ear < self.blink_ratio_threshold

    def _calculate_eye_aspect_ratio(self, eye_landmarks):
        # Compute the vertical distances
        vertical1 = math.dist(eye_landmarks[1], eye_landmarks[5])
        vertical2 = math.dist(eye_landmarks[2], eye_landmarks[4])

        # Compute the horizontal distance
        horizontal = math.dist(eye_landmarks[0], eye_landmarks[3])

        # Calculate eye aspect ratio
        ear = (vertical1 + vertical2) / (2.0 * horizontal)
        return ear

    def get_position_normalized(
        self,
    ) -> Tuple[float, float]:
        nose = self.face_landmarks.landmark[1]
        x = nose.x
        y = nose.y
        return (x, y)

    def draw_force(self, image: np.ndarray, force_xy, zoom=4.0):
        nose_normalized = self.get_position_normalized()
        nose_xy = (int(self.w*nose_normalized[0]), int(self.h*nose_normalized[1]))
        force_point = (nose_xy[0] + int(zoom*force_xy[0]), nose_xy[1] + int(zoom*force_xy[1]))
        cv2.line(image, nose_xy, force_point, RED_COLOR, 3)
        return image


class FaceMouseApp:
    """Main application class"""

    def __init__(self):
        self.detector = FaceDetector()
        # self.visualizer = FaceVisualizer()
        self.mouse_controller = MouseController()
        self.cap = cv2.VideoCapture(0)

    def run(self):
        """Run the main application loop"""
        try:
            while self.cap.isOpened():
                success, image = self.cap.read()
                if not success:
                    continue

                # Flip image horizontally for a mirror effect
                image = cv2.flip(image, 1)

                face_landmarks = self.detector.detect(image)

                if face_landmarks:
                    image = self.detector.draw(image, face_landmarks)

                    h, w = image.shape[:2]
                    self.mouse_controller.update_mouse(
                        face_landmarks, (w, h), click=self.detector.left_blink
                    )
                    self.detector.draw_force(image, self.mouse_controller.input_force.force)
                if not self.detector.show(image):
                    break

        finally:
            self.cap.release()
            cv2.destroyAllWindows()


def main():
    """Main function"""
    app = FaceMouseApp()
    app.run()


if __name__ == "__main__":
    main()
