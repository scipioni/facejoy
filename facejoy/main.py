import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import math
from typing import Tuple, List, Optional

# Constants
MOUSE_SMOOTHING = 0.1  # Lower is smoother
MOUSE_SCALE = 1.0  # How much to scale face movement to mouse movement
MOUSE_DEADZONE = 0.1  # Minimal movement required to move mouse
MOUTH_OPEN_THRESHOLD = 0.4  # Ratio of mouth height to width to trigger click
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()


class FaceDetector:
    """Handles face detection using MediaPipe"""

    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    def detect(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Detect face landmarks in the image"""
        results = self.face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return None
        return results.multi_face_landmarks[0]


class FaceVisualizer:
    """Handles visualization of face landmarks"""

    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        self.drawing_styles = mp.solutions.drawing_styles

    def draw(self, image: np.ndarray, face_landmarks) -> np.ndarray:
        """Draw face landmarks on the image"""
        # self.mp_drawing.draw_landmarks(
        #     image=image,
        #     landmark_list=face_landmarks,
        #     connections=self.mp_face_mesh.FACEMESH_TESSELATION, #FACEMESH_CONTOURS,
        #     landmark_drawing_spec=self.drawing_spec,
        #     #connection_drawing_spec=self.drawing_spec,
        #     connection_drawing_spec=self.drawing_styles.get_default_face_mesh_tesselation_style()
        # )
        self.mp_drawing.draw_landmarks(
          image=image,
          landmark_list=face_landmarks,
          connections=self.mp_face_mesh.FACEMESH_CONTOURS,
          landmark_drawing_spec=None,
          connection_drawing_spec=self.drawing_styles.get_default_face_mesh_contours_style())
        # self.mp_drawing.draw_landmarks(
        #   image=image,
        #   landmark_list=face_landmarks,
        #   connections=self.mp_face_mesh.FACEMESH_IRISES,
        #   landmark_drawing_spec=None,
        #   connection_drawing_spec=self.drawing_styles.get_default_face_mesh_iris_connections_style())
        return image


class MouseController:
    """Controls mouse movement based on face position and mouth state"""

    def __init__(self):
        self.prev_mouse_pos = None
        self.click_triggered = False

    def get_normalized_face_position(
        self, face_landmarks, image_shape: Tuple[int, int]
    ) -> Tuple[float, float]:
        """Get normalized face position (0-1) in the frame"""
        # Use nose tip (landmark 1) as reference
        nose = face_landmarks.landmark[1]
        x = nose.x
        y = nose.y

        # Convert to screen coordinates (flip y-axis)
        screen_x = x
        screen_y = y #1 - y

        return screen_x, screen_y

    def get_mouth_state(self, face_landmarks) -> Tuple[float, bool]:
        """Calculate mouth openness and determine if mouth is open enough for click"""
        # Mouth outer corners (61, 291) and top/bottom (13, 14)
        left = face_landmarks.landmark[61]
        right = face_landmarks.landmark[291]
        top = face_landmarks.landmark[13]
        bottom = face_landmarks.landmark[14]

        # Calculate mouth width and height
        mouth_width = math.sqrt((right.x - left.x) ** 2 + (right.y - left.y) ** 2)
        mouth_height = math.sqrt((bottom.x - top.x) ** 2 + (bottom.y - top.y) ** 2)

        # Calculate mouth open ratio
        ratio = mouth_height / mouth_width if mouth_width > 0 else 0
        is_open = ratio > MOUTH_OPEN_THRESHOLD

        return ratio, is_open

    def update_mouse(self, face_landmarks, image_shape: Tuple[int, int]):
        """Update mouse position based on face position and handle clicks"""
        # Get normalized face position (0-1)
        face_x, face_y = self.get_normalized_face_position(face_landmarks, image_shape)

        # Apply deadzone
        if abs(face_x - 0.5) < MOUSE_DEADZONE and abs(face_y - 0.5) < MOUSE_DEADZONE:
            return

        # Convert to screen coordinates
        screen_x = face_x * SCREEN_WIDTH * MOUSE_SCALE
        screen_y = face_y * SCREEN_HEIGHT * MOUSE_SCALE

        # Apply smoothing
        if self.prev_mouse_pos:
            smoothed_x = (
                MOUSE_SMOOTHING * screen_x
                + (1 - MOUSE_SMOOTHING) * self.prev_mouse_pos[0]
            )
            smoothed_y = (
                MOUSE_SMOOTHING * screen_y
                + (1 - MOUSE_SMOOTHING) * self.prev_mouse_pos[1]
            )
        else:
            smoothed_x, smoothed_y = screen_x, screen_y
        # Move mouse
        pyautogui.moveTo(smoothed_x, smoothed_y, duration=0.0, logScreenshot=False, _pause=False)
        self.prev_mouse_pos = (smoothed_x, smoothed_y)

        # Check mouth state for click
        _, is_open = self.get_mouth_state(face_landmarks)

        if is_open and not self.click_triggered:
            print("Mouth is open")
            pyautogui.click()
            self.click_triggered = True
        elif not is_open:
            self.click_triggered = False


class FaceMouseApp:
    """Main application class"""

    def __init__(self):
        self.detector = FaceDetector()
        self.visualizer = FaceVisualizer()
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
                h, w = image.shape[:2]

                # Detect face
                face_landmarks = self.detector.detect(image)

                if face_landmarks:
                    # Visualize face
                    image = self.visualizer.draw(image, face_landmarks)

                    # Control mouse
                    self.mouse_controller.update_mouse(face_landmarks, (w, h))

                # Show the image
                cv2.imshow("Face Controlled Mouse", image)

                # Exit on 'q' key
                if cv2.waitKey(5) & 0xFF == ord("q"):
                    break

        finally:
            self.cap.release()
            cv2.destroyAllWindows()


def main():
    """Main function"""
    # Create an instance of the application
    app = FaceMouseApp()
    app.run()


if __name__ == "__main__":
    main()
