import math
from typing import List, Optional, Tuple
import time

import pyautogui

MOUSE_SMOOTHING = 0.1  # Lower is smoother
MOUSE_SCALE = 1.0  # How much to scale face movement to mouse movement
MOUSE_DEADZONE = 0.13  # Minimal movement required to move mouse
MOUTH_OPEN_THRESHOLD = 0.4  # Ratio of mouth height to width to trigger click
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()

class MovingPoint:
    def __init__(self, x=0, y=0, prev_point=None):
        """
        Initialize a MovingPoint with coordinates (x, y).
        If prev_point is provided, velocity is calculated relative to it.
        
        :param x: current x coordinate
        :param y: current y coordinate
        :param prev_point: previous MovingPoint instance (optional)
        """
        self.x = x
        self.y = y
        self.timestamp = time.time()
        self.prev_point = prev_point
        self.velocity = self._calculate_velocity()
    
    def _calculate_velocity(self):
        """
        Calculate velocity based on previous point.
        Returns (vx, vy) or (0, 0) if no previous point.
        """
        if self.prev_point is None:
            return (0, 0)
        
        delta_time = time.time() - self.prev_point.timestamp
        if not delta_time:
            return (0, 0)  # Avoid division by zer
        
        vx = (self.x - self.prev_point.x)/delta_time
        vy = (self.y - self.prev_point.y)/delta_time


        return (vx, vy)
    
    def update_position(self, new_x, new_y):
        """
        Update the point's position and recalculate velocity.
        The current point becomes the previous point for future velocity calculations.
        
        :param new_x: new x coordinate
        :param new_y: new y coordinate
        :return: the new MovingPoint instance with updated position and velocity
        """
        
        # media
        n=8
        new_x = new_x/n + (n-1)*self.x/n
        new_y = new_y/n + (n-1)*self.y/n 

        
        new_point = MovingPoint(new_x, new_y, self)

        return new_point
    
    def get_velocity_xy(self):
        """
        Get the current velocity as a tuple (vx, vy)
        """
        return self.velocity
    
    def get_velocity(self):
        # module of vector vx and vy
        vx, vy = self.velocity
        return math.sqrt(vx**2 + vy**2)    
    def __repr__(self):
        return f"MovingPoint(x={self.x}, y={self.y}, velocity={self.velocity})"
    
class MouseController:
    """Controls mouse movement based on face position and mouth state"""

    def __init__(self):
        self.prev_mouse_pos = None
        self.prev_face_xy = None

        self.click_triggered = False
        self.point = MovingPoint(0, 0)

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
        screen_y = y  # 1 - y

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

    def update_mouse(self, face_landmarks, image_shape: Tuple[int, int], click=False):
        """Update mouse position based on face position and handle clicks"""
        # Get normalized face position (0-1)
        face_x, face_y = self.get_normalized_face_position(face_landmarks, image_shape)
        #print("x", abs(face_x - 0.5), MOUSE_DEADZONE)
        #print("y",abs(face_y - 0.5), MOUSE_DEADZONE)

        self.point = self.point.update_position(face_x, face_y)
        #print(self.point)
        # Apply deadzone
        # if abs(face_x - 0.5) < MOUSE_DEADZONE and abs(face_y - 0.5) < MOUSE_DEADZONE:
        #     return
        

        velocity = round(100*self.point.get_velocity())

        print(f"v={velocity}")

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
        #pyautogui.moveTo(
        #    smoothed_x, smoothed_y, duration=0.0, logScreenshot=False, _pause=False
        #)
        self.prev_mouse_pos = (smoothed_x, smoothed_y)

        # Check mouth state for click
        _, is_open = self.get_mouth_state(face_landmarks)

        #if is_open and not self.click_triggered:
        if click and not self.click_triggered:

            print("click fired")
            #pyautogui.click()
            self.click_triggered = True
        elif not is_open:
            self.click_triggered = False
