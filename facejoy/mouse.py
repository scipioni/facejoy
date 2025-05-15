import math
import time
from typing import List, Optional, Tuple

import pyautogui

MOUSE_SMOOTHING = 0.1  # Lower is smoother
MOUSE_SCALE = 1.0  # How much to scale face movement to mouse movement
MOUSE_DEADZONE = 0.13  # Minimal movement required to move mouse
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()


class MovingPoint:
    def __init__(self, x=0, y=0, prev_point=None):
        self.x = x
        self.y = y
        self.timestamp = time.time()
        self.prev_point = prev_point
        self.force = self._calculate_force()

    def _calculate_force(self):
        """
        force is velocity in pixels per second
        """
        if self.prev_point is None:
            return (0, 0)

        delta_time = time.time() - self.prev_point.timestamp
        if not delta_time:
            return (0, 0)  # Avoid division by zer

        vx = round(100*(self.x - self.prev_point.x) / delta_time)
        vy = round(100*(self.y - self.prev_point.y) / delta_time)

        return (vx, vy)

    def update_position(self, new_x, new_y, mean=10):
        #new_x = new_x / mean + (mean - 1) * self.x / mean
        #new_y = new_y / mean + (mean - 1) * self.y / mean

        new_point = MovingPoint(new_x, new_y, self)

        return new_point

    def get_force_xy(self):
        print(f"force={self.force}")
        return self.force

    def get_force(self):
        fx, fy = self.force
        return int(round(100*math.sqrt(fx**2 + fy**2)))

    def __repr__(self):
        return f"MovingPoint(x={self.x}, y={self.y}, force={self.get_force()} force_xy={self.force})"


class Cursor:
    def __init__(self, m=1.0):
        #self.current = pyautogui.position()
        self.previous_f = None
        self.previous_time = time.time()
        self.velocity = (0, 0)
        self.m = m

    def move(self, fx, fy):
        """
        v(t)=v0+a⋅t
        """
        if not self.previous_f:
            self.previous_f = (fx, fy)
            self.previous_time = time.time()
        
        ax = (fx-self.previous_f[0])/self.m
        ay = (fy-self.previous_f[1])/self.m
        v0x, v0y = self.velocity
        x0, y0 = pyautogui.position()
        now = time.time()
        delta_time = now - self.previous_time
        x = x0 + v0x * delta_time + ax * delta_time**2  / 2
        y = y0 + v0y * delta_time + ay * delta_time**2  / 2
        #pyautogui.moveTo(x, y, duration=0.0, logScreenshot=False, _pause=False)
        self.previous_time = now
        self.velocity = (v0x+ax*delta_time, v0y+ay*delta_time)
        self.previous_f = (fx, fy)




class MouseController:
    """Controls mouse movement based on face position and mouth state"""

    def __init__(self):
        self.prev_mouse_pos = None
        self.prev_face_xy = None

        self.click_triggered = False
        self.input_force = None #MovingPoint(0, 0)
        self.cursor = Cursor(m=.1)


    def update_mouse(self, face_x, face_y, click=False):
        """Update mouse position based on face position and handle clicks"""
        if not self.input_force:
            self.input_force = MovingPoint(face_x, face_y)

        self.input_force = self.input_force.update_position(face_x, face_y)
        #print(f"force={self.input_force}")

        fx, fy = self.input_force.get_force_xy()
        self.cursor.move(fx, fy)


        # # print(self.point)
        # # Apply deadzone
        # # if abs(face_x - 0.5) < MOUSE_DEADZONE and abs(face_y - 0.5) < MOUSE_DEADZONE:
        # #     return

        # velocity = round(100 * self.point.get_force())

        # print(f"v={velocity}")

        # # Convert to screen coordinates
        # screen_x = face_x * SCREEN_WIDTH * MOUSE_SCALE
        # screen_y = face_y * SCREEN_HEIGHT * MOUSE_SCALE

        # # Apply smoothing
        # if self.prev_mouse_pos:
        #     smoothed_x = (
        #         MOUSE_SMOOTHING * screen_x
        #         + (1 - MOUSE_SMOOTHING) * self.prev_mouse_pos[0]
        #     )
        #     smoothed_y = (
        #         MOUSE_SMOOTHING * screen_y
        #         + (1 - MOUSE_SMOOTHING) * self.prev_mouse_pos[1]
        #     )
        # else:
        #     smoothed_x, smoothed_y = screen_x, screen_y
        # # Move mouse
        # # pyautogui.moveTo(
        # #    smoothed_x, smoothed_y, duration=0.0, logScreenshot=False, _pause=False
        # # )
        # self.prev_mouse_pos = (smoothed_x, smoothed_y)

        # # Check mouth state for click
        # _, is_open = self.get_mouth_state(face_landmarks)

        # # if is_open and not self.click_triggered:
        # if click and not self.click_triggered:
        #     print("click fired")
        #     # pyautogui.click()
        #     self.click_triggered = True
        # elif not is_open:
        #     self.click_triggered = False
