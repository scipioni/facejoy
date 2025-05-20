import math
import time
from typing import List, Optional, Tuple
import logging
import pyautogui

from .config import config

MOUSE_SMOOTHING = 0.1  # Lower is smoother
MOUSE_SCALE = 1.0  # How much to scale face movement to mouse movement
MOUSE_DEADZONE = 0.13  # Minimal movement required to move mouse
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()



class MouseController:
    """Controls mouse movement based on face position and mouth state"""

    def __init__(self):
        self.prev_mouse_pos = None
        self.prev_face_xy = None

        #self.click_triggered = False
        #self.input_force = None #MovingPoint(0, 0)
        #self.cursor = Cursor(m=.1)

        self.previous_force = None
        self.previous_time = time.time()
        self.previous_velocity = (0, 0)

        self.x0, self.y0 = pyautogui.position()

        self.screen_width, self.screen_height = pyautogui.size()
        logging.info(f"Screen Resolution: {self.screen_width}x{self.screen_height}")


    def update_mouse(self, force_xy, click=False, limit=99.):
        """Update mouse position based on face position and handle clicks
        
        V(t) = V0 + a·t 
        S(t) = ½·a·t2 + V0·t + S0.
        """
        #if not self.input_force:
        #    self.input_force = Force(face_x, face_y)

        #self.input_force = self.input_force.update_position(face_x, face_y)
        #print(f"force={self.input_force}")

        #fx, fy = self.input_force.get_force_xy()
        #self.cursor.move(fx, fy)
        force_x, force_y = force_xy

        if not (-limit <= force_x <= limit):
            force_x = 0.
        if not (-limit <= force_y <= limit):
            force_y = 0.

        if not self.previous_force:
            self.previous_force = (force_x, force_y)
            self.previous_time = time.time()

        self.x0, self.y0 = pyautogui.position()
        now = time.time()
        delta_t = now - self.previous_time

        v0x, v0y = self.previous_velocity

        # s(t) = s₀ + v₀ * t + (1/2) * a * t²
        resistenza_x = -v0x*config.k1
        resistenza_y = -v0y*config.k1

        ax = (force_x+resistenza_x)/config.m
        ay = (force_y+resistenza_y)/config.m


        vx = v0x + ax * delta_t
        vy = v0y + ay * delta_t

        
        k3 = 200
        delta_x = (config.k2 if abs(v0x) > k3 else 1)*(v0x * delta_t + ax * delta_t**2  / 2.)
        delta_y = (config.k2 if abs(v0y) > k3 else 1)*(v0y * delta_t + ay * delta_t**2  / 2.)

        x = self.x0 + int(round(delta_x))
        y = self.y0 + int(round(delta_y))

        self.previous_velocity = (vx, vy)
        self.previous_time = now
        logging.debug(f"x={x} y={y} vx={vx:.2f}, vy={vy:.2f}")

        #return
        if 1 <= x <= self.screen_width and 1 <= self.screen_height <= self.screen_height:
            pyautogui.moveTo(
                x, y, duration=delta_t, logScreenshot=False, _pause=False
            )
            self.x0, self.y0 = x,y


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
