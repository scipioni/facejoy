import math
import time


class Force:
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
            return (0, 0)  # Avoid division by zero

        fx = round(100 * (self.x - self.prev_point.x) / delta_time)
        fy = round(100 * (self.y - self.prev_point.y) / delta_time)

        return (fx, fy)

    def update(self, new_x, new_y):
        # new_x = new_x / mean + (mean - 1) * self.x / mean
        # new_y = new_y / mean + (mean - 1) * self.y / mean

        new_point = Force(new_x, new_y, self)

        return new_point

    def get_force_xy(self):
        return self.force

    def get_force(self):
        fx, fy = self.force
        return int(round(100 * math.sqrt(fx**2 + fy**2)))

    def __repr__(self):
        return f"Force: x={self.x}, y={self.y}, force={self.get_force()}"
