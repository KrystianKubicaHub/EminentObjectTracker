import cv2 as cv
import numpy as np
from config.constants import Colors

class TraceDrawer:
    def __init__(self):
        self.trace_points = []
        self.max_points = 500
        
    def add_point(self, center):
        if center is not None:
            self.trace_points.append(center)
            if len(self.trace_points) > self.max_points:
                self.trace_points.pop(0)
    
    def draw(self, frame):
        if len(self.trace_points) < 2:
            return frame
        
        points = np.array(self.trace_points, dtype=np.int32)
        for i in range(1, len(points)):
            alpha = i / len(points)
            thickness = max(1, int(3 * alpha))
            cv.line(frame, tuple(points[i-1]), tuple(points[i]), Colors.TRACE_LINE, thickness)
        
        return frame
    
    def clear(self):
        self.trace_points = []
