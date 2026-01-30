import cv2 as cv
import numpy as np
from config.constants import Colors, Sizes

class TraceDrawer:
    def __init__(self, color=None, thickness=None):
        self.trace_points = []
        self.max_points = Sizes.MAX_TRACE_POINTS
        self.color = color if color else Colors.TRACE_LINE
        self.thickness = thickness if thickness else Colors.TRACE_THICKNESS
        
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
            cv.line(frame, tuple(points[i-1]), tuple(points[i]), self.color, self.thickness)
        
        return frame
    
    def clear(self):
        self.trace_points = []
