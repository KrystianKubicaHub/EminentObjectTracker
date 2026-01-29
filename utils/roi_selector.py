import cv2 as cv
import numpy as np
from config.constants import Colors

class ROISelector:
    def __init__(self, max_width=1600):
        self.max_width = max_width
        
    def resize_frame(self, frame):
        h, w = frame.shape[:2]
        if w <= self.max_width:
            return frame, 1.0
        
        scale = self.max_width / w
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv.resize(frame, (new_w, new_h), interpolation=cv.INTER_AREA)
        return resized, scale
        
    def select(self, video_capture):
        ret, frame = video_capture.read()
        if not ret:
            return None
        
        display_frame, scale = self.resize_frame(frame)
        
        window_name = "Select Region of Interest"
        cv.namedWindow(window_name, cv.WINDOW_NORMAL)
        cv.resizeWindow(window_name, display_frame.shape[1], display_frame.shape[0])
        
        region = cv.selectROI(window_name, display_frame, fromCenter=False, showCrosshair=True)
        cv.destroyWindow(window_name)
        
        if region[2] == 0 or region[3] == 0:
            return None
        
        x, y, w, h = map(int, region)
        if scale != 1.0:
            x = int(x / scale)
            y = int(y / scale)
            w = int(w / scale)
            h = int(h / scale)
        
        return (x, y, w, h)
