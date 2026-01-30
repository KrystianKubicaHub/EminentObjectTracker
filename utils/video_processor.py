import cv2 as cv
import numpy as np
from config.constants import Colors, Sizes

class VideoProcessor:
    def __init__(self, video_path, tracker, enable_trace=False):
        self.video_path = video_path
        self.tracker = tracker
        self.enable_trace = enable_trace
        self.capture = None
        self.fps = 0
        self.frame_count = 0
        self.current_frame = 0
        
    def open(self):
        self.capture = cv.VideoCapture(self.video_path)
        if self.capture.isOpened():
            self.fps = self.capture.get(cv.CAP_PROP_FPS)
            self.frame_count = int(self.capture.get(cv.CAP_PROP_FRAME_COUNT))
            return True
        return False
    
    def read_frame(self):
        if self.capture is None:
            return None
        ret, frame = self.capture.read()
        if ret:
            self.current_frame += 1
            return self.resize_frame(frame)
        return None
    
    def resize_frame(self, frame):
        h, w = frame.shape[:2]
        if w <= Sizes.VIDEO_MAX_WIDTH:
            return frame
        
        scale = Sizes.VIDEO_MAX_WIDTH / w
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv.resize(frame, (new_w, new_h), interpolation=cv.INTER_AREA)
    
    def draw_bbox(self, frame, result):
        if result is None:
            return frame
        
        if isinstance(result, tuple) and len(result) == 4:
            x, y, w, h = map(int, result)
            cv.rectangle(frame, (x, y), (x+w, y+h), Colors.TRACKER_BOX, 3)
        
        return frame
    
    def release(self):
        if self.capture:
            self.capture.release()
    
    def reset(self):
        if self.capture:
            self.capture.set(cv.CAP_PROP_POS_FRAMES, 0)
            self.current_frame = 0
