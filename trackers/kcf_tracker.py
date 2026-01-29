import cv2 as cv
from trackers.base_tracker import BaseTracker

class KCFTracker(BaseTracker):
    def __init__(self, color_space=None):
        super().__init__(color_space)
        self.name = "KCF"
        self.supports_color_space = True
        self.tracker = cv.TrackerKCF_create()
        
    def init(self, frame, bbox):
        self.bbox = bbox
        frame_converted = self.convert_frame(frame)
        self.tracker.init(frame_converted, bbox)
        
    def update(self, frame):
        frame_converted = self.convert_frame(frame)
        success, bbox = self.tracker.update(frame_converted)
        if success:
            self.bbox = tuple(map(int, bbox))
        return success, self.bbox
