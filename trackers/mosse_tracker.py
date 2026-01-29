import cv2 as cv
from trackers.base_tracker import BaseTracker

class MOSSETracker(BaseTracker):
    def __init__(self):
        super().__init__(None)
        self.name = "MOSSE"
        self.supports_color_space = False
        self.tracker = cv.legacy.TrackerMOSSE_create()
        
    def init(self, frame, bbox):
        self.bbox = bbox
        self.tracker.init(frame, bbox)
        
    def update(self, frame):
        success, bbox = self.tracker.update(frame)
        if success:
            self.bbox = tuple(map(int, bbox))
        return success, self.bbox
