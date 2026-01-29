from abc import ABC, abstractmethod
import cv2 as cv

class BaseTracker(ABC):
    def __init__(self, color_space=None):
        self.name = "BaseTracker"
        self.supports_color_space = False
        self.color_space = color_space
        self.bbox = None
        
    @abstractmethod
    def init(self, frame, bbox):
        pass
        
    @abstractmethod
    def update(self, frame):
        pass
        
    def get_center(self, bbox):
        if bbox is None or len(bbox) != 4:
            return None
        x, y, w, h = bbox
        return (int(x + w/2), int(y + h/2))
    
    def convert_frame(self, frame):
        if not self.supports_color_space or self.color_space is None or self.color_space == "RGB":
            return frame
        
        from config.constants import ColorSpaces
        if self.color_space in ColorSpaces.AVAILABLE:
            conversion = getattr(cv, ColorSpaces.AVAILABLE[self.color_space]["convert"])
            return cv.cvtColor(frame, conversion)
        return frame
