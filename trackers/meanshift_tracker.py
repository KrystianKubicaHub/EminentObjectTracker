import cv2 as cv
import numpy as np
from trackers.base_tracker import BaseTracker
from config.constants import ColorSpaces

class MeanShiftTracker(BaseTracker):
    def __init__(self, color_space='HSV'):
        super().__init__(color_space)
        self.name = "MeanShift"
        self.supports_color_space = True
        self.track_window = None
        self.term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)
        self.roi_hist = None
    
    def init(self, frame, bbox):
        self.bbox = bbox
        x, y, w, h = bbox
        self.track_window = (x, y, w, h)
        
        roi_frame = self.convert_frame(frame[y:y+h, x:x+w])
        cfg = ColorSpaces.AVAILABLE.get(self.color_space, ColorSpaces.AVAILABLE["HSV"])
        
        if self.color_space == 'HSV':
            mask = cv.inRange(roi_frame, np.array([0, 60, 32]), np.array([180, 255, 255]))
            roi_hist = cv.calcHist([roi_frame], [0], mask, [180], [0, 180])
        else:
            roi_hist = cv.calcHist([roi_frame], cfg["channels"], None, cfg["bins"], cfg["ranges"])
        
        cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)
        self.roi_hist = roi_hist
    
    def update(self, frame):
        converted = self.convert_frame(frame)
        cfg = ColorSpaces.AVAILABLE.get(self.color_space, ColorSpaces.AVAILABLE["HSV"])
        
        if self.color_space == 'HSV':
            dst = cv.calcBackProject([converted], [0], self.roi_hist, [0, 180], 1)
        else:
            dst = cv.calcBackProject([converted], cfg["channels"], self.roi_hist, cfg["ranges"], 1)
        
        ret, self.track_window = cv.meanShift(dst, self.track_window, self.term_crit)
        
        x, y, w, h = self.track_window
        return True, (x, y, w, h)
