import cv2 as cv
import numpy as np
from trackers.base_tracker import BaseTracker
from config.constants import ColorSpaces

class CamShiftTracker(BaseTracker):
    def __init__(self, color_space="HSV"):
        super().__init__(color_space)
        self.name = "CamShift"
        self.supports_color_space = True
        self.histogram = None
        self.track_window = None
        
    def init(self, frame, bbox):
        self.bbox = bbox
        x, y, w, h = bbox
        
        roi = frame[y:y+h, x:x+w]
        roi_converted = self.convert_frame(roi)
        
        cfg = ColorSpaces.AVAILABLE.get(self.color_space, ColorSpaces.AVAILABLE["HSV"])
        
        if self.color_space == "HSV":
            h_channel, s_channel, v_channel = cv.split(roi_converted)
            
            h_mean, h_std = np.mean(h_channel), np.std(h_channel)
            s_mean, s_std = np.mean(s_channel), np.std(s_channel)
            v_mean, v_std = np.mean(v_channel), np.std(v_channel)
            
            tolerance = 1.5
            lower_bound = np.array([
                max(0, h_mean - tolerance * h_std),
                max(0, s_mean - tolerance * s_std),
                max(0, v_mean - tolerance * v_std)
            ])
            upper_bound = np.array([
                min(179, h_mean + tolerance * h_std),
                min(255, s_mean + tolerance * s_std),
                min(255, v_mean + tolerance * v_std)
            ])
            
            mask = cv.inRange(roi_converted, lower_bound, upper_bound)
            self.histogram = cv.calcHist([roi_converted], [0], mask, [180], [0, 180])
        else:
            self.histogram = cv.calcHist(
                [roi_converted],
                cfg["channels"],
                None,
                cfg["bins"],
                cfg["ranges"]
            )
        
        cv.normalize(self.histogram, self.histogram, 0, 255, cv.NORM_MINMAX)
        self.track_window = tuple(bbox)
        
    def update(self, frame):
        if self.histogram is None or self.track_window is None:
            return False, None
        
        frame_converted = self.convert_frame(frame)
        cfg = ColorSpaces.AVAILABLE.get(self.color_space, ColorSpaces.AVAILABLE["HSV"])
        
        back_project = cv.calcBackProject(
            [frame_converted],
            cfg["channels"],
            self.histogram,
            cfg["ranges"],
            1
        )
        
        term_criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)
        ret, self.track_window = cv.CamShift(back_project, self.track_window, term_criteria)
        
        if ret[0][0] > 0 and ret[0][1] > 0:
            (cx, cy), (w, h), angle = ret
            x = int(cx - w/2)
            y = int(cy - h/2)
            self.bbox = (x, y, int(w), int(h))
            return True, self.bbox
        return False, None
