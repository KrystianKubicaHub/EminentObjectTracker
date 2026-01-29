from trackers.base_tracker import BaseTracker
from ultralytics import YOLO
import numpy as np

class YOLOTracker(BaseTracker):
    def __init__(self):
        super().__init__(None)
        self.name = "YOLOv8"
        self.supports_color_space = False
        self.model = YOLO('yolov8n.pt')
        self.track_id = None
        
    def init(self, frame, bbox):
        self.bbox = bbox
        x, y, w, h = bbox
        cx, cy = x + w/2, y + h/2
        results = self.model.track(frame, persist=True, verbose=False)
        
        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            if boxes.id is not None:
                for i, box in enumerate(boxes.xywh):
                    bx, by = box[0].item(), box[1].item()
                    if abs(bx - cx) < w and abs(by - cy) < h:
                        self.track_id = int(boxes.id[i].item())
                        break
        
    def update(self, frame):
        results = self.model.track(frame, persist=True, verbose=False)
        
        if not results or len(results) == 0:
            return False, None
            
        boxes = results[0].boxes
        if boxes is None or boxes.id is None:
            return False, None
        
        for i, track_id in enumerate(boxes.id):
            if int(track_id.item()) == self.track_id:
                xyxy = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy)
                self.bbox = (x1, y1, x2-x1, y2-y1)
                return True, self.bbox
        
        return False, None
