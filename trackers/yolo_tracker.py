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
        
    def reset(self):
        self.track_id = None
        self.model.predictor = None
        
    def init(self, frame, bbox):
        self.bbox = bbox
        x, y, w, h = bbox
        cx, cy = x + w/2, y + h/2
        
        print(f"[YOLO] Initializing with bbox: {bbox}")
        results = self.model.track(frame, persist=True, verbose=False)
        
        print(f"[YOLO] Results: {results}")
        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            print(f"[YOLO] Boxes detected: {len(boxes)}")
            print(f"[YOLO] Boxes.id: {boxes.id}")
            
            if boxes.id is not None:
                min_dist = float('inf')
                closest_idx = None
                
                for i, box in enumerate(boxes.xywh):
                    bx, by = box[0].item(), box[1].item()
                    dist = ((bx - cx)**2 + (by - cy)**2)**0.5
                    print(f"[YOLO] Box {i}: center=({bx}, {by}), distance from ROI center: {dist:.1f}")
                    
                    if dist < min_dist:
                        min_dist = dist
                        closest_idx = i
                
                if closest_idx is not None:
                    self.track_id = int(boxes.id[closest_idx].item())
                    print(f"[YOLO] Using closest object (distance: {min_dist:.1f}px) with track_id: {self.track_id}")
                    return
            else:
                print("[YOLO] WARNING: No track IDs in first frame")
        else:
            print("[YOLO] WARNING: No boxes detected in first frame")
        
        if self.track_id is None:
            print("[YOLO] ERROR: Failed to initialize - no object found")
            raise Exception("YOLO could not detect any object in the video")
        
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
