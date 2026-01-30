import cv2 as cv
import numpy as np
import json
from datetime import datetime
from pathlib import Path

class BenchmarkEngine:
    def __init__(self, video_path, roi, models_list):
        self.video_path = video_path
        self.roi = roi
        self.models_list = models_list
        self.results = {}
        self.reference_data = None
        self.errors = {}
        
    def run_benchmark(self, progress_callback=None):
        total_models = len(self.models_list)
        completed = 0
        
        for idx, (model_name, tracker) in enumerate(self.models_list):
            if progress_callback:
                progress_callback(completed, total_models, f"Testing {model_name}... ({completed}/{total_models} completed)")
            
            frame_data = self.track_video(tracker, model_name)
            self.results[model_name] = frame_data
            
            if model_name.startswith("YOLOv8"):
                self.reference_data = frame_data
            
            completed += 1
        
        if progress_callback:
            progress_callback(total_models, total_models, f"Calculating errors... ({total_models}/{total_models} completed)")
        
        self.calculate_errors()
        
        return self.results
    
    def track_video(self, tracker, model_name):
        capture = cv.VideoCapture(self.video_path)
        frame_data = []
        frame_idx = 0
        
        ret, first_frame = capture.read()
        if not ret:
            capture.release()
            return frame_data
        
        try:
            tracker.init(first_frame, self.roi)
        except Exception as e:
            print(f"[Benchmark] Failed to init {model_name}: {e}")
            capture.release()
            return frame_data
        
        while True:
            ret, frame = capture.read()
            if not ret:
                break
            
            success, bbox = tracker.update(frame)
            
            if success and bbox:
                if isinstance(bbox, tuple) and len(bbox) == 4:
                    x, y, w, h = bbox
                    cx = x + w / 2
                    cy = y + h / 2
                    area = w * h
                else:
                    cx, cy, area = None, None, None
            else:
                cx, cy, area = None, None, None
            
            frame_data.append({
                'frame': frame_idx,
                'cx': cx,
                'cy': cy,
                'area': area,
                'success': success
            })
            
            frame_idx += 1
        
        capture.release()
        return frame_data
    
    def calculate_errors(self):
        if not self.reference_data:
            return
        
        errors_dict = {}
        
        for model_name, frame_data in self.results.items():
            if model_name.startswith("YOLOv8"):
                errors_dict[model_name] = {
                    'mse_position': 0.0,
                    'mse_area': 0.0
                }
                continue
            
            mse_position = 0.0
            mse_area = 0.0
            valid_frames = 0
            
            for i, frame_info in enumerate(frame_data):
                if i >= len(self.reference_data):
                    break
                
                ref = self.reference_data[i]
                
                if frame_info['cx'] is not None and ref['cx'] is not None:
                    dx = frame_info['cx'] - ref['cx']
                    dy = frame_info['cy'] - ref['cy']
                    mse_position += dx**2 + dy**2
                    
                    if frame_info['area'] is not None and ref['area'] is not None:
                        area_diff = frame_info['area'] - ref['area']
                        mse_area += area_diff**2
                    
                    valid_frames += 1
            
            if valid_frames > 0:
                mse_position /= valid_frames
                mse_area /= valid_frames
            
            errors_dict[model_name] = {
                'mse_position': np.sqrt(mse_position),
                'mse_area': np.sqrt(mse_area)
            }
        
        self.errors = errors_dict
    
    def export_to_json(self, output_path=None):
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"benchmark_results_{timestamp}.json"
        
        export_data = {
            'video_path': self.video_path,
            'roi': self.roi,
            'timestamp': datetime.now().isoformat(),
            'models': {},
            'errors': self.errors
        }
        
        for model_name, frame_data in self.results.items():
            export_data['models'][model_name] = {
                'frames': frame_data
            }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return output_path
    
    def get_error_summary(self):
        summary = []
        for model_name, error_data in self.errors.items():
            summary.append({
                'model': model_name,
                'mse_position': error_data['mse_position'],
                'mse_area': error_data['mse_area']
            })
        
        summary.sort(key=lambda x: x['mse_position'])
        return summary
