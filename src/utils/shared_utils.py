import cv2
import os
import sys
import time
from typing import Optional, Tuple


class CameraManager:
    def __init__(self, width: int = 640, height: int = 480, fps_limit: int = 30):
        self.width = width
        self.height = height
        self.fps_limit = fps_limit
        self.cap: Optional[cv2.VideoCapture] = None
        self.last_frame_time = 0
        
    def initialize(self) -> bool:
        for idx in range(3):
            self.cap = cv2.VideoCapture(idx)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.cap.set(cv2.CAP_PROP_FPS, self.fps_limit)
                
                actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
                
                print(f"Resolução: {actual_width}x{actual_height}, FPS: {actual_fps}")
                
                ret, frame = self.cap.read()
                if ret:
                    return True
                else:
                    self.cap.release()
        return False
    
    def read_frame(self) -> Tuple[bool, Optional[cv2.Mat]]:
        if not self.cap or not self.cap.isOpened():
            return False, None
            
        current_time = time.time()
        time_since_last = current_time - self.last_frame_time
        target_time = 1.0 / self.fps_limit
        
        if time_since_last < target_time:
            time.sleep(target_time - time_since_last)
            
        success, frame = self.cap.read()
        if success:
            self.last_frame_time = time.time()
            
        return success, frame
    
    def release(self):
        if self.cap:
            self.cap.release()
            self.cap = None


def setup_project_paths(current_file: str) -> dict:
    script_dir = os.path.dirname(os.path.abspath(current_file))
    
    if 'src/utils' in script_dir:
        utils_dir = os.path.dirname(script_dir)
        src_dir = os.path.dirname(utils_dir)
    elif 'src/main' in script_dir:
        src_dir = os.path.dirname(script_dir)
    else:
        src_dir = script_dir
    
    project_root = os.path.dirname(src_dir)
    
    if src_dir not in sys.path:
        sys.path.append(src_dir)
    
    return {
        'script_dir': script_dir,
        'src_dir': src_dir,
        'project_root': project_root,
        'dataset_dir': os.path.join(project_root, 'dataset'),
        'models_dir': os.path.join(project_root, 'models'),
        'faces_dir': os.path.join(project_root, 'dataset_faces')
    }


def cleanup_cv_resources():
    cv2.destroyAllWindows()
    for i in range(5):
        cv2.waitKey(1)


def add_performance_overlay(frame: cv2.Mat, fps: float, info_text: str = "") -> cv2.Mat:
    height, width = frame.shape[:2]
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, 50), (0, 0, 0), -1)
    frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    if info_text:
        cv2.putText(frame, info_text, (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame 