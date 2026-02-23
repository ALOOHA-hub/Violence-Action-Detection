import cv2
import base64
from .interface import IVisionReasoner
from src.utils.config_loader import cfg

class BaseVisionReasoner(IVisionReasoner):
    def __init__(self):
        self.num_frames = cfg['vlm']['extraction'].get('num_frames', 8)
        self.resize_dim = (
            cfg['vlm']['extraction'].get('resize_width', 480),
            cfg['vlm']['extraction'].get('resize_height', 270)
        )
        self.jpeg_quality = cfg['vlm']['extraction'].get('jpeg_quality', 80)
        self.prompt = cfg['vlm'].get('prompt', "Analyze frames. Output JSON.")

    def _extract_frames_as_base64(self, video_path: str) -> list:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0: return []
            
        step = max(1, total_frames // self.num_frames)
        frame_indices = [i * step for i in range(self.num_frames)]
        
        base64_frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, self.resize_dim)
                _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
                base64_frames.append(base64.b64encode(buffer).decode('utf-8'))
        cap.release()
        return base64_frames