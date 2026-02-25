import cv2
import numpy as np
from collections import deque
from src.utils.logger import logger
from src.utils.config_loader import cfg

class EvidenceManager:
    """
    The Bridge (Memory Layer).
    Responsibility: 
    1. Holds a temporal buffer (history) for each tracked person.
    2. Pre-processes crops (Resize to 224x224) for the Action Engine.
    """
    def __init__(self):
        # Configuration
        self.window_size = cfg['action'].get('window_size', 16)
        
        # Standard input size for CoCa/CLIP
        self.target_size = (224, 224) 
        
        # The Main Database: { tracker_id: deque([frame1, frame2, ...]) }
        self.buffers = {}
        
    def update(self, frame, detections):
        """
        Input: 
            - frame: The full raw video frame (H, W, 3)
            - detections: The Supervision Detections object (boxes, tracker_ids)
        
        Output:
            - ready_clips: A Dictionary { tracker_id: [window_size frames] } of who is ready to be analyzed.
        """
        active_ids = set()
        ready_clips = {}

        # If no detections, return empty
        if detections.tracker_id is None:
            return ready_clips

        # 1. Loop through every detected person
        for box, tracker_id in zip(detections.xyxy, detections.tracker_id):
            tracker_id = int(tracker_id)
            active_ids.add(tracker_id)
            
            # 2. Crop and Resize (The "Smart" Pre-processing)
            crop = self._process_crop(frame, box)
            
            # 3. Initialize buffer if new person
            if tracker_id not in self.buffers:
                self.buffers[tracker_id] = deque(maxlen=self.window_size)
            
            # 4. Add to Memory
            self.buffers[tracker_id].append(crop)
            
            # 5. Check if we have enough history to analyze
            # We only send data if we have exactly window_size frames
            if len(self.buffers[tracker_id]) == self.window_size:
                # Convert deque to list for the AI Engine
                ready_clips[tracker_id] = list(self.buffers[tracker_id])

        # 6. Garbage Collection (Memory Cleanup)
        # If a person left the frame, delete their buffer to save RAM
        self._cleanup_inactive_ids(active_ids)

        return ready_clips

    def _process_crop(self, frame, box):
        """
        Safe cropping logic. Handles cases where the box goes off-screen.
        """
        x1, y1, x2, y2 = map(int, box)
        h, w, _ = frame.shape
        
        # Clamp coordinates to be inside the image (Prevent crashes)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        # Extract the crop
        crop = frame[y1:y2, x1:x2]
        
        # Safety check: If crop is empty (0 pixels), return a black square
        if crop.size == 0:
            return np.zeros((self.target_size[1], self.target_size[0], 3), dtype=np.uint8)
            
        # Resize to 224x224 for the AI
        return cv2.resize(crop, self.target_size)

    def _cleanup_inactive_ids(self, active_ids):
        """
        Removes buffers for IDs that are no longer visible.
        """
        # Find IDs that are in memory but NOT in the current frame
        # list() is needed because we can't delete while iterating
        existing_ids = list(self.buffers.keys())
        
        for mid in existing_ids:
            if mid not in active_ids:
                del self.buffers[mid]