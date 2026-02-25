import torch
from ultralytics import YOLO
import supervision as sv
from src.utils.config_loader import cfg
from src.utils.logger import logger

class Detector:
    def __init__(self):
        # 1. Load Settings
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_path = cfg['detection']['model_weights']
        self.conf_thresh = cfg['detection']['confidence_threshold']
        self.target_classes = cfg['detection']['target_classes']

        logger.info(f"Loading YOLO model from {model_path} to {self.device}...")
        
        # 2. Initialize YOLO
        try:
            self.model = YOLO(model_path)
            self.model.to(self.device)
            
            # Warmup (prevents lag on first frame)
            self.model(torch.zeros(1, 3, 640, 640).to(self.device).half() if self.device == 'cuda' else torch.zeros(1, 3, 640, 640))
            
        except Exception as e:
            logger.critical(f"Failed to load YOLO model: {e}")
            raise e

        # 3. Initialize ByteTrack (via Supervision)
        logger.info("Initializing ByteTrack...")
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=30,
            minimum_matching_threshold=0.8,
            frame_rate=30
        )

    def process_frame(self, frame):
        """
        Input: Raw Frame (numpy array)
        Output: Tracked Detections (Supervision object)
        """
        # A. Inference
        results = self.model(frame, verbose=False, conf=self.conf_thresh)[0]

        # B. Convert to Supervision Detections
        detections = sv.Detections.from_ultralytics(results)

        # C. Filter (Keep only Persons - Class ID 0)
        # We assume '0' is person in the config. 
        # Ideally, we filter by the list in config, but for now we hardcode class_id comparison for speed
        detections = detections[detections.class_id == 0]

        # D. Update Tracker
        detections = self.tracker.update_with_detections(detections)

        return detections