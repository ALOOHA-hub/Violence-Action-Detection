import cv2
import queue
import threading
import time
from src.utils.logger import logger
from src.utils.config_loader import cfg

# Import our modular components
from src.core.perception.detector import Detector
from src.core.perception.visualization import Visualizer
from src.core.memory.evidence import EvidenceManager
from src.core.analysis.action_rec import ActionRecognizer

class RapidPipeline:
    """
    The Orchestrator.
    Connects Phase 1 (Perception) -> Memory -> Phase 2 (Analysis).
    """
    def __init__(self):
        logger.info("Initializing Rapid Pipeline...")
        
        # 1. Load Components
        self.detector = Detector()          # The Eyes (YOLO)
        self.memory = EvidenceManager()     # The Bridge (Buffer)
        self.brain = ActionRecognizer()     # The Brain (CoCa)
        self.visualizer = Visualizer()      # The Painter
        
        # 2. Config
        self.conf_threshold = cfg['action']['threshold'] # e.g. 0.75
        
        # 3. State
        self.alerts = {} # Stores active alerts {tracker_id: "Violence detected"}

    def run(self, source_path):
        """
        Main Loop: Reads video, processes it, and displays results.
        """
        cap = cv2.VideoCapture(source_path)
        
        # Get Display Resolution
        disp_w, disp_h = cfg['system'].get('display_resolution', [1280, 720])

        logger.info("Pipeline started. Press 'Q' to exit.")
        
        while True:
            # A. Read Frame
            ret, frame = cap.read()
            if not ret:
                break
            
            # B. Phase 1: Detection & Tracking
            detections = self.detector.process_frame(frame)
            
            # C. Update Memory (The Bridge)
            # ready_clips is a dict: { tracker_id: [16 frames] }
            ready_clips = self.memory.update(frame, detections)
            
            # D. Phase 2: Analysis (The Brain)
            # ideally, this should be in a separate thread, but for now we run it sequentially 
            # to verify logic. (We can optimize to async later if it lags).
            if ready_clips:
                for tracker_id, clip in ready_clips.items():
                    # Only analyze if we haven't already flagged them (Optimization)
                    if tracker_id not in self.alerts:
                        logger.info(f"Analyzing ID {tracker_id}...")
                        
                        # Ask the Brain
                        scores = self.brain.get_action_score(clip)
                        
                        # Check results
                        top_action = max(scores, key=scores.get)
                        top_score = scores[top_action]
                        
                        logger.info(f"ID {tracker_id}: {top_action} ({top_score:.2f})")
                        
                        # Trigger Alert?
                        # We ignore 'normal walking' and 'standing still'
                        safe_actions = ['normal walking', 'standing still']
                        if top_action not in safe_actions and top_score > self.conf_threshold:
                            self.alerts[tracker_id] = f"{top_action} ({top_score:.0%})"
                            logger.warning(f"ALERT: {top_action} detected on ID {tracker_id}!")

            # E. Visualization
            # Draw standard boxes
            out_frame = self.visualizer.draw(frame, detections)
            
            # Draw Alert Overlays
            for tracker_id, alert_msg in self.alerts.items():
                # Simple text overlay for now. 
                # Ideally, visualizer.py should handle this, but we do it here for speed.
                cv2.putText(out_frame, f"ALERT: {alert_msg}", (50, 50 + (tracker_id*30)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            # F. Display
            disp_frame = cv2.resize(out_frame, (disp_w, disp_h))
            cv2.imshow("SentinAI System", disp_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()