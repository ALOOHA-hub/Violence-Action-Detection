import cv2
import queue
import threading
import time
import os
from src.utils.logger import logger
from src.utils.config_loader import cfg
from src.utils.visualization import Visualizer

# Import our modular components
from src.core.perception.detector import Detector
from src.core.memory.evidence import EvidenceManager
from src.core.analysis.action_rec import ActionRecognizer

class RapidPipeline:
    """
    The Orchestrator.
    Connects Phase 1 (Perception) -> Memory -> Phase 2 (Analysis).
    Records the final output to disk.
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
        Main Loop: Reads, Processes, Displays, and SAVES video.
        """
        cap = cv2.VideoCapture(source_path)
        if not cap.isOpened():
            logger.error(f"Could not open video: {source_path}")
            return

        # --- 1. Setup Video Recorder ---
        # We need the width/height of the ORIGINAL video to save it correctly
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Ensure output directory exists
        out_dir = cfg['paths']['output_dir']
        os.makedirs(out_dir, exist_ok=True)
        
        # Define the Output Path (e.g., data/outputs/evidence.mp4)
        out_path = os.path.join(out_dir, "evidence.mp4")
        
        # Initialize Writer (mp4v is a standard codec)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(out_path, fourcc, fps, (frame_width, frame_height))
        
        logger.info(f"Recording evidence to: {out_path}")
        
        # Get Display Resolution (For your screen only, not the file)
        disp_w, disp_h = cfg['system'].get('display_resolution', [1280, 720])

        logger.info("Pipeline started. Press 'Q' to exit.")
        
        try:
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
                if ready_clips:
                    for tracker_id, clip in ready_clips.items():
                        # Only analyze if we haven't already flagged them (Optimization)
                        if tracker_id not in self.alerts:
                            # logger.info(f"Analyzing ID {tracker_id}...")
                            scores = self.brain.get_action_score(clip)
                            top_action = max(scores, key=scores.get)
                            top_score = scores[top_action]
                            
                            # Trigger Alert logic
                            safe_actions = ['normal walking', 'standing still']
                            if top_action not in safe_actions and top_score > self.conf_threshold:
                                self.alerts[tracker_id] = f"{top_action} ({top_score:.0%})"
                                logger.warning(f"ALERT: {top_action} detected on ID {tracker_id}!")

                # E. Visualization (Draw on Frame)
                out_frame = self.visualizer.draw(frame, detections)
                
                # Draw Alert Overlays directly on the frame
                for tracker_id, alert_msg in self.alerts.items():
                    cv2.putText(out_frame, f"ALERT: {alert_msg}", (50, 50 + (tracker_id*50)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

                # --- F. SAVE THE FRAME ---
                out_writer.write(out_frame)

                # G. Display (Resize for screen)
                disp_frame = cv2.resize(out_frame, (disp_w, disp_h))
                cv2.imshow("SentinAI System", disp_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        except KeyboardInterrupt:
            logger.info("Stopping pipeline...")
            
        finally:
            # Cleanup
            cap.release()
            out_writer.release() # <--- Important: Closes the file safely
            cv2.destroyAllWindows()
            logger.info(f"Evidence saved successfully to {out_path}")