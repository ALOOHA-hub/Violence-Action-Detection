import cv2
import queue
import threading
import os
from src.utils.logger import logger
from src.utils.config_loader import cfg
from src.utils.visualization import Visualizer

from src.core.perception.detector import Detector
from src.core.memory.evidence import EvidenceManager
from src.core.analysis.action_rec import ActionRecognizer

class RapidPipeline:
    def __init__(self):
        logger.info("Initializing Asynchronous Pipeline...")
        
        self.detector = Detector()
        self.memory = EvidenceManager()
        self.visualizer = Visualizer()
        self.brain = ActionRecognizer() 
        
        self.conf_threshold = cfg['action']['threshold']
        self.alert_trigger_count = cfg['action'].get('alert_trigger_count', 3)
        # Must match the new config prompts exactly
        self.safe_actions = [
            'a person standing completely upright and casually walking forward', 
            'a person standing completely still and doing nothing'
        ]

        # Shared State for Visualization
        self.actions = {} 
        self.alert_counters = {} # Track how many times an action has been detected
        
        # Async Threading Setup
        self.analysis_queue = queue.Queue(maxsize=1)
        self.running = True
        self.next_target_index = 0  
        
        # Start Background Worker
        self.worker_thread = threading.Thread(target=self._analysis_worker, daemon=True)
        self.worker_thread.start()

    def _analysis_worker(self):
        """Runs in the background so the video never freezes."""
        while self.running:
            try:
                tracker_id, clip = self.analysis_queue.get(timeout=0.1)
                
                # Heavy Math
                scores = self.brain.get_action_score(clip)
                
                top_action = max(scores, key=scores.get)
                top_score = scores[top_action]
                
                # --- EVENT-DRIVEN STATE MACHINE (DEBOUNCING) ---
                is_violent = top_action not in self.safe_actions
                
                if is_violent and top_score > self.conf_threshold:
                    # Increment strike counter
                    self.alert_counters[tracker_id] = self.alert_counters.get(tracker_id, 0) + 1
                    
                    # State: CONFIRMED ALERT (3+ strikes)
                    if self.alert_counters[tracker_id] >= self.alert_trigger_count:
                        # Simplify the label for the screen (e.g., just show "PUNCHING" instead of the whole sentence)
                        display_text = top_action.split()[2] if len(top_action.split()) > 2 else top_action
                        self.actions[tracker_id] = f"🚨 {display_text.upper()} ({top_score:.0%})"
                        logger.warning(f"CONFIRMED THREAT: {top_action} on ID {tracker_id}")
                    
                    # State: SUSPICIOUS (1-2 strikes)
                    else:
                        strikes = self.alert_counters[tracker_id]
                        self.actions[tracker_id] = f"suspicious ({strikes}/{self.alert_trigger_count})"
                
                # State: IDLE (Safe)
                else:
                    self.alert_counters[tracker_id] = 0 # Reset counter
                    
                    # Show a clean, short label on screen so the UI looks nice
                    if "walking" in top_action:
                        self.actions[tracker_id] = f"walking ({top_score:.0%})"
                    elif "standing" in top_action:
                        self.actions[tracker_id] = f"standing ({top_score:.0%})"
                    else:
                        self.actions[tracker_id] = f"safe ({top_score:.0%})"

                self.analysis_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker Error: {e}")

    def run(self, source_path):
        cap = cv2.VideoCapture(source_path)
        if not cap.isOpened(): 
            logger.error("Failed to open input video.")
            return

        w, h, fps = int(cap.get(3)), int(cap.get(4)), int(cap.get(5))
        out_dir = cfg['paths']['output_dir']
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "evidence_async.mp4")
        out_writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        
        disp_w, disp_h = cfg['system'].get('display_resolution', [1280, 720])
        logger.info(f"Pipeline started. Recording to {out_path}")
        
        # --- SWE UI FIX: Responsive Window Rendering ---
        # 1. Create a resizable window that forces the aspect ratio to stay perfect
        cv2.namedWindow("SentinAI Async System", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        
        # 2. Set the default UI window size
        cv2.resizeWindow("SentinAI Async System", disp_w, disp_h)

        try:
            while True:
                ret, frame = cap.read()
                if not ret: break
                
                detections = self.detector.process_frame(frame)
                ready_clips = self.memory.update(frame, detections)
                
                # Round-Robin Scheduler
                if ready_clips:
                    available_ids = list(ready_clips.keys())
                    if available_ids:
                        idx = self.next_target_index % len(available_ids)
                        target_id = available_ids[idx]
                        
                        if not self.analysis_queue.full():
                            self.analysis_queue.put((target_id, ready_clips[target_id]))
                            self.next_target_index += 1

                # Draw with actions
                out_frame = self.visualizer.draw(frame, detections, actions=self.actions)
                
                # Status Indicator (Green = Free, Orange = Thinking)
                status_color = (0, 255, 0) if self.analysis_queue.empty() else (0, 165, 255)
                cv2.circle(out_frame, (30, 30), 10, status_color, -1) 

                out_writer.write(out_frame)
                
                cv2.imshow("SentinAI Async System", out_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'): break
                
        finally:
            self.running = False 
            cap.release()
            out_writer.release()
            cv2.destroyAllWindows()
            logger.info("System shutdown complete.")