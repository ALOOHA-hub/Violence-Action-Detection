import cv2
import queue
import threading
import os
import time
import json

from collections import deque
from src.utils.logger import logger
from src.utils.config_loader import cfg
from src.utils.visualization import Visualizer

from src.core.perception.detector import Detector
from src.core.memory.evidence import EvidenceManager
from src.core.analysis.action_rec import ActionRecognizer
from src.core.analysis.vlm_reasoner import IncidentReasoner
 json

class RapidPipeline:
    def __init__(self):
        logger.info("Initializing Asynchronous Pipeline...")
        
        self.detector = Detector()
        self.memory = EvidenceManager()
        self.visualizer = Visualizer()
        self.brain = ActionRecognizer() 

        # --- THE NEW PHASE 3 PIPELINE ---
        self.vlm = IncidentReasoner()
        self.vlm_queue = queue.Queue()
        
        self.conf_threshold = cfg['action']['threshold']
        self.alert_trigger_count = cfg['action'].get('alert_trigger_count', 3)
        
        self.safe_actions = [
            'a person standing completely upright and casually walking forward', 
            'a person standing completely still and doing nothing',
            'unknown_benign_activity'
        ]

        self.actions = {} 
        self.alert_counters = {} 
        
        self.analysis_queue = queue.Queue(maxsize=1)
        self.running = True
        self.next_target_index = 0  
        
        # Start Background Worker
        self.worker_thread = threading.Thread(target=self._analysis_worker, daemon=True)
        self.worker_thread.start()

        self.vlm_thread = threading.Thread(target=self._vlm_worker, daemon=True)
        self.vlm_thread.start()

        # --- SWE ARCHITECTURE: Incident Recording Settings ---
        self.record_incidents = cfg['system'].get('record_incident', True)
        self.fps_estimate = 30 # Default, will be updated in run()
        self.pre_buffer_size = cfg['system'].get('pre_event_seconds', 2) * self.fps_estimate
        self.post_buffer_size = cfg['system'].get('post_event_seconds', 3) * self.fps_estimate
        
        self.frame_buffer = deque(maxlen=self.pre_buffer_size)
        self.is_recording_incident = False
        self.post_alert_counter = 0
        self.incident_writer = None

    def _analysis_worker(self):
        while self.running:
            try:
                tracker_id, clip = self.analysis_queue.get(timeout=0.1)
                scores = self.brain.get_action_score(clip)
                
                top_action = max(scores, key=scores.get)
                top_score = scores[top_action]
                
                is_violent = top_action not in self.safe_actions
                
                if is_violent and top_score > self.conf_threshold:
                    self.alert_counters[tracker_id] = self.alert_counters.get(tracker_id, 0) + 1
                    
                    if self.alert_counters[tracker_id] >= self.alert_trigger_count:
                        display_text = top_action.split()[2] if len(top_action.split()) > 2 else top_action
                        self.actions[tracker_id] = f"🚨 {display_text.upper()} ({top_score:.0%})"
                        logger.warning(f"CONFIRMED THREAT: {top_action} on ID {tracker_id}")
                        
                        # --- TRIGGER INCIDENT RECORDING ---
                        if self.record_incidents:
                            self.is_recording_incident = True
                            self.post_alert_counter = self.post_buffer_size
                    
                    else:
                        strikes = self.alert_counters[tracker_id]
                        self.actions[tracker_id] = f"suspicious ({strikes}/{self.alert_trigger_count})"
                
                else:
                    self.alert_counters[tracker_id] = 0 
                    if "unknown_benign_activity" in top_action:
                        self.actions[tracker_id] = "analyzing..."
                    elif "walking" in top_action:
                        self.actions[tracker_id] = f"walking ({top_score:.0%})"
                    elif "standing" in top_action:
                        self.actions[tracker_id] = f"standing ({top_score:.0%})"
                    else:
                        self.actions[tracker_id] = "safe"

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
        self.fps_estimate = fps if fps > 0 else 30
        
        out_dir = cfg['paths']['output_dir']
        os.makedirs(out_dir, exist_ok=True)
        
        disp_w, disp_h = cfg['system'].get('display_resolution', [1280, 720])
        logger.info(f"Pipeline started. Monitoring for incidents...")
        
        cv2.namedWindow("SentinAI Async System", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow("SentinAI Async System", disp_w, disp_h)

        try:
            while True:
                ret, frame = cap.read()
                if not ret: break
                
                # Update Ring Buffer with Raw Frame (Pre-event context)
                self.frame_buffer.append(frame.copy())
                
                detections = self.detector.process_frame(frame)
                ready_clips = self.memory.update(frame, detections)
                
                if ready_clips:
                    available_ids = list(ready_clips.keys())
                    if available_ids:
                        idx = self.next_target_index % len(available_ids)
                        target_id = available_ids[idx]
                        if not self.analysis_queue.full():
                            self.analysis_queue.put((target_id, ready_clips[target_id]))
                            self.next_target_index += 1

                out_frame = self.visualizer.draw(frame, detections, actions=self.actions)
                
                # --- EVENT-DRIVEN RECORDING LOGIC ---
                if self.is_recording_incident:
                    # Initialize writer if this is the start of an incident
                    if self.incident_writer is None:
                        timestamp = time.strftime("%Y%m%d-%H%M%S")
                        incident_path = os.path.join(out_dir, f"incident_{timestamp}.mp4")
                        self.incident_writer = cv2.VideoWriter(
                            incident_path, cv2.VideoWriter_fourcc(*'mp4v'), self.fps_estimate, (w, h)
                        )
                        # Flush the PRE-EVENT buffer to file
                        for buffered_frame in self.frame_buffer:
                            self.incident_writer.write(buffered_frame)
                        logger.info(f"Writing incident evidence to {incident_path}")

                    # Write the current frame
                    self.incident_writer.write(out_frame)
                    self.post_alert_counter -= 1
                    
                    # Finalize clip when aftermath window closes
                    if self.post_alert_counter <= 0:
                        self.incident_writer.release()
                        self.incident_writer = None
                        self.is_recording_incident = False
                        logger.info("Incident recording finalized.")

                        # --- TRIGGER PHASE 3 ---
                        self.vlm_queue.put(incident_path)

                # UI Indicators
                status_color = (0, 165, 255) if not self.analysis_queue.empty() else (0, 255, 0)
                if self.is_recording_incident: status_color = (0, 0, 255) # Red for recording
                cv2.circle(out_frame, (30, 30), 10, status_color, -1) 

                cv2.imshow("SentinAI Async System", out_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
                
        finally:
            self.running = False 
            cap.release()
            if self.incident_writer: self.incident_writer.release()
            cv2.destroyAllWindows()
            logger.info("System shutdown complete.")


    def _vlm_worker(self):
            """Phase 3 Consumer: Runs deep reasoning on saved incident clips without blocking the camera."""
            while self.running:
                try:
                    # Wait for Phase 2 to hand off a completed video path
                    incident_path = self.vlm_queue.get(timeout=1.0)
                    logger.info(f"VLM Pipeline activated. Analyzing: {incident_path}")
                    
                    # Run the heavy CoVT reasoning
                    report = self.vlm.analyze_incident(incident_path)
                    
                    # Save the JSON report right next to the video file
                    report_path = incident_path.replace('.mp4', '_report.json')
                    with open(report_path, 'w') as f:
                        json.dump(report, f, indent=4)
                    
                    # Print the final verdict to the terminal!
                    threat = report.get('threat_level', 'UNKNOWN')
                    classification = report.get('final_classification', 'Unclassified')
                    logger.warning(f"🚨 PHASE 3 VERDICT [{threat}]: {classification}")
                    logger.info(f"Full report saved to: {report_path}")
                    
                    self.vlm_queue.task_done()

                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"VLM Worker Error: {e}")