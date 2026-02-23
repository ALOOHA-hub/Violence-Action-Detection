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
from src.core.analysis.vlm import VisionReasonerFactory

class RapidPipeline:
    def __init__(self):
        logger.info("Initializing Asynchronous Pipeline...")
        
        self.detector = Detector()
        self.memory = EvidenceManager()
        self.visualizer = Visualizer()
        self.brain = ActionRecognizer()
        self.reasoner = VisionReasonerFactory.create()
        
        self.conf_threshold = cfg['action']['threshold']
        self.alert_trigger_count = cfg['action'].get('alert_trigger_count', 3)
        
        self.safe_actions = cfg['action'].get('safe_prompts', []) + ['unknown_benign_activity']    
        self.ui_labels = cfg['action'].get('ui_labels', {})

        self.actions = {} 
        self.alert_counters = {} 
        
        self.analysis_queue = queue.Queue(maxsize=1)
        self.running = True
        self.next_target_index = 0  
        
        # Start Background Worker
        self.worker_thread = threading.Thread(target=self._analysis_worker, daemon=True)
        self.worker_thread.start()

        # ---: Incident Recording Settings ---
        self.record_incidents = cfg['system'].get('record_incident', True)
        self.fps_estimate = 30 # Default, will be updated in run()
        self.pre_buffer_size = cfg['system'].get('pre_event_seconds', 2) * self.fps_estimate
        self.post_buffer_size = cfg['system'].get('post_event_seconds', 3) * self.fps_estimate
        

        # --- Phase 3 VLM ---
        self.vlm_queue = queue.Queue()
        self.vlm_worker_thread = threading.Thread(target=self._vlm_worker, daemon=True)
        self.vlm_worker_thread.start()


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
                    display_name = self.ui_labels.get(top_action, top_action)
                    self.actions[tracker_id] = f"{display_name} ({top_score:.0%})"

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
                
               # Recognition logic
                if self.is_recording_incident:
                    # Initialize writer if this is the start of an incident
                    if self.incident_writer is None:
                        timestamp = time.strftime("%Y%m%d-%H%M%S")
                        self.current_incident_path = os.path.join(out_dir, f"incident_{timestamp}.mp4")                        
                        self.incident_writer = cv2.VideoWriter(
                            self.current_incident_path, cv2.VideoWriter_fourcc(*'mp4v'), self.fps_estimate, (w, h)
                            )
                        # Flush the PRE-EVENT buffer to file
                        for buffered_frame in self.frame_buffer:
                            self.incident_writer.write(buffered_frame)

                        logger.info(f"Writing incident evidence to {self.current_incident_path}")
                    
                    # Write the current frame
                    self.incident_writer.write(out_frame)
                    self.post_alert_counter -= 1
                    
                    # Finalize clip when aftermath window closes
                    if self.post_alert_counter <= 0:
                        self.incident_writer.release()
                        self.incident_writer = None
                        self.is_recording_incident = False
                        logger.info(f"Incident recording finalized: {self.current_incident_path}")

                        # Send to Phase 3
                        self.vlm_queue.put(self.current_incident_path)

                # UI Indicators
                status_color = (0, 165, 255) if not self.analysis_queue.empty() else (0, 255, 0)
                if self.is_recording_incident: status_color = (0, 0, 255) # Red for recording
                cv2.circle(out_frame, (30, 30), 10, status_color, -1) 

                cv2.imshow("SentinAI Async System", out_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
                
        finally:
            cap.release()

            # Graceful Shutdown Handoff
            if self.incident_writer: 
                self.incident_writer.release()
                logger.info(f"Incident recording force-finalized due to shutdown: {self.current_incident_path}")
                self.vlm_queue.put(self.current_incident_path)


            cv2.destroyAllWindows()

            # Keep the main program alive just long enough for Ollama to finish its current job
            # Prevent Race Condition 
            # Unconditionally block the main thread until the VLM finishes its API calls
            logger.info("Waiting for Phase 3 VLM to finish generating final reports...")
            self.vlm_queue.join()
            
            # Safe to kill worker loops now
            self.running = False

            logger.info("System shutdown complete.")

    def _vlm_worker(self):
            """Background thread that processes saved incident videos through the Ollama VLM."""
            while self.running:
                try:
                    # Sleep and wait for a completed video path to arrive
                    video_path = self.vlm_queue.get(timeout=1.0)
                    logger.info(f"Phase 3 Worker analyzing new evidence: {video_path}")
                    
                    # Send to Ollama (This takes a few seconds, but won't block the camera)
                    report = self.reasoner.analyze_incident(video_path)
                    
                    # Save the JSON report right next to the video file
                    report_path = video_path.replace('.mp4', '_report.json')
                    with open(report_path, 'w', encoding='utf-8') as f:
                        json.dump(report, f, indent=4)
                        
                    logger.info(f"🚨 Official Incident Report generated: {report_path}")
                    self.vlm_queue.task_done()

                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Phase 3 Worker Error: {e}")