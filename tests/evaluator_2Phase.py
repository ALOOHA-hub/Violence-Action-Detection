import os
import cv2
import time
import json
from glob import glob
from src.utils.logger import logger
from src.utils.config_loader import cfg

from src.core.perception.detector import Detector
from src.core.memory.evidence import EvidenceManager
from src.core.analysis.action_rec import ActionRecognizer

class ThesisEvaluator:
    def __init__(self, dataset_path="data/dataset"):
        logger.info("Initializing Headless Evaluator...")
        self.dataset_path = dataset_path
        
        # Load core modules (No UI, No Threading)
        self.detector = Detector()
        self.memory = EvidenceManager()
        self.brain = ActionRecognizer()
        
        self.conf_threshold = cfg['action']['threshold']
        self.trigger_count = cfg['action'].get('alert_trigger_count', 3)
        self.safe_actions = [
            'a person standing completely upright and casually walking forward', 
            'a person standing completely still and doing nothing',
            'unknown_benign_activity'
        ]

    def process_video(self, video_path):
        """Runs the AI on a single video and returns True if violence is confirmed."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False, 0
            
        alert_counters = {}
        violence_detected = False
        frame_count = 0
        
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame_count += 1
            
            # 1. Perception
            detections = self.detector.process_frame(frame)
            ready_clips = self.memory.update(frame, detections)
            
            # 2. Analysis
            for tracker_id, clip in ready_clips.items():
                scores = self.brain.get_action_score(clip)
                if not scores: continue
                
                top_action = max(scores, key=scores.get)
                top_score = scores[top_action]
                
                # 3. State Machine Logic
                if top_action not in self.safe_actions and top_score > self.conf_threshold:
                    alert_counters[tracker_id] = alert_counters.get(tracker_id, 0) + 1
                    if alert_counters[tracker_id] >= self.trigger_count:
                        violence_detected = True
                        break # Stop processing, we found violence!
                else:
                    alert_counters[tracker_id] = 0
                    
            if violence_detected:
                break
                
        end_time = time.time()
        cap.release()
        
        processing_time = end_time - start_time
        return violence_detected, frame_count, processing_time

    def run_benchmark(self):
        """Processes the dataset and calculates scientific metrics."""
        results = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
        total_frames = 0
        total_time = 0.0
        
        # --- SWE FIX: Robust File Discovery & Validation ---
        violent_dir = os.path.join(self.dataset_path, "violent")
        safe_dir = os.path.join(self.dataset_path, "safe")
        
        if not os.path.exists(violent_dir) or not os.path.exists(safe_dir):
            logger.error(f"Dataset folders missing! Looked for: {violent_dir} and {safe_dir}")
            return
            
        # Support multiple video formats
        violent_vids = []
        safe_vids = []
        for ext in ('*.mp4', '*.MP4', '*.avi', '*.AVI', '*.mkv'):
            violent_vids.extend(glob(os.path.join(violent_dir, ext)))
            safe_vids.extend(glob(os.path.join(safe_dir, ext)))

        if len(violent_vids) == 0 and len(safe_vids) == 0:
            logger.error(f"No videos found! Check if files are directly inside {violent_dir} and {safe_dir}.")
            return
            
        logger.info(f"Found {len(violent_vids)} violent videos and {len(safe_vids)} safe videos. Starting...")
        # --------------------------------------------------
        
        # 1. Test Violent Videos (Should trigger alert)
        for vid in violent_vids:
            logger.info(f"Testing Violent Video: {os.path.basename(vid)}")
            alerted, frames, p_time = self.process_video(vid)
            total_frames += frames
            total_time += p_time
            if alerted: results["TP"] += 1  # True Positive
            else: results["FN"] += 1        # False Negative

        # 2. Test Safe Videos (Should NOT trigger alert)
        for vid in safe_vids:
            logger.info(f"Testing Safe Video: {os.path.basename(vid)}")
            alerted, frames, p_time = self.process_video(vid)
            total_frames += frames
            total_time += p_time
            if alerted: results["FP"] += 1  # False Positive
            else: results["TN"] += 1        # True Negative

        if total_frames > 0:
            self._generate_report(results, total_frames, total_time)

    def _generate_report(self, res, frames, p_time):
        TP, FP, TN, FN = res["TP"], res["FP"], res["TN"], res["FN"]
        
        # Avoid division by zero
        accuracy = (TP + TN) / max((TP + TN + FP + FN), 1)
        precision = TP / max((TP + FP), 1)
        recall = TP / max((TP + FN), 1)
        fps = frames / max(p_time, 0.001)

        report = {
            "metrics": {
                "accuracy": round(accuracy, 3),
                "precision": round(precision, 3),
                "recall": round(recall, 3),
                "false_positive_rate": round(FP / max((FP + TN), 1), 3)
            },
            "performance": {
                "average_fps": round(fps, 2),
                "total_frames_processed": frames
            },
            "confusion_matrix": res
        }

        os.makedirs("data/outputs", exist_ok=True)
        with open("data/outputs/evaluation_report.json", "w") as f:
            json.dump(report, f, indent=4)
            
        logger.info("\n=== EVALUATION COMPLETE ===")
        logger.info(f"Accuracy:  {accuracy:.1%}")
        logger.info(f"Precision: {precision:.1%}")
        logger.info(f"Recall:    {recall:.1%}")
        logger.info(f"Speed:     {fps:.1f} FPS")
        logger.info("Report saved to data/outputs/evaluation_report.json")

if __name__ == "__main__":
    evaluator = ThesisEvaluator()
    evaluator.run_benchmark()