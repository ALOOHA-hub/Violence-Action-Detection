import cv2
import base64
import json
import ollama
from src.utils.logger import logger
from src.utils.config_loader import cfg

class IncidentReasoner:
    def __init__(self):
        logger.info("Initializing Phase 3: Ollama VLM Reasoner...")
        
        # Pull configuration dynamically (Single Source of Truth)
        self.model_id = cfg['vlm'].get('model_id', 'qwen2.5vl:3b')
        self.num_frames = cfg['vlm']['extraction'].get('num_frames', 8)
        self.resize_dim = (
            cfg['vlm']['extraction'].get('resize_width', 480),
            cfg['vlm']['extraction'].get('resize_height', 270)
        )
        self.jpeg_quality = cfg['vlm']['extraction'].get('jpeg_quality', 80)
        self.prompt = cfg['vlm'].get('prompt', "Analyze these frames. Output JSON with 'threat_detected' and 'description'.")

        logger.info(f"Connected to local Ollama service using {self.model_id}.")

    def _extract_frames_as_base64(self, video_path):
        """Extracts evenly spaced frames from a video and encodes them to base64."""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            logger.error(f"Cannot read frames from {video_path}")
            return []
            
        # Calculate timestamps to grab an even spread across the video
        step = max(1, total_frames // self.num_frames)
        frame_indices = [i * step for i in range(self.num_frames)]
        
        base64_frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Downscale to protect memory and speed up inference
                frame = cv2.resize(frame, self.resize_dim)
                # Encode to JPEG, then to Base64
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
                _, buffer = cv2.imencode('.jpg', frame, encode_param)
                b64_str = base64.b64encode(buffer).decode('utf-8')
                base64_frames.append(b64_str)
                
        cap.release()
        return base64_frames

    def analyze_incident(self, video_path):
        """
        Takes a 5-second incident MP4, extracts frames, and pings the local Ollama API.
        """
        logger.info(f"VLM extracting {self.num_frames} frames from incident footage: {video_path}")
        
        # 1. Get the visual context
        b64_images = self._extract_frames_as_base64(video_path)
        if not b64_images:
            return {"threat_detected": False, "description": "Video read error."}

        try:
            # 2. Call the local Ollama API
            response = ollama.chat(
                model=self.model_id,
                messages=[{
                    'role': 'user',
                    'content': self.prompt,
                    'images': b64_images
                }],
                # SWE Feature: Force the LLM to reply strictly in JSON format
                format='json'
            )
            
            # 3. Parse and return the output
            output_text = response['message']['content']
            return json.loads(output_text)

        except Exception as e:
            logger.error(f"Ollama API Call Failed: {e}")
            return {"error": str(e), "threat_detected": False, "description": "API Failure"}