import json
import ollama
from .base_strategy import BaseVisionReasoner
from src.utils.logger import logger
from src.utils.config_loader import cfg

class LocalOllamaStrategy(BaseVisionReasoner):
    """
    The "Eyes" (Reasoning Layer).
    Responsibility: 
    1. Takes the video file of the incident.
    2. Extracts keyframes (snapshots).
    3. Asks Qwen: "What is happening here?"
    """
    def __init__(self):
        super().__init__()
        self.client = ollama
        self.model_id = cfg['vlm'].get('model_id', 'qwen2.5vl:3b')
        logger.info(f"Local VLM Strategy Initialized: {self.model_id}")

    def analyze_incident(self, video_path: str) -> dict:
        """
        Analyzes the incident video using the local Ollama model.
        """
        b64_images = self._extract_frames_as_base64(video_path)
        if not b64_images: return {"threat_detected": False, "description": "Read error."}

        try:
            response = self.client.chat(
                model=self.model_id,
                messages=[{'role': 'user', 'content': self.prompt, 'images': b64_images}],
                format='json'
            )
            return json.loads(response['message']['content'])
        except Exception as e:
            logger.error(f"Local Ollama API Failed: {e}")
            return {"error": str(e), "threat_detected": False}