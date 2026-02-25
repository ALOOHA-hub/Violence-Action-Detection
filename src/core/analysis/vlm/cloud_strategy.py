import os
from .base_strategy import BaseVisionReasoner
from src.utils.logger import logger
from src.utils.config_loader import cfg

class CloudQwenStrategy(BaseVisionReasoner):
    """
    The "Eyes" (Reasoning Layer).
    Responsibility: 
    1. Takes the video file of the incident.
    2. Extracts keyframes (snapshots).
    3. Asks Qwen: "What is happening here?"
    """
    def __init__(self):
        super().__init__()
        self.model_id = cfg['vlm'].get('cloud_model_id', 'qwen-vl-max')
        logger.info(f"Cloud VLM Strategy Initialized: {self.model_id}")

    def analyze_incident(self, video_path: str) -> dict:
        """
        Analyzes the incident video using the cloud-based Qwen model.
        """
        try:
            import dashscope
            dashscope.api_key = os.getenv("QWEN_API_KEY", "MISSING_KEY")
            b64_images = self._extract_frames_as_base64(video_path)
            
            response = dashscope.MultiModalConversation.call(
                model=self.model_id,
                messages=[{'role': 'user', 'content': [{"image": f"data:image/jpeg;base64,{img}"} for img in b64_images] + [{"text": self.prompt}]}]
            )
            return {"threat_detected": True, "description": "Cloud processing placeholder."}
        except ImportError:
            logger.error("Cloud SDK 'dashscope' not installed. Run: pip install dashscope")
            return {"threat_detected": False}