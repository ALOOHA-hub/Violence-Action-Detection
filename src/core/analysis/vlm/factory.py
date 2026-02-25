from .interface import IVisionReasoner
from .local_strategy import LocalOllamaStrategy
from .cloud_strategy import CloudQwenStrategy
from src.utils.config_loader import cfg

class VisionReasonerFactory:
    """
    The "Eyes" (Reasoning Layer).
    Responsibility: 
    1. Takes the video file of the incident.
    2. Extracts keyframes (snapshots).
    3. Asks Qwen: "What is happening here?"
    """
    @staticmethod
    def create() -> IVisionReasoner:
        provider = cfg['vlm'].get('provider', 'local').lower()
        if provider == 'cloud':
            return CloudQwenStrategy()
        return LocalOllamaStrategy()