from .interface import IVisionReasoner
from .local_strategy import LocalOllamaStrategy
from .cloud_strategy import CloudQwenStrategy
from src.utils.config_loader import cfg

class VisionReasonerFactory:
    @staticmethod
    def create() -> IVisionReasoner:
        provider = cfg['vlm'].get('provider', 'local').lower()
        if provider == 'cloud':
            return CloudQwenStrategy()
        return LocalOllamaStrategy()