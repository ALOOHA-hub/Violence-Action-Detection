import importlib
from .interface import IVisionReasoner
from src.utils.config_loader import cfg
from src.utils.logger import logger

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
        
        # Look up the strategy details in the config
        strategy_config = cfg['vlm'].get('strategies', {}).get(provider)
        
        if not strategy_config:
            logger.error(f"No strategy configured for VLM provider: {provider}. Falling back to LocalOllamaStrategy.")
            from .local_strategy import LocalOllamaStrategy
            return LocalOllamaStrategy()
            
        module_path = strategy_config.get('module')
        class_name = strategy_config.get('class_name')
        
        try:
            # Dynamically import the module
            module = importlib.import_module(module_path)
            # Retrieve the class from the module
            StrategyClass = getattr(module, class_name)
            # Instantiate and return the strategy class
            return StrategyClass()
        except (ImportError, AttributeError) as e:
            logger.error(f"Failed to dynamically load VLM Strategy {module_path}.{class_name}: {e}")
            raise e