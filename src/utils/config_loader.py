import yaml
import os
from pathlib import Path

class ConfigLoader:
    """
    Singleton class to load and access configuration settings.
    Implements the Singleton Pattern to ensure one instance system-wide.
    """
    _instance = None
    _config = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigLoader, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        # Dynamically find the config file relative to this script
        base_dir = Path(__file__).resolve().parent.parent.parent
        config_path = base_dir / "configs" / "config.yaml"

        if not config_path.exists():
            raise FileNotFoundError(f"[CRITICAL] Config file missing at: {config_path}")
        
        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f)
            print(f"[SYSTEM] Configuration loaded from {config_path}")

    @property
    def config(self):
        return self._config

# Expose a single global instance
cfg = ConfigLoader().config