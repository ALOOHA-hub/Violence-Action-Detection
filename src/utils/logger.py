import logging
import sys
from .config_loader import cfg

def setup_logger(name="Violence Action Detection"):
    """
    Sets up a standardized logger with formatting.
    Format: [TIME] [LEVEL] [MODULE]: Message
    """
    logger = logging.getLogger(name)
    
    # prevent duplicate logs if called multiple times
    if logger.hasHandlers():
        return logger

    # Set Level from Config
    level_str = cfg['system'].get('log_level', 'INFO')
    level = getattr(logging, level_str.upper())
    logger.setLevel(level)

    # Create Console Handler
    handler = logging.StreamHandler(sys.stdout)
    
    # Define Format
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s',
        datefmt='%H:%M:%S'
    )
    
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger

# Global logger instance
logger = setup_logger()