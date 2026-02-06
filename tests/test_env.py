# test_env.py
from src.utils.config_loader import cfg
from src.utils.logger import logger
import torch

def smoke_test():
    logger.info("--- Starting System Smoke Test ---")
    
    # 1. Test Config
    logger.info(f"Project: {cfg['project_name']} v{cfg['version']}")
    
    # 2. Test Torch/CUDA
    if torch.cuda.is_available():
        logger.info(f"GPU Detected: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("No GPU detected. System will run in CPU mode (Slow).")

    logger.info("--- Environment Check Passed ---")

if __name__ == "__main__":
    smoke_test()

# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121