import numpy as np
import torch
import time
from src.core.analysis.action_rec import ActionRecognizer
from src.utils.logger import logger

def run_test():
    logger.info("--- Phase 2 Smoke Test Starting ---")

    # 1. Load the Engine
    try:
        engine = ActionRecognizer()
    except RuntimeError:
        logger.error("Failed to load model. You might be out of VRAM.")
        return

    # 2. Create a Fake Video Clip (16 black frames)
    # 224x224 is the size the model expects
    fake_frames = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(16)]
    
    logger.info("Running Inference on dummy data...")
    start = time.time()
    
    # 3. Ask the AI: "What is happening in these black frames?"
    scores = engine.get_action_score(fake_frames)
    
    end = time.time()
    logger.info(f"Inference Time for 16 frames: {end - start:.4f}s")
    
    # 4. Print Results
    logger.info("--- Prediction Results ---")
    for action, score in scores.items():
        logger.info(f"{action}: {score:.4f}")

    # 5. Check VRAM usage
    if torch.cuda.is_available():
        mem = torch.cuda.memory_allocated() / 1e9 # Convert to GB
        logger.info(f"Current VRAM Usage: {mem:.2f} GB / 4.00 GB")
        if mem > 3.5:
            logger.warning("CRITICAL: VRAM is nearly full! We may need to optimize.")
    
    logger.info("--- Test Complete ---")

if __name__ == "__main__":
    run_test()