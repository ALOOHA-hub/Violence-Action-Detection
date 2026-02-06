import cv2
import time
from src.utils.config_loader import cfg
from src.utils.logger import logger
from src.utils.video_io import VideoStream
from src.utils.visualization import Visualizer
from src.core.detector import Detector

def main():
    logger.info("--- Phase 1: Detection & Tracking System Starting ---")

    # 1. Setup
    video_path = cfg['paths']['input_source']
    
    # Initialize Core Components
    stream = VideoStream(video_path).start()
    detector = Detector()
    visualizer = Visualizer()

    logger.info("System Ready. Processing video...")
    
    # Performance Monitoring
    fps_start = time.time()
    frame_count = 0

    try:
        while stream.running():
            # 2. Get Frame (Non-blocking)
            frame = stream.read()
            if frame is None: break

            # 3. Detect & Track (The AI)
            detections = detector.process_frame(frame)

            # 4. Visualize (The Output)
            output_frame = visualizer.draw(frame, detections)

            # 5. Display
            cv2.imshow("SentinAI - Phase 1", output_frame)
            
            # FPS Calc
            frame_count += 1
            if frame_count % 30 == 0:
                elapsed = time.time() - fps_start
                logger.info(f"FPS: {30/elapsed:.2f} | Active Tracks: {len(detections)}")
                fps_start = time.time()

            # Exit logic (Press 'q')
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        logger.info("Stopping system...")
    finally:
        stream.stop()
        cv2.destroyAllWindows()
        logger.info("System Shutdown Complete.")

if __name__ == "__main__":
    main()