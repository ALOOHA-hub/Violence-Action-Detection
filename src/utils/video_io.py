import cv2
import threading
import queue
import time
from src.utils.logger import logger

class VideoStream:
    """Reads frames from a video file in a separate thread."""
    def __init__(self, path, queue_size=4):
        """Initializes the video stream."""
        self.path = path
        self.stopped = False
        self.Q = queue.Queue(maxsize=queue_size)
        
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            logger.error(f"Could not open video file: {path}")
            raise ValueError("Video file not found or corrupted.")

        # Get video properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        logger.info(f"Video Stream initialized: {self.width}x{self.height} @ {self.fps} FPS")

    def start(self):
        """Starts the thread to read frames from the video."""
        t = threading.Thread(target=self.update, args=())
        t.daemon = True # Thread dies when main program dies
        t.start()
        return self

    def update(self):
        """Worker function: Keeps reading frames until the queue is full."""
        while not self.stopped:
            if not self.Q.full():
                ret, frame = self.cap.read()
                if not ret:
                    self.stop()
                    return
                self.Q.put(frame)
            else:
                time.sleep(0.01) # Don't burn CPU if queue is full

    def read(self):
        """Main Thread calls this to get the next frame."""
        return self.Q.get()

    def running(self):
        """Returns True if the video stream is running."""
        return not self.stopped or not self.Q.empty()

    def stop(self):
        """Stops the video stream."""
        self.stopped = True
        self.cap.release()