import cv2

class Visualizer:
    """
    Draws dynamic bounding boxes based on the State Manager's threat level.
    """
    def draw(self, frame, detections, state_manager):
        """
        Draws dynamic bounding boxes based on the State Manager's threat level.
        
        Args:
            frame: The frame to draw on.
            detections: The detections to draw.
            state_manager: The state manager to get the threat level from.
        
        Returns:
            The annotated frame.
        """
        annotated_frame = frame.copy()

        if detections.tracker_id is None:
            return annotated_frame

        for box, track_id in zip(detections.xyxy, detections.tracker_id):   
            tracker_id = int(track_id)
            
            # Ask the brain what color and text to use
            color, text = state_manager.get_ui_data(tracker_id)
            label = f"ID #{tracker_id} | {text}"

            # Draw Bounding Box
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
            
            # Draw Text Background
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated_frame, (x1, y1 - 25), (x1 + tw, y1), color, -1)
            
            # Draw Text
            cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
        return annotated_frame