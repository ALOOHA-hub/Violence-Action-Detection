import supervision as sv
import cv2

class Visualizer:
    def __init__(self):
        # Create annotators for Boxes and Labels
        self.box_annotator = sv.BoxAnnotator(
            thickness=3
        )
        self.label_annotator = sv.LabelAnnotator(
            text_scale=1.5,
            text_thickness=2,
            text_padding=10,
            text_position=sv.Position.TOP_LEFT
        )

    def draw(self, frame, detections):
        """
        Draws bounding boxes and Tracker IDs on the frame.
        """
        if detections.tracker_id is None:
            return frame

        labels = []
        for track_id, conf in zip(detections.tracker_id, detections.confidence):   
            tracker_id = int(track_id)
            label_text = f"ID: #{tracker_id}"

            if actions and tracker_id in actions:
                label_text += f" | {actions[tracker_id]}"

            labels.append(label_text)

        annotated_frame = frame.copy()
        annotated_frame = self.box_annotator.annotate(scene=annotated_frame, detections=detections)
        annotated_frame = self.label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        
        return annotated_frame