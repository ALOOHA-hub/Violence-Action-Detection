import supervision as sv
import cv2

class Visualizer:
    def __init__(self):
        # Create annotators for Boxes and Labels
        self.box_annotator = sv.BoxAnnotator(
            thickness=2
        )
        self.label_annotator = sv.LabelAnnotator(
            text_scale=0.5,
            text_thickness=1,
            text_padding=5
        )

    def draw(self, frame, detections):
        """
        Draws bounding boxes and Tracker IDs on the frame.
        """
        # Generate labels: "#ID ClassName Conf"
        labels = [
            f"#{tracker_id} Person {conf:.2f}"
            for tracker_id, conf in zip(detections.tracker_id, detections.confidence)
        ]

        # Draw on a copy of the frame
        annotated_frame = frame.copy()
        annotated_frame = self.box_annotator.annotate(
            scene=annotated_frame, detections=detections
        )
        annotated_frame = self.label_annotator.annotate(
            scene=annotated_frame, detections=detections, labels=labels
        )
        
        return annotated_frame