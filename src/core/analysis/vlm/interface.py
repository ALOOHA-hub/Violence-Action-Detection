from abc import ABC, abstractmethod

class IVisionReasoner(ABC):
    @abstractmethod
    def analyze_incident(self, video_path: str) -> dict:
        """Analyze video footage and return a structured JSON threat report."""
        pass