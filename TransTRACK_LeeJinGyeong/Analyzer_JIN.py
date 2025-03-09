import logging
from abc import ABC, abstractmethod

class DrowsinessAnalyzer(ABC):
    """Abstract base class for drowsiness analysis implementations."""
    
    @abstractmethod
    def analyze(self, detected, total_frames):
        """
        Analyze drowsiness based on detection metrics.
        
        Args:
            detected (int) : Number of frames with issues
            total_frames (int): Total number of frames processed
            
        Returns:
            dict: Analysis results containing at least:
                - is_drowsy (bool): Whether drowsiness was detected
                - confidence (float): Confidence score of the detection (0-1)
                - details (dict): Additional detection details
        """
        pass

class ThresholdBasedAnalyzer(DrowsinessAnalyzer):
    """Simple threshold-based drowsiness analysis."""
    
    def __init__(self, threshold):
        self.threshold = threshold
        
    def analyze(self, yawn_count, eye_closed_frames, total_frames):
        logging.info(f"Analyzing drowsiness: Yawns={yawn_count}, Eye Closed Frames={eye_closed_frames}, Total Frames={total_frames}")
        
        # Calculate basic metrics
        eye_closed_ratio = eye_closed_frames / total_frames if total_frames > 0 else 0
        yawn_rate = yawn_count / (total_frames / 30) if total_frames > 0 else 0  # Assuming 30 fps
        
        # Check thresholds
        drowsiness = yawn_count + eye_closed_frames > self.threshold
        
        # Calculate confidence score (0-1)
        yawn_confidence = min(yawn_count / (self.threshold * 2), 1.0)
        eye_confidence = min(eye_closed_frames / (self.threshold * 2), 1.0)
        confidence = max(yawn_confidence, eye_confidence)
        
        return {
            'is_drowsy': drowsiness,
            'confidence': confidence,
            'details': {
                'yawn_count': yawn_count,
                'eye_closed_frames': eye_closed_frames,
                'eye_closed_ratio': eye_closed_ratio,
                'yawn_rate': yawn_rate,
                # 'threshold_exceeded': is_drowsy_yawn,
                # 'eye_threshold_exceeded': is_drowsy_eyes
            }
        }

# Factory function to create analyzers
def create_analyzer(analyzer_type="threshold", **kwargs):
    """
    Create and return a drowsiness analyzer instance.
    
    Args:
        analyzer_type (str): Type of analyzer to create
        **kwargs: Configuration parameters for the analyzer
        
    Returns:
        DrowsinessAnalyzer: An instance of the requested analyzer
    """
    if analyzer_type == "threshold":
        threshold = kwargs.get('threshold', 4)
        return ThresholdBasedAnalyzer(threshold)
    else:
        raise ValueError(f"Unknown analyzer type: {analyzer_type}")