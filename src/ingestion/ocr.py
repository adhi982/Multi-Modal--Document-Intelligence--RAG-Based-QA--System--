"""OCR processing using EasyOCR."""

import easyocr
from typing import Dict, List, Optional
from PIL import Image
import numpy as np

from ..utils import get_logger

logger = get_logger(__name__)


class OCRProcessor:
    """Process images with OCR using EasyOCR."""
    
    def __init__(self, languages: List[str] = ['en'], gpu: bool = True):
        """
        Initialize OCR processor.
        
        Args:
            languages: List of language codes (e.g., ['en', 'fr'])
            gpu: Whether to use GPU acceleration
        """
        self.logger = logger
        self.languages = languages
        self.gpu = gpu
        self._reader = None
    
    @property
    def reader(self):
        """Lazy load EasyOCR reader."""
        if self._reader is None:
            self.logger.info(f"Loading EasyOCR with languages: {self.languages}")
            self._reader = easyocr.Reader(self.languages, gpu=self.gpu)
        return self._reader
    
    def process_image(
        self,
        image: Image.Image,
        confidence_threshold: float = 0.5
    ) -> Dict:
        """
        Process image with OCR.
        
        Args:
            image: PIL Image object
            confidence_threshold: Minimum confidence score (0-1)
            
        Returns:
            Dictionary containing OCR results
        """
        try:
            # Convert PIL Image to numpy array
            image_np = np.array(image)
            
            # Run OCR
            results = self.reader.readtext(image_np)
            
            # Filter by confidence
            filtered_results = []
            all_text = []
            
            for bbox, text, confidence in results:
                if confidence >= confidence_threshold:
                    filtered_results.append({
                        "text": text,
                        "bbox": bbox,
                        "confidence": confidence
                    })
                    all_text.append(text)
            
            # Combine all text
            combined_text = " ".join(all_text)
            
            # Calculate average confidence
            avg_confidence = (
                sum(r["confidence"] for r in filtered_results) / len(filtered_results)
                if filtered_results else 0.0
            )
            
            return {
                "text": combined_text,
                "results": filtered_results,
                "avg_confidence": avg_confidence,
                "num_detections": len(filtered_results)
            }
        
        except Exception as e:
            self.logger.error(f"OCR processing error: {str(e)}")
            return {
                "text": "",
                "results": [],
                "avg_confidence": 0.0,
                "num_detections": 0,
                "error": str(e)
            }
    
    def process_image_bytes(
        self,
        image_bytes: bytes,
        confidence_threshold: float = 0.5
    ) -> Dict:
        """
        Process image from bytes with OCR.
        
        Args:
            image_bytes: Image bytes
            confidence_threshold: Minimum confidence score
            
        Returns:
            Dictionary containing OCR results
        """
        import io
        image = Image.open(io.BytesIO(image_bytes))
        return self.process_image(image, confidence_threshold)
    
    def is_text_heavy(self, image: Image.Image, threshold: float = 0.3) -> bool:
        """
        Check if image contains significant text.
        
        Args:
            image: PIL Image object
            threshold: Minimum proportion of text area
            
        Returns:
            True if image is text-heavy
        """
        try:
            image_np = np.array(image)
            results = self.reader.readtext(image_np)
            
            if not results:
                return False
            
            # Calculate text coverage
            image_area = image.width * image.height
            text_area = 0
            
            for bbox, _, confidence in results:
                if confidence >= 0.5:
                    # Calculate bbox area
                    x_coords = [p[0] for p in bbox]
                    y_coords = [p[1] for p in bbox]
                    width = max(x_coords) - min(x_coords)
                    height = max(y_coords) - min(y_coords)
                    text_area += width * height
            
            coverage = text_area / image_area if image_area > 0 else 0
            return coverage >= threshold
        
        except Exception as e:
            self.logger.warning(f"Error checking text coverage: {str(e)}")
            return False
