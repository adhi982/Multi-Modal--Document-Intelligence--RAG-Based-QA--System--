"""PDF parsing using PyMuPDF."""

import fitz  # PyMuPDF
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image
import io

from ..utils import get_logger

logger = get_logger(__name__)


class PDFParser:
    """Parse PDF documents to extract text, images, and metadata."""
    
    def __init__(self):
        """Initialize PDF parser."""
        self.logger = logger
    
    def parse(self, pdf_path: str) -> Dict:
        """
        Parse PDF file and extract all content.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary containing document info and pages
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        self.logger.info(f"Parsing PDF: {pdf_path.name}")
        
        doc = None
        try:
            doc = fitz.open(str(pdf_path))
            
            num_pages = len(doc)
            doc_info = {
                "doc_id": pdf_path.stem,
                "filename": pdf_path.name,
                "num_pages": num_pages,
                "metadata": doc.metadata,
                "pages": []
            }
            
            for page_num in range(num_pages):
                page_data = self._parse_page(doc, page_num)
                doc_info["pages"].append(page_data)
            
            self.logger.info(f"Successfully parsed {num_pages} pages from {pdf_path.name}")
            return doc_info
            
        except Exception as e:
            self.logger.error(f"Error parsing PDF {pdf_path}: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
        finally:
            if doc is not None:
                doc.close()
    
    def _parse_page(self, doc: fitz.Document, page_num: int) -> Dict:
        """
        Parse a single page from PDF.
        
        Args:
            doc: PyMuPDF document object
            page_num: Page number (0-indexed)
            
        Returns:
            Dictionary containing page content
        """
        page = doc[page_num]
        
        page_data = {
            "page_num": page_num + 1,  # 1-indexed for user display
            "text_blocks": self._extract_text_blocks(page),
            "images": self._extract_images(page, doc, page_num),
            "size": {
                "width": page.rect.width,
                "height": page.rect.height
            }
        }
        
        return page_data
    
    def _extract_text_blocks(self, page: fitz.Page) -> List[Dict]:
        """
        Extract text blocks with bounding boxes.
        
        Args:
            page: PyMuPDF page object
            
        Returns:
            List of text blocks with metadata
        """
        blocks = []
        
        text_dict = page.get_text("dict")
        
        for block in text_dict.get("blocks", []):
            if block.get("type") == 0:  # Text block
                block_text = ""
                bbox = block.get("bbox", [0, 0, 0, 0])
                
                # Extract text from lines
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        block_text += span.get("text", "")
                    block_text += "\n"
                
                if block_text.strip():
                    blocks.append({
                        "text": block_text.strip(),
                        "bbox": list(bbox),
                        "type": "text"
                    })
        
        return blocks
    
    def _extract_images(
        self,
        page: fitz.Page,
        doc: fitz.Document,
        page_num: int
    ) -> List[Dict]:
        """
        Extract images from page.
        
        Args:
            page: PyMuPDF page object
            doc: PyMuPDF document object
            page_num: Page number
            
        Returns:
            List of images with metadata
        """
        images = []
        
        image_list = page.get_images(full=True)
        
        for img_index, img_info in enumerate(image_list):
            try:
                xref = img_info[0]
                base_image = doc.extract_image(xref)
                
                if base_image:
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    
                    # Get image bounding box
                    img_rects = page.get_image_rects(xref)
                    bbox = list(img_rects[0]) if img_rects else [0, 0, 0, 0]
                    
                    # Convert to PIL Image
                    image = Image.open(io.BytesIO(image_bytes))
                    
                    images.append({
                        "image_bytes": image_bytes,
                        "image": image,
                        "bbox": bbox,
                        "format": image_ext,
                        "size": image.size,
                        "xref": xref,
                        "type": "image"
                    })
            
            except Exception as e:
                self.logger.warning(f"Error extracting image {img_index} from page {page_num}: {str(e)}")
                continue
        
        return images
    
    def extract_text_only(self, pdf_path: str) -> str:
        """
        Extract plain text from PDF (fast method).
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Concatenated text from all pages
        """
        doc = fitz.open(str(pdf_path))
        text = ""
        
        for page in doc:
            text += page.get_text() + "\n\n"
        
        doc.close()
        return text.strip()
