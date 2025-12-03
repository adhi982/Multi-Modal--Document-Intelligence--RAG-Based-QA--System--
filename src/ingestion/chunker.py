"""Intelligent chunking for multi-modal content."""

from typing import Dict, List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken

from ..utils import get_logger, compute_hash

logger = get_logger(__name__)


class SmartChunker:
    """Smart chunking for text, tables, and images."""
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        table_max_size: int = 800,
        image_context_window: int = 100
    ):
        """
        Initialize chunker.
        
        Args:
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks in tokens
            table_max_size: Maximum table size in tokens
            image_context_window: Context window around images in tokens
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.table_max_size = table_max_size
        self.image_context_window = image_context_window
        self.logger = logger
        
        # Initialize tokenizer
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except:
            self.logger.warning("Could not load tiktoken, using character-based estimation")
            self.tokenizer = None
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size * 4,  # Approximate chars per token
            chunk_overlap=chunk_overlap * 4,
            length_function=self._count_tokens,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def _count_tokens(self, text: str) -> int:
        """
        Count tokens in text.
        
        Args:
            text: Input text
            
        Returns:
            Number of tokens
        """
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Fallback: estimate 1 token = 4 characters
            return len(text) // 4
    
    def chunk_document(self, doc_data: Dict) -> List[Dict]:
        """
        Chunk entire document with all modalities.
        
        Args:
            doc_data: Document data from parser
            
        Returns:
            List of chunks with metadata
        """
        all_chunks = []
        doc_id = doc_data["doc_id"]
        
        self.logger.info(f"Chunking document: {doc_id}")
        
        for page_data in doc_data["pages"]:
            page_num = page_data["page_num"]
            
            # Extract text blocks
            page_text = "\n\n".join([
                block["text"] for block in page_data.get("text_blocks", [])
            ])
            
            # Chunk text
            if page_text.strip():
                text_chunks = self._chunk_text(page_text, doc_id, page_num)
                all_chunks.extend(text_chunks)
            
            # Process images (will be added separately with context)
            images = page_data.get("images", [])
            if images:
                image_chunks = self._chunk_images(images, page_text, doc_id, page_num)
                all_chunks.extend(image_chunks)
        
        self.logger.info(f"Created {len(all_chunks)} chunks for {doc_id}")
        return all_chunks
    
    def _chunk_text(self, text: str, doc_id: str, page_num: int) -> List[Dict]:
        """
        Chunk text content.
        
        Args:
            text: Page text
            doc_id: Document ID
            page_num: Page number
            
        Returns:
            List of text chunks
        """
        chunks = []
        
        # Split text using LangChain splitter
        text_splits = self.text_splitter.split_text(text)
        
        for idx, chunk_text in enumerate(text_splits):
            token_count = self._count_tokens(chunk_text)
            
            chunk = {
                "chunk_id": f"{doc_id}_p{page_num}_c{idx}",
                "doc_id": doc_id,
                "page": page_num,
                "chunk_type": "text",
                "content_text": chunk_text,
                "token_count": token_count,
                "chunk_index": idx,
                "metadata": {
                    "hash": compute_hash(chunk_text)
                }
            }
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_images(
        self,
        images: List[Dict],
        page_text: str,
        doc_id: str,
        page_num: int
    ) -> List[Dict]:
        """
        Create chunks for images with context.
        
        Args:
            images: List of images from page
            page_text: Full page text for context
            doc_id: Document ID
            page_num: Page number
            
        Returns:
            List of image chunks
        """
        chunks = []
        
        # Get context tokens (split page text into words for context extraction)
        context_tokens = page_text.split()
        half_window = self.image_context_window // 2
        
        for idx, image in enumerate(images):
            # Extract context around image position (simplified)
            # In a full implementation, would use bbox to find surrounding text
            start_idx = max(0, len(context_tokens) // 2 - half_window)
            end_idx = min(len(context_tokens), len(context_tokens) // 2 + half_window)
            context_text = " ".join(context_tokens[start_idx:end_idx])
            
            chunk = {
                "chunk_id": f"{doc_id}_p{page_num}_img{idx}",
                "doc_id": doc_id,
                "page": page_num,
                "chunk_type": "image",
                "content_text": context_text,  # Context for text embedding
                "image_data": {
                    "image_bytes": image.get("image_bytes"),
                    "format": image.get("format"),
                    "size": image.get("size"),
                    "bbox": image.get("bbox")
                },
                "token_count": self._count_tokens(context_text),
                "chunk_index": idx,
                "metadata": {
                    "has_image": True
                }
            }
            chunks.append(chunk)
        
        return chunks
    
    def chunk_tables(self, tables: List[Dict]) -> List[Dict]:
        """
        Chunk tables (keep whole table if possible).
        
        Args:
            tables: List of tables from extractor
            
        Returns:
            List of table chunks
        """
        chunks = []
        
        for table in tables:
            # Use plain_text for better retrieval, markdown for display
            content_for_embedding = table.get("plain_text", table["markdown"])
            token_count = table.get("token_count", 0)
            
            # If table is small enough, keep as single chunk
            if token_count <= self.table_max_size:
                chunk = {
                    "chunk_id": f"{table['doc_id']}_p{table['page']}_t{table['table_idx']}",
                    "doc_id": table["doc_id"],
                    "page": table["page"],
                    "chunk_type": "table",
                    "content_text": content_for_embedding,
                    "table_data": {
                        "markdown": table["markdown"],
                        "plain_text": content_for_embedding,
                        "csv": table["csv"],
                        "dataframe": table["dataframe"],
                        "shape": table["shape"]
                    },
                    "token_count": token_count,
                    "chunk_index": 0,
                    "metadata": {
                        "is_table": True,
                        "table_preview": content_for_embedding[:200]  # First 200 chars as preview
                    }
                }
                chunks.append(chunk)
            
            else:
                # Split large table by rows (preserve headers)
                # For now, just truncate - full implementation would split intelligently
                truncated_text = content_for_embedding[:self.table_max_size * 4]
                
                chunk = {
                    "chunk_id": f"{table['doc_id']}_p{table['page']}_t{table['table_idx']}",
                    "doc_id": table["doc_id"],
                    "page": table["page"],
                    "chunk_type": "table",
                    "content_text": truncated_text,
                    "table_data": {
                        "markdown": table["markdown"][:self.table_max_size * 4],
                        "plain_text": truncated_text,
                        "shape": table["shape"]
                    },
                    "token_count": self._count_tokens(truncated_text),
                    "chunk_index": 0,
                    "metadata": {
                        "is_table": True,
                        "truncated": True
                    }
                }
                chunks.append(chunk)
        
        return chunks
    
    def add_ocr_to_image_chunks(
        self,
        chunks: List[Dict],
        ocr_results: Dict[str, Dict]
    ) -> List[Dict]:
        """
        Add OCR text to image chunks.
        
        Args:
            chunks: List of chunks
            ocr_results: Dictionary mapping chunk_id to OCR results
            
        Returns:
            Updated chunks with OCR text
        """
        for chunk in chunks:
            if chunk["chunk_type"] == "image":
                chunk_id = chunk["chunk_id"]
                if chunk_id in ocr_results:
                    ocr_data = ocr_results[chunk_id]
                    
                    # Append OCR text to content
                    ocr_text = ocr_data.get("text", "")
                    if ocr_text:
                        chunk["content_text"] = f"{chunk['content_text']}\n\nOCR: {ocr_text}"
                        chunk["token_count"] = self._count_tokens(chunk["content_text"])
                    
                    # Store OCR metadata
                    chunk["metadata"]["ocr"] = {
                        "text": ocr_text,
                        "confidence": ocr_data.get("avg_confidence", 0.0),
                        "num_detections": ocr_data.get("num_detections", 0)
                    }
        
        return chunks
