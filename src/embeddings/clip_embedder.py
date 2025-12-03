"""CLIP-based embeddings for text and images."""

import torch
from transformers import CLIPProcessor, CLIPModel
from typing import List, Union, Dict
from PIL import Image
import numpy as np

from ..utils import get_logger, normalize_embeddings, batch_iterator

logger = get_logger(__name__)


class CLIPEmbedder:
    """Unified CLIP embedder for text and images."""
    
    def __init__(
        self,
        model_name: str = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
        device: str = None,
        batch_size: int = 32
    ):
        """
        Initialize CLIP embedder.
        
        Args:
            model_name: HuggingFace model name
            device: Device to use ('cuda', 'cpu', or None for auto)
            batch_size: Batch size for embedding
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.logger = logger
        
        # Auto-detect device with GPU priority
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
                self.logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
            else:
                self.device = "cpu"
                self.logger.warning("No GPU detected, using CPU")
        else:
            self.device = device
        
        self.logger.info(f"Loading CLIP model: {model_name} on {self.device}")
        
        # Load model and processor with timeout handling
        import time
        import os
        from pathlib import Path
        
        # Check if model is cached locally
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        model_cached = False
        if cache_dir.exists():
            # Look for cached model
            for item in cache_dir.iterdir():
                if "laion" in item.name.lower() and "clip" in item.name.lower():
                    model_cached = True
                    self.logger.info(f"Found cached model at {item}")
                    break
        
        max_retries = 3
        retry_delay = 5
        
        for attempt in range(max_retries):
            try:
                # Try local first if cached, otherwise download with timeout
                if model_cached:
                    self.logger.info("Attempting to load from local cache...")
                    self.model = CLIPModel.from_pretrained(
                        model_name,
                        local_files_only=True  # Use cached version
                    )
                else:
                    self.model = CLIPModel.from_pretrained(
                        model_name,
                        timeout=60  # 60 second timeout
                    )
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    self.logger.warning(f"Failed to load model (attempt {attempt + 1}/{max_retries}): {e}")
                    # On first failure, try downloading if we were using cache
                    if model_cached and attempt == 0:
                        self.logger.info("Local cache failed, will try downloading...")
                        model_cached = False
                    self.logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    self.logger.error(f"Failed to load model after {max_retries} attempts")
                    raise
        
        # Use half precision on GPU to save memory
        if self.device == "cuda":
            self.model = self.model.half().to(self.device)
            self.logger.info("Using half precision (FP16) for CLIP on GPU")
        else:
            self.model = self.model.to(self.device)
        
        # Load processor with retry
        for attempt in range(max_retries):
            try:
                if model_cached:
                    self.processor = CLIPProcessor.from_pretrained(
                        model_name,
                        local_files_only=True
                    )
                else:
                    self.processor = CLIPProcessor.from_pretrained(
                        model_name,
                        timeout=60
                    )
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    self.logger.warning(f"Failed to load processor (attempt {attempt + 1}/{max_retries}): {e}")
                    if model_cached and attempt == 0:
                        model_cached = False
                    self.logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    self.logger.error(f"Failed to load processor after {max_retries} attempts")
                    raise
        
        # Set to eval mode
        self.model.eval()
        
        self.logger.info("CLIP model loaded successfully")
    
    def embed_text(
        self,
        texts: Union[str, List[str]],
        normalize: bool = True
    ) -> np.ndarray:
        """
        Embed text using CLIP text encoder.
        
        Args:
            texts: Single text or list of texts
            normalize: Whether to L2 normalize embeddings
            
        Returns:
            Embeddings array of shape (n, 512)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts:
            return np.array([])
        
        all_embeddings = []
        
        with torch.no_grad():
            for batch in batch_iterator(texts, self.batch_size):
                # Process inputs
                inputs = self.processor(
                    text=batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=77  # CLIP's max text length
                )
                
                # Move to device and convert to half precision if using GPU
                if self.device == "cuda":
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                else:
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get embeddings
                embeddings = self.model.get_text_features(**inputs)
                
                # Convert back to float32 for numpy
                embeddings = embeddings.float().cpu().numpy()
                all_embeddings.append(embeddings)
        
        # Concatenate all batches
        embeddings = np.concatenate(all_embeddings, axis=0)
        
        # Normalize if requested
        if normalize:
            embeddings = normalize_embeddings(embeddings)
        
        return embeddings
    
    def embed_image(
        self,
        images: Union[Image.Image, List[Image.Image]],
        normalize: bool = True
    ) -> np.ndarray:
        """
        Embed images using CLIP image encoder.
        
        Args:
            images: Single PIL Image or list of PIL Images
            normalize: Whether to L2 normalize embeddings
            
        Returns:
            Embeddings array of shape (n, 512)
        """
        if isinstance(images, Image.Image):
            images = [images]
        
        if not images:
            return np.array([])
        
        all_embeddings = []
        
        with torch.no_grad():
            for batch in batch_iterator(images, self.batch_size):
                # Process images
                inputs = self.processor(
                    images=batch,
                    return_tensors="pt"
                )
                
                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get embeddings
                embeddings = self.model.get_image_features(**inputs)
                
                # Convert back to float32 for numpy
                embeddings = embeddings.float().cpu().numpy()
                all_embeddings.append(embeddings)
        
        # Concatenate all batches
        embeddings = np.concatenate(all_embeddings, axis=0)
        
        # Normalize if requested
        if normalize:
            embeddings = normalize_embeddings(embeddings)
        
        return embeddings
    
    def embed_chunks(
        self,
        chunks: List[Dict],
        normalize: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Embed chunks (handles text, tables, and images).
        
        Args:
            chunks: List of chunks with content_text and optional image_data
            normalize: Whether to L2 normalize embeddings
            
        Returns:
            Dictionary mapping chunk_id to embeddings
        """
        embeddings = {}
        
        # Separate chunks by type
        text_chunks = []
        image_chunks = []
        
        for chunk in chunks:
            chunk_id = chunk["chunk_id"]
            chunk_type = chunk["chunk_type"]
            
            if chunk_type == "image" and "image_data" in chunk:
                image_chunks.append((chunk_id, chunk))
            else:
                text_chunks.append((chunk_id, chunk))
        
        # Embed text chunks
        if text_chunks:
            self.logger.info(f"Embedding {len(text_chunks)} text chunks")
            texts = [chunk["content_text"] for _, chunk in text_chunks]
            text_embeddings = self.embed_text(texts, normalize=normalize)
            
            for (chunk_id, _), embedding in zip(text_chunks, text_embeddings):
                embeddings[chunk_id] = embedding
        
        # Embed image chunks
        if image_chunks:
            self.logger.info(f"Embedding {len(image_chunks)} image chunks")
            
            for chunk_id, chunk in image_chunks:
                try:
                    # Get image from chunk
                    image_data = chunk["image_data"]
                    
                    if "image_bytes" in image_data:
                        import io
                        image = Image.open(io.BytesIO(image_data["image_bytes"]))
                    else:
                        # Fallback: embed context text only
                        text_emb = self.embed_text(chunk["content_text"], normalize=normalize)
                        embeddings[chunk_id] = text_emb[0]
                        continue
                    
                    # Embed image
                    image_emb = self.embed_image(image, normalize=normalize)
                    embeddings[chunk_id] = image_emb[0]
                
                except Exception as e:
                    self.logger.warning(f"Error embedding image chunk {chunk_id}: {str(e)}")
                    # Fallback: embed context text
                    text_emb = self.embed_text(chunk["content_text"], normalize=normalize)
                    embeddings[chunk_id] = text_emb[0]
        
        self.logger.info(f"Embedded {len(embeddings)} chunks total")
        return embeddings
    
    def compute_similarity(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int = 10
    ) -> List[tuple]:
        """
        Compute CLIP similarity between query and candidates.
        
        Args:
            query: Query text
            candidates: List of candidate chunks
            top_k: Number of top results to return
            
        Returns:
            List of (chunk, score) tuples sorted by score
        """
        # Embed query
        query_emb = self.embed_text(query, normalize=True)[0]
        
        scored_candidates = []
        
        for candidate in candidates:
            chunk_type = candidate.get("chunk_type")
            
            try:
                # Embed candidate
                if chunk_type == "image" and "image_data" in candidate:
                    import io
                    image = Image.open(io.BytesIO(candidate["image_data"]["image_bytes"]))
                    candidate_emb = self.embed_image(image, normalize=True)[0]
                else:
                    candidate_emb = self.embed_text(candidate["content_text"], normalize=True)[0]
                
                # Compute cosine similarity (dot product of normalized vectors)
                similarity = np.dot(query_emb, candidate_emb)
                scored_candidates.append((candidate, float(similarity)))
            
            except Exception as e:
                self.logger.warning(f"Error computing similarity for chunk {candidate.get('chunk_id')}: {str(e)}")
                continue
        
        # Sort by score descending
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        return scored_candidates[:top_k]
