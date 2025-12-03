"""FAISS vector index management."""

import faiss
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from ..utils import get_logger, normalize_embeddings

logger = get_logger(__name__)


class FAISSIndex:
    """FAISS index manager for vector search."""
    
    def __init__(self, dimension: int = 512, index_type: str = "flat"):
        """
        Initialize FAISS index.
        
        Args:
            dimension: Embedding dimension
            index_type: Type of index ('flat', 'ivf')
        """
        self.dimension = dimension
        self.index_type = index_type
        self.logger = logger
        self._index = None
        self._num_vectors = 0
    
    def create_index(self, index_type: Optional[str] = None):
        """
        Create a new FAISS index.
        
        Args:
            index_type: Type of index to create
        """
        if index_type:
            self.index_type = index_type
        
        self.logger.info(f"Creating FAISS index: {self.index_type}, dim={self.dimension}")
        
        if self.index_type == "flat":
            # Flat index with inner product (cosine similarity with normalized vectors)
            self._index = faiss.IndexFlatIP(self.dimension)
        
        elif self.index_type == "ivf":
            # IVF index for larger datasets
            quantizer = faiss.IndexFlatIP(self.dimension)
            nlist = 100  # Number of clusters
            self._index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
        
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
        
        self._num_vectors = 0
    
    def add_vectors(
        self,
        embeddings: np.ndarray,
        normalize: bool = True
    ):
        """
        Add vectors to the index.
        
        Args:
            embeddings: Array of embeddings, shape (n, d)
            normalize: Whether to normalize embeddings
        """
        if self._index is None:
            self.create_index()
        
        # Ensure float32
        embeddings = embeddings.astype('float32')
        
        # Normalize if requested
        if normalize:
            embeddings = normalize_embeddings(embeddings)
        
        # Train IVF index if needed
        if self.index_type == "ivf" and not self._index.is_trained:
            self.logger.info("Training IVF index...")
            self._index.train(embeddings)
        
        # Add to index
        self._index.add(embeddings)
        self._num_vectors += len(embeddings)
        
        self.logger.info(f"Added {len(embeddings)} vectors to index. Total: {self._num_vectors}")
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        normalize: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for similar vectors.
        
        Args:
            query_embedding: Query embedding, shape (1, d) or (d,)
            k: Number of results to return
            normalize: Whether to normalize query
            
        Returns:
            Tuple of (distances, indices)
        """
        if self._index is None:
            raise ValueError("Index not created. Call create_index() or load().")
        
        # Reshape if needed
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Ensure float32
        query_embedding = query_embedding.astype('float32')
        
        # Normalize if requested
        if normalize:
            query_embedding = normalize_embeddings(query_embedding)
        
        # Search
        distances, indices = self._index.search(query_embedding, k)
        
        return distances[0], indices[0]
    
    def save(self, filepath: str):
        """
        Save index to disk.
        
        Args:
            filepath: Path to save index
        """
        if self._index is None:
            raise ValueError("No index to save")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        faiss.write_index(self._index, str(filepath))
        self.logger.info(f"Saved FAISS index to {filepath}")
    
    def load(self, filepath: str):
        """
        Load index from disk.
        
        Args:
            filepath: Path to load index from
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Index file not found: {filepath}")
        
        self._index = faiss.read_index(str(filepath))
        self._num_vectors = self._index.ntotal
        
        self.logger.info(f"Loaded FAISS index from {filepath}. Vectors: {self._num_vectors}")
    
    @property
    def num_vectors(self) -> int:
        """Get number of vectors in index."""
        return self._num_vectors
    
    @property
    def is_trained(self) -> bool:
        """Check if index is trained (for IVF)."""
        if self._index is None:
            return False
        return self._index.is_trained if hasattr(self._index, 'is_trained') else True
