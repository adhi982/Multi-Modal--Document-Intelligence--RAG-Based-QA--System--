"""Embedding utilities."""

import numpy as np
from typing import List, Dict


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """
    L2 normalize embeddings for cosine similarity.
    
    Args:
        embeddings: Array of embeddings, shape (n, d)
        
    Returns:
        Normalized embeddings
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms == 0, 1, norms)
    return embeddings / norms


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Args:
        a: First vector
        b: Second vector
        
    Returns:
        Cosine similarity score
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def batch_embed(
    items: List[str],
    embed_fn,
    batch_size: int = 32
) -> np.ndarray:
    """
    Embed items in batches.
    
    Args:
        items: List of items to embed
        embed_fn: Embedding function
        batch_size: Batch size
        
    Returns:
        Array of embeddings
    """
    all_embeddings = []
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        embeddings = embed_fn(batch)
        all_embeddings.append(embeddings)
    
    return np.concatenate(all_embeddings, axis=0)
