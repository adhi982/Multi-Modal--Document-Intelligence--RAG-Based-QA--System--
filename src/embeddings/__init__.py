"""Embedding modules."""

from .clip_embedder import CLIPEmbedder
from .embed_utils import normalize_embeddings, cosine_similarity, batch_embed

__all__ = [
    'CLIPEmbedder',
    'normalize_embeddings',
    'cosine_similarity',
    'batch_embed'
]
