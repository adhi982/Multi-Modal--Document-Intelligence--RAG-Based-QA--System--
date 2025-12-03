"""Cross-modal reranking using CLIP."""

from typing import List, Dict
import numpy as np

from ..embeddings import CLIPEmbedder
from ..utils import get_logger

logger = get_logger(__name__)


class CLIPReranker:
    """Cross-modal reranker using CLIP similarity."""
    
    def __init__(self, embedder: CLIPEmbedder):
        """
        Initialize reranker.
        
        Args:
            embedder: CLIP embedder instance
        """
        self.embedder = embedder
        self.logger = logger
    
    def rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int = 10
    ) -> List[Dict]:
        """
        Rerank candidates using CLIP similarity.
        
        Args:
            query: Query string
            candidates: List of candidate chunks
            top_k: Number of top results to return
            
        Returns:
            Reranked list of chunks with CLIP scores
        """
        if not candidates:
            return []
        
        self.logger.info(f"Reranking {len(candidates)} candidates")
        
        # Use CLIP's compute_similarity method
        scored_candidates = self.embedder.compute_similarity(
            query,
            candidates,
            top_k=len(candidates)  # Get all scores first
        )
        
        # Add CLIP scores to chunks
        reranked = []
        for chunk, clip_score in scored_candidates[:top_k]:
            chunk_copy = chunk.copy()
            chunk_copy["clip_score"] = float(clip_score)
            reranked.append(chunk_copy)
        
        self.logger.info(f"Reranked to top {len(reranked)} results")
        return reranked
    
    def rerank_with_weights(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int = 10,
        rrf_weight: float = 0.5,
        clip_weight: float = 0.5
    ) -> List[Dict]:
        """
        Rerank with weighted combination of RRF and CLIP scores.
        
        Args:
            query: Query string
            candidates: List of candidates with rrf_score
            top_k: Number of results
            rrf_weight: Weight for RRF score
            clip_weight: Weight for CLIP score
            
        Returns:
            Reranked list with combined scores
        """
        if not candidates:
            return []
        
        self.logger.info(f"Reranking with weights (RRF={rrf_weight}, CLIP={clip_weight})")
        
        # Get CLIP scores
        scored_candidates = self.embedder.compute_similarity(
            query,
            candidates,
            top_k=len(candidates)
        )
        
        # Normalize scores to [0, 1]
        rrf_scores = [c.get("rrf_score", 0.0) for c in candidates]
        clip_scores = [score for _, score in scored_candidates]
        
        # Normalize
        rrf_min, rrf_max = min(rrf_scores), max(rrf_scores)
        rrf_range = rrf_max - rrf_min if rrf_max != rrf_min else 1.0
        
        clip_min, clip_max = min(clip_scores), max(clip_scores)
        clip_range = clip_max - clip_min if clip_max != clip_min else 1.0
        
        # Compute combined scores
        combined = []
        for (chunk, clip_score), rrf_score in zip(scored_candidates, rrf_scores):
            rrf_norm = (rrf_score - rrf_min) / rrf_range
            clip_norm = (clip_score - clip_min) / clip_range
            
            final_score = rrf_weight * rrf_norm + clip_weight * clip_norm
            
            chunk_copy = chunk.copy()
            chunk_copy["clip_score"] = float(clip_score)
            chunk_copy["combined_score"] = float(final_score)
            combined.append((chunk_copy, final_score))
        
        # Sort by combined score
        combined.sort(key=lambda x: x[1], reverse=True)
        
        return [chunk for chunk, _ in combined[:top_k]]


class MultiStageRetriever:
    """Multi-stage retrieval: Hybrid → Rerank."""
    
    def __init__(
        self,
        hybrid_retriever,
        reranker: CLIPReranker
    ):
        """
        Initialize multi-stage retriever.
        
        Args:
            hybrid_retriever: HybridRetriever instance
            reranker: CLIPReranker instance
        """
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.logger = logger
    
    def retrieve(
        self,
        query: str,
        stage1_top_k: int = 50,
        stage2_top_k: int = 10
    ) -> List[Dict]:
        """
        Two-stage retrieval: hybrid → rerank.
        
        Args:
            query: Query string
            stage1_top_k: Results from stage 1 (hybrid)
            stage2_top_k: Final results after reranking
            
        Returns:
            Final reranked results
        """
        self.logger.info("Multi-stage retrieval started")
        
        # Stage 1: Hybrid retrieval
        stage1_results = self.hybrid_retriever.retrieve(
            query,
            final_top_k=stage1_top_k
        )
        
        if not stage1_results:
            self.logger.warning("No results from stage 1")
            return []
        
        # Stage 2: CLIP reranking
        stage2_results = self.reranker.rerank(
            query,
            stage1_results,
            top_k=stage2_top_k
        )
        
        self.logger.info(f"Multi-stage retrieval complete. Final: {len(stage2_results)} results")
        return stage2_results
