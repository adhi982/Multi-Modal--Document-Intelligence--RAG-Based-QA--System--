"""Hybrid retrieval combining BM25 and FAISS with RRF fusion."""

from typing import List, Dict, Tuple
import numpy as np

from ..embeddings import CLIPEmbedder
from ..indexing import FAISSIndex, BM25Index, MetadataStore
from ..utils import get_logger

logger = get_logger(__name__)


class HybridRetriever:
    """Hybrid retrieval using BM25 + FAISS with RRF fusion."""
    
    def __init__(
        self,
        faiss_index: FAISSIndex,
        bm25_index: BM25Index,
        metadata_store: MetadataStore,
        embedder: CLIPEmbedder,
        rrf_k: int = 60,
        bm25_weight: float = 1.5  # Boost BM25 for text queries
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            faiss_index: FAISS vector index
            bm25_index: BM25 lexical index
            metadata_store: Metadata store
            embedder: CLIP embedder
            rrf_k: RRF constant (typically 60)
        """
        self.faiss_index = faiss_index
        self.bm25_index = bm25_index
        self.metadata_store = metadata_store
        self.embedder = embedder
        self.rrf_k = rrf_k
        self.bm25_weight = bm25_weight
        self.logger = logger
    
    def _expand_query(self, query: str) -> str:
        """Expand query with synonyms for better retrieval."""
        expansions = {
            'assess': ['assessment', 'evaluate', 'review', 'analysis'],
            'macroeconomic': ['economic', 'fiscal', 'monetary'],
            'stability': ['stable', 'resilient', 'outlook', 'performance'],
            'growth': ['expansion', 'GDP', 'output'],
            'inflation': ['prices', 'CPI'],
            'risks': ['challenges', 'vulnerabilities', 'downside']
        }
        
        expanded = query
        for key, synonyms in expansions.items():
            if key.lower() in query.lower():
                # Add first 2 synonyms
                expanded += ' ' + ' '.join(synonyms[:2])
        
        return expanded
    
    def retrieve(
        self,
        query: str,
        bm25_top_k: int = 100,
        faiss_top_k: int = 100,
        final_top_k: int = 50
    ) -> List[Dict]:
        """
        Hybrid retrieval with RRF fusion.
        
        Args:
            query: Query string
            bm25_top_k: Number of results from BM25
            faiss_top_k: Number of results from FAISS
            final_top_k: Final number of results to return
            
        Returns:
            List of retrieved chunks with scores
        """
        self.logger.info(f"Hybrid retrieval for query: {query[:50]}...")
        
        # Expand query for better BM25 matching
        expanded_query = self._expand_query(query)
        self.logger.info(f"Expanded query: {expanded_query[:100]}...")
        
        # BM25 lexical search with expanded query
        bm25_results = self.bm25_index.search(expanded_query, k=bm25_top_k)
        bm25_ranks = {idx: rank + 1 for rank, (idx, _) in enumerate(bm25_results)}
        
        # FAISS semantic search
        query_emb = self.embedder.embed_text(query, normalize=True)
        distances, faiss_indices = self.faiss_index.search(query_emb, k=faiss_top_k)
        faiss_ranks = {int(idx): rank + 1 for rank, idx in enumerate(faiss_indices)}
        
        # RRF fusion with BM25 weight
        fused_scores = self._reciprocal_rank_fusion(
            bm25_ranks,
            faiss_ranks,
            k=self.rrf_k,
            bm25_weight=self.bm25_weight
        )
        
        # Sort by fused score
        sorted_items = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        top_items = sorted_items[:final_top_k]
        
        # Retrieve full chunks from metadata store
        results = []
        for faiss_id, rrf_score in top_items:
            chunk = self.metadata_store.get_chunk_by_faiss_id(faiss_id)
            if chunk:
                chunk["rrf_score"] = float(rrf_score)
                chunk["bm25_rank"] = bm25_ranks.get(faiss_id, None)
                chunk["faiss_rank"] = faiss_ranks.get(faiss_id, None)
                results.append(chunk)
        
        self.logger.info(f"Retrieved {len(results)} results after RRF fusion")
        return results
    
    def _reciprocal_rank_fusion(
        self,
        bm25_ranks: Dict[int, int],
        faiss_ranks: Dict[int, int],
        k: int = 60,
        bm25_weight: float = 1.0
    ) -> Dict[int, float]:
        """
        Compute weighted RRF scores.
        
        Formula: RRF_score(d) = bm25_weight * (1/(k + bm25_rank)) + 1/(k + faiss_rank)
        
        Args:
            bm25_ranks: Dictionary mapping doc_id to BM25 rank
            faiss_ranks: Dictionary mapping doc_id to FAISS rank
            k: RRF constant
            bm25_weight: Weight multiplier for BM25 scores (>1 boosts BM25)
            
        Returns:
            Dictionary mapping doc_id to RRF score
        """
        rrf_scores = {}
        
        # Get all document IDs from both rankings
        all_doc_ids = set(bm25_ranks.keys()) | set(faiss_ranks.keys())
        
        for doc_id in all_doc_ids:
            score = 0.0
            
            # Add weighted BM25 contribution
            if doc_id in bm25_ranks:
                score += bm25_weight * (1.0 / (k + bm25_ranks[doc_id]))
            
            # Add FAISS contribution
            if doc_id in faiss_ranks:
                score += 1.0 / (k + faiss_ranks[doc_id])
            
            rrf_scores[doc_id] = score
        
        return rrf_scores
    
    def retrieve_with_scores(
        self,
        query: str,
        bm25_top_k: int = 100,
        faiss_top_k: int = 100,
        final_top_k: int = 50
    ) -> Tuple[List[Dict], Dict]:
        """
        Retrieve with detailed scoring information.
        
        Args:
            query: Query string
            bm25_top_k: Number of results from BM25
            faiss_top_k: Number of results from FAISS
            final_top_k: Final number of results
            
        Returns:
            Tuple of (results, score_info)
        """
        # Get retrieval results
        results = self.retrieve(query, bm25_top_k, faiss_top_k, final_top_k)
        
        # Compute scoring info
        bm25_only = sum(1 for r in results if r.get("faiss_rank") is None)
        faiss_only = sum(1 for r in results if r.get("bm25_rank") is None)
        both = sum(1 for r in results if r.get("bm25_rank") and r.get("faiss_rank"))
        
        score_info = {
            "bm25_only": bm25_only,
            "faiss_only": faiss_only,
            "both": both,
            "total": len(results)
        }
        
        return results, score_info
