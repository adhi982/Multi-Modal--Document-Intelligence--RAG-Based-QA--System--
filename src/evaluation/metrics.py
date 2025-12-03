"""Evaluation metrics for retrieval and generation."""

import numpy as np
from typing import List, Dict, Tuple
import time

from ..utils import get_logger

logger = get_logger(__name__)


class RetrievalMetrics:
    """Compute retrieval metrics."""
    
    @staticmethod
    def recall_at_k(retrieved: List[str], ground_truth: List[str], k: int) -> float:
        """
        Compute Recall@k.
        
        Args:
            retrieved: List of retrieved document IDs
            ground_truth: List of relevant document IDs
            k: Cut-off rank
            
        Returns:
            Recall@k score
        """
        if not ground_truth:
            return 0.0
        
        retrieved_k = set(retrieved[:k])
        relevant = set(ground_truth)
        
        return len(retrieved_k & relevant) / len(relevant)
    
    @staticmethod
    def precision_at_k(retrieved: List[str], ground_truth: List[str], k: int) -> float:
        """
        Compute Precision@k.
        
        Args:
            retrieved: List of retrieved document IDs
            ground_truth: List of relevant document IDs
            k: Cut-off rank
            
        Returns:
            Precision@k score
        """
        if k == 0:
            return 0.0
        
        retrieved_k = set(retrieved[:k])
        relevant = set(ground_truth)
        
        return len(retrieved_k & relevant) / k
    
    @staticmethod
    def reciprocal_rank(retrieved: List[str], ground_truth: List[str]) -> float:
        """
        Compute reciprocal rank (RR).
        
        Args:
            retrieved: List of retrieved document IDs
            ground_truth: List of relevant document IDs
            
        Returns:
            Reciprocal rank
        """
        relevant = set(ground_truth)
        
        for i, doc_id in enumerate(retrieved, start=1):
            if doc_id in relevant:
                return 1.0 / i
        
        return 0.0
    
    @staticmethod
    def mean_reciprocal_rank(
        retrieved_lists: List[List[str]],
        ground_truth_lists: List[List[str]]
    ) -> float:
        """
        Compute Mean Reciprocal Rank (MRR).
        
        Args:
            retrieved_lists: List of retrieved document lists
            ground_truth_lists: List of ground truth lists
            
        Returns:
            MRR score
        """
        rr_scores = []
        
        for retrieved, ground_truth in zip(retrieved_lists, ground_truth_lists):
            rr = RetrievalMetrics.reciprocal_rank(retrieved, ground_truth)
            rr_scores.append(rr)
        
        return np.mean(rr_scores) if rr_scores else 0.0
    
    @staticmethod
    def average_precision(retrieved: List[str], ground_truth: List[str]) -> float:
        """
        Compute Average Precision (AP).
        
        Args:
            retrieved: List of retrieved document IDs
            ground_truth: List of relevant document IDs
            
        Returns:
            AP score
        """
        if not ground_truth:
            return 0.0
        
        relevant = set(ground_truth)
        precisions = []
        num_relevant = 0
        
        for i, doc_id in enumerate(retrieved, start=1):
            if doc_id in relevant:
                num_relevant += 1
                precision = num_relevant / i
                precisions.append(precision)
        
        return np.mean(precisions) if precisions else 0.0
    
    @staticmethod
    def compute_all_metrics(
        retrieved: List[str],
        ground_truth: List[str],
        k_values: List[int] = [1, 5, 10, 20]
    ) -> Dict[str, float]:
        """
        Compute all retrieval metrics.
        
        Args:
            retrieved: List of retrieved document IDs
            ground_truth: List of relevant document IDs
            k_values: List of k values for Recall@k and Precision@k
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Recall@k
        for k in k_values:
            metrics[f"recall@{k}"] = RetrievalMetrics.recall_at_k(retrieved, ground_truth, k)
        
        # Precision@k
        for k in k_values:
            metrics[f"precision@{k}"] = RetrievalMetrics.precision_at_k(retrieved, ground_truth, k)
        
        # MRR (single query)
        metrics["reciprocal_rank"] = RetrievalMetrics.reciprocal_rank(retrieved, ground_truth)
        
        # AP
        metrics["average_precision"] = RetrievalMetrics.average_precision(retrieved, ground_truth)
        
        return metrics


class LatencyTracker:
    """Track latency for different components."""
    
    def __init__(self):
        """Initialize latency tracker."""
        self.timings = {}
        self._start_times = {}
    
    def start(self, component: str):
        """
        Start timing a component.
        
        Args:
            component: Component name
        """
        self._start_times[component] = time.perf_counter()
    
    def stop(self, component: str):
        """
        Stop timing a component.
        
        Args:
            component: Component name
        """
        if component in self._start_times:
            elapsed = time.perf_counter() - self._start_times[component]
            
            if component not in self.timings:
                self.timings[component] = []
            
            self.timings[component].append(elapsed * 1000)  # Convert to ms
            del self._start_times[component]
    
    def get_stats(self, component: str) -> Dict[str, float]:
        """
        Get timing statistics for a component.
        
        Args:
            component: Component name
            
        Returns:
            Dictionary with statistics
        """
        if component not in self.timings or not self.timings[component]:
            return {}
        
        times = self.timings[component]
        
        return {
            "mean": np.mean(times),
            "median": np.median(times),
            "min": np.min(times),
            "max": np.max(times),
            "p95": np.percentile(times, 95),
            "p99": np.percentile(times, 99),
            "count": len(times)
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistics for all components.
        
        Returns:
            Dictionary of statistics per component
        """
        return {
            component: self.get_stats(component)
            for component in self.timings.keys()
        }
    
    def reset(self):
        """Reset all timings."""
        self.timings = {}
        self._start_times = {}


class FaithfulnessChecker:
    """Check if generated answer is faithful to context."""
    
    @staticmethod
    def compute_similarity(answer: str, context: str) -> float:
        """
        Compute semantic similarity between answer and context.
        
        Args:
            answer: Generated answer
            context: Retrieved context
            
        Returns:
            Similarity score (0-1)
        """
        # Simple word overlap for now
        # In production, use sentence-transformers
        answer_words = set(answer.lower().split())
        context_words = set(context.lower().split())
        
        if not answer_words:
            return 0.0
        
        overlap = len(answer_words & context_words)
        return overlap / len(answer_words)
    
    @staticmethod
    def check_faithfulness(
        answer: str,
        chunks: List[Dict],
        threshold: float = 0.5
    ) -> Tuple[bool, float]:
        """
        Check if answer is faithful to context.
        
        Args:
            answer: Generated answer
            chunks: Retrieved context chunks
            threshold: Minimum similarity threshold
            
        Returns:
            Tuple of (is_faithful, similarity_score)
        """
        if not chunks:
            return False, 0.0
        
        # Combine all context
        context = " ".join([chunk.get("content_text", "") for chunk in chunks])
        
        # Compute similarity
        similarity = FaithfulnessChecker.compute_similarity(answer, context)
        
        is_faithful = similarity >= threshold
        
        return is_faithful, similarity
