"""Multi-document summarization and briefing."""

from typing import List, Dict
from sklearn.cluster import KMeans
import numpy as np

from ..utils import get_logger
from .prompts import create_summarization_prompt, create_briefing_prompt


logger = get_logger(__name__)


class Summarizer:
    """Multi-document summarizer with clustering."""
    
    def __init__(self, generator):
        """
        Initialize summarizer.
        
        Args:
            generator: Generator instance
        """
        self.generator = generator
        self.logger = logger
    
    def summarize(
        self,
        chunks: List[Dict],
        max_length: int = 150
    ) -> str:
        """
        Summarize multiple chunks.
        
        Args:
            chunks: List of chunks to summarize
            max_length: Maximum summary length in tokens
            
        Returns:
            Summary text
        """
        if not chunks:
            return ""
        
        # Extract text passages
        passages = [chunk.get("content_text", "") for chunk in chunks]
        
        # Truncate passages to avoid context overflow
        truncated_passages = [p[:500] for p in passages]
        
        # Create prompt
        prompt = create_summarization_prompt(truncated_passages)
        
        # Generate summary
        summary = self.generator.generate(prompt, max_new_tokens=max_length)
        
        return summary
    
    def generate_briefing(
        self,
        query: str,
        chunks: List[Dict],
        n_clusters: int = 3
    ) -> Dict:
        """
        Generate multi-document briefing with comprehensive summary.
        
        Args:
            query: Original query
            chunks: Retrieved chunks
            n_clusters: Number of topic clusters
            
        Returns:
            Dictionary with briefing and metadata
        """
        self.logger.info(f"Generating briefing for {len(chunks)} chunks")
        
        if not chunks:
            return {
                "briefing": "No content available for briefing.",
                "topics": []
            }
        
        # Create comprehensive summary of all content
        all_text = "\n\n".join([
            chunk.get("content_text", "")[:800]  # Increased from 500 to get more context
            for chunk in chunks[:15]  # Increased from 10 to 15 chunks
        ])
        
        # Create enhanced briefing prompt
        briefing_prompt = f"""You are a financial analyst writing an executive briefing report.

Topic: {query}

Context Information:
{all_text}

Write a comprehensive executive briefing (minimum 4-5 paragraphs) that:
1. Provides a detailed overview of the topic with specific examples and data
2. Explains the key risks, challenges, or factors involved
3. Discusses implications and potential impacts
4. Includes relevant numbers, percentages, and timeframes from the context
5. Uses professional but accessible language
6. Organizes information into coherent paragraphs with clear topic sentences
7. Does NOT include page numbers or citations within the text

Write the briefing:"""
        
        # Generate main summary with more tokens
        main_summary = self.generator.generate(briefing_prompt, max_new_tokens=600, temperature=0.3)
        
        # Clean up the summary
        import re
        main_summary = re.sub(r'\[[\w\-]+_[\w_]+,\s*page\s*\d+\]', '', main_summary)
        main_summary = re.sub(r'\[page\s*\d+\]', '', main_summary)
        main_summary = main_summary.strip()
        
        # Extract sources for citation at the end
        sources = self._extract_sources(chunks[:10])
        sources_text = ", ".join(sources)
        
        # Format as natural briefing
        briefing = f"{main_summary}\n\n**Sources:** {sources_text}"
        
        return {
            "briefing": briefing,
            "topics": [{
                "topic_id": 1,
                "summary": main_summary,
                "sources": sources
            }],
            "num_clusters": 1
        }
    
    def _cluster_chunks(
        self,
        chunks: List[Dict],
        n_clusters: int
    ) -> List[List[Dict]]:
        """
        Cluster chunks by semantic similarity.
        
        Args:
            chunks: List of chunks
            n_clusters: Number of clusters
            
        Returns:
            List of cluster lists
        """
        try:
            # Extract embeddings (if available in chunks)
            # For simplicity, we'll use a basic approach
            # In production, use actual embeddings
            
            # Simple clustering by page number as fallback
            pages = [chunk.get("page", 0) for chunk in chunks]
            
            # Create pseudo-embeddings from page numbers
            embeddings = np.array([[p] for p in pages])
            
            # KMeans clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings)
            
            # Group chunks by cluster
            clusters = [[] for _ in range(n_clusters)]
            for chunk, label in zip(chunks, labels):
                clusters[label].append(chunk)
            
            # Remove empty clusters
            clusters = [c for c in clusters if c]
            
            return clusters
        
        except Exception as e:
            self.logger.warning(f"Clustering failed: {str(e)}. Using single cluster.")
            return [chunks]
    
    def _extract_sources(self, chunks: List[Dict]) -> List[str]:
        """
        Extract unique sources from chunks.
        
        Args:
            chunks: List of chunks
            
        Returns:
            List of source strings
        """
        sources = set()
        
        for chunk in chunks:
            doc_id = chunk.get("doc_id", "unknown")
            page = chunk.get("page", "?")
            sources.add(f"{doc_id} (p.{page})")
        
        return sorted(list(sources))
