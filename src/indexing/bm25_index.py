"""BM25 lexical index using rank-bm25."""

from rank_bm25 import BM25Okapi
from pathlib import Path
from typing import List, Dict, Tuple
import pickle

from ..utils import get_logger

logger = get_logger(__name__)


class BM25Index:
    """BM25 lexical search index."""
    
    def __init__(self):
        """Initialize BM25 index."""
        self.logger = logger
        self._index = None
        self._documents = []
        self._tokenized_corpus = []
    
    def build_index(self, documents: List[Dict]):
        """
        Build BM25 index from documents.
        
        Args:
            documents: List of document dictionaries with 'content_text'
        """
        self.logger.info(f"Building BM25 index for {len(documents)} documents")
        
        self._documents = documents
        
        # Tokenize documents
        self._tokenized_corpus = [
            self._tokenize(doc.get("content_text", ""))
            for doc in documents
        ]
        
        # Build BM25 index
        self._index = BM25Okapi(self._tokenized_corpus)
        
        self.logger.info("BM25 index built successfully")
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text (simple whitespace tokenization).
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        # Simple tokenization - lowercase and split by whitespace
        return text.lower().split()
    
    def search(
        self,
        query: str,
        k: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Search BM25 index.
        
        Args:
            query: Query string
            k: Number of results to return
            
        Returns:
            List of (index, score) tuples
        """
        if self._index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Tokenize query
        query_tokens = self._tokenize(query)
        
        # Get scores for all documents
        scores = self._index.get_scores(query_tokens)
        
        # Get top-k indices
        top_k_indices = scores.argsort()[-k:][::-1]
        
        # Return (index, score) pairs
        results = [(int(idx), float(scores[idx])) for idx in top_k_indices]
        
        return results
    
    def get_top_k(
        self,
        query: str,
        k: int = 10
    ) -> List[Dict]:
        """
        Get top-k documents for query.
        
        Args:
            query: Query string
            k: Number of results
            
        Returns:
            List of documents with scores
        """
        results = self.search(query, k)
        
        top_docs = []
        for idx, score in results:
            if idx < len(self._documents):
                doc = self._documents[idx].copy()
                doc["bm25_score"] = score
                doc["bm25_rank"] = len(top_docs) + 1
                top_docs.append(doc)
        
        return top_docs
    
    def save(self, filepath: str):
        """
        Save BM25 index to disk.
        
        Args:
            filepath: Path to save index
        """
        if self._index is None:
            raise ValueError("No index to save")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "index": self._index,
            "documents": self._documents,
            "tokenized_corpus": self._tokenized_corpus
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        self.logger.info(f"Saved BM25 index to {filepath}")
    
    def load(self, filepath: str):
        """
        Load BM25 index from disk.
        
        Args:
            filepath: Path to load index from
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Index file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self._index = data["index"]
        self._documents = data["documents"]
        self._tokenized_corpus = data["tokenized_corpus"]
        
        self.logger.info(f"Loaded BM25 index from {filepath}. Documents: {len(self._documents)}")
    
    @property
    def num_documents(self) -> int:
        """Get number of documents in index."""
        return len(self._documents)
