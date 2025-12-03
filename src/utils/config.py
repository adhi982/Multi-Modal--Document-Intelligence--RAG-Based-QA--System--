"""Configuration loader and management."""

import yaml
from pathlib import Path
from typing import Any, Dict


class Config:
    """Configuration manager for the application."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize configuration from YAML file.
        
        Args:
            config_path: Path to the configuration YAML file
        """
        self.config_path = Path(config_path)
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key (supports nested keys with dot notation).
        
        Args:
            key: Configuration key (e.g., "models.clip_model")
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    @property
    def clip_model(self) -> str:
        """Get CLIP model name."""
        return self.get("models.clip_model")
    
    @property
    def generation_model(self) -> str:
        """Get generation model name."""
        return self.get("models.generation_model")
    
    @property
    def text_chunk_size(self) -> int:
        """Get text chunk size."""
        return self.get("chunking.text_chunk_size", 512)
    
    @property
    def text_overlap(self) -> int:
        """Get text overlap size."""
        return self.get("chunking.text_overlap", 50)
    
    @property
    def embedding_dimension(self) -> int:
        """Get embedding dimension."""
        return self.get("embeddings.dimension", 512)
    
    @property
    def rrf_k(self) -> int:
        """Get RRF constant."""
        return self.get("retrieval.rrf_k", 60)
    
    @property
    def final_top_k(self) -> int:
        """Get final top-k results."""
        return self.get("retrieval.final_top_k", 10)
    
    @property
    def raw_data_path(self) -> Path:
        """Get raw data directory path."""
        return Path(self.get("paths.raw_data", "data/raw/"))
    
    @property
    def processed_data_path(self) -> Path:
        """Get processed data directory path."""
        return Path(self.get("paths.processed_data", "data/processed/"))
    
    @property
    def faiss_index_path(self) -> Path:
        """Get FAISS index path."""
        return Path(self.get("paths.faiss_index", "data/indices/faiss.index"))
    
    @property
    def bm25_index_path(self) -> Path:
        """Get BM25 index path."""
        return Path(self.get("paths.bm25_index", "data/indices/bm25.pkl"))
    
    @property
    def metadata_db_path(self) -> Path:
        """Get metadata database path."""
        return Path(self.get("paths.metadata_db", "data/metadata/chunks.db"))


# Global config instance
_config_instance = None


def get_config(config_path: str = "config.yaml") -> Config:
    """
    Get global configuration instance (singleton).
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Config instance
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = Config(config_path)
    return _config_instance
