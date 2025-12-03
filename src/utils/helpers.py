"""Helper utilities and common functions."""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Union
import numpy as np


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


def compute_hash(text: str) -> str:
    """
    Compute MD5 hash of text.
    
    Args:
        text: Input text
        
    Returns:
        Hex digest of hash
    """
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def save_json(data: Union[Dict, List], filepath: Union[str, Path]) -> None:
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        filepath: Output file path
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(filepath: Union[str, Path]) -> Union[Dict, List]:
    """
    Load data from JSON file.
    
    Args:
        filepath: Input file path
        
    Returns:
        Loaded data
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"JSON file not found: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def batch_iterator(items: List[Any], batch_size: int):
    """
    Iterate over items in batches.
    
    Args:
        items: List of items
        batch_size: Size of each batch
        
    Yields:
        Batches of items
    """
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to maximum length.
    
    Args:
        text: Input text
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def format_bbox(bbox: List[float]) -> str:
    """
    Format bounding box coordinates as string.
    
    Args:
        bbox: Bounding box [x0, y0, x1, y1]
        
    Returns:
        Formatted string
    """
    return f"[{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]"


def parse_bbox(bbox_str: str) -> List[float]:
    """
    Parse bounding box string to list of floats.
    
    Args:
        bbox_str: Bounding box string
        
    Returns:
        List of coordinates
    """
    return json.loads(bbox_str)
