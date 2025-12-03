"""CLI script to build FAISS and BM25 indices."""

import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from tqdm import tqdm

from src.utils import get_config, setup_logging, load_json
from src.embeddings import CLIPEmbedder
from src.indexing import FAISSIndex, BM25Index, MetadataStore

logger = setup_logging(level="INFO")


def build_indices(chunks_file: str = None):
    """
    Build FAISS and BM25 indices from processed chunks.
    
    Args:
        chunks_file: Path to chunks JSON file
    """
    config = get_config()
    
    # Load chunks
    if chunks_file is None:
        chunks_file = config.processed_data_path / "chunks.json"
    else:
        chunks_file = Path(chunks_file)
    
    if not chunks_file.exists():
        logger.error(f"Chunks file not found: {chunks_file}")
        logger.info("Please run ingest_documents.py first")
        return
    
    logger.info(f"Loading chunks from {chunks_file}")
    chunks = load_json(chunks_file)
    logger.info(f"Loaded {len(chunks)} chunks")
    
    # Initialize components
    logger.info("Initializing embedder...")
    embedder = CLIPEmbedder(
        model_name=config.clip_model,
        batch_size=config.get("embeddings.batch_size", 32)
    )
    
    logger.info("Initializing indices...")
    faiss_index = FAISSIndex(dimension=config.embedding_dimension)
    bm25_index = BM25Index()
    metadata_store = MetadataStore(db_path=str(config.metadata_db_path))
    
    # Embed chunks
    logger.info("Embedding chunks...")
    chunk_embeddings = embedder.embed_chunks(chunks, normalize=True)
    
    # Convert to array
    embeddings_array = np.array([
        chunk_embeddings[chunk["chunk_id"]]
        for chunk in chunks
        if chunk["chunk_id"] in chunk_embeddings
    ])
    
    logger.info(f"Generated {len(embeddings_array)} embeddings")
    
    # Build FAISS index
    logger.info("Building FAISS index...")
    faiss_index.create_index("flat")
    faiss_index.add_vectors(embeddings_array, normalize=False)  # Already normalized
    
    # Save FAISS index
    faiss_index.save(str(config.faiss_index_path))
    logger.info(f"Saved FAISS index to {config.faiss_index_path}")
    
    # Build BM25 index
    logger.info("Building BM25 index...")
    bm25_index.build_index(chunks)
    bm25_index.save(str(config.bm25_index_path))
    logger.info(f"Saved BM25 index to {config.bm25_index_path}")
    
    # Add chunks to metadata store
    logger.info("Populating metadata store...")
    metadata_store.add_chunks_batch(chunks, start_index=0)
    logger.info(f"Added {len(chunks)} chunks to metadata store")
    
    metadata_store.close()
    
    logger.info("\nIndex building complete!")
    logger.info(f"  - FAISS vectors: {faiss_index.num_vectors}")
    logger.info(f"  - BM25 documents: {bm25_index.num_documents}")
    logger.info(f"  - Metadata chunks: {len(chunks)}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Build FAISS and BM25 indices")
    parser.add_argument(
        "--chunks",
        type=str,
        default=None,
        help="Path to chunks JSON file"
    )
    
    args = parser.parse_args()
    
    build_indices(args.chunks)


if __name__ == "__main__":
    main()
