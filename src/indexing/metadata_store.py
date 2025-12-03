"""SQLite metadata store for chunks."""

import sqlite3
from pathlib import Path
from typing import List, Dict, Optional
import json
import threading

from ..utils import get_logger

logger = get_logger(__name__)


class MetadataStore:
    """SQLite-based metadata store for document chunks."""
    
    def __init__(self, db_path: str):
        """
        Initialize metadata store.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger = logger
        self._local = threading.local()
        self._create_tables()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False
            )
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn
    
    def _create_tables(self):
        """Create database tables."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Chunks table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                page INTEGER,
                chunk_type TEXT,
                content_text TEXT,
                bbox TEXT,
                parent_chunk_id TEXT,
                token_count INTEGER,
                faiss_index_id INTEGER,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Indices for faster queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_doc_id ON chunks(doc_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_faiss_id ON chunks(faiss_index_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_chunk_type ON chunks(chunk_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_page ON chunks(page)')
        
        conn.commit()
        self.logger.info(f"Database initialized at {self.db_path}")
    
    def add_chunk(self, chunk: Dict, faiss_index_id: int):
        """
        Add a chunk to the metadata store.
        
        Args:
            chunk: Chunk dictionary
            faiss_index_id: Index in FAISS array
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO chunks (
                chunk_id, doc_id, page, chunk_type, content_text,
                bbox, parent_chunk_id, token_count, faiss_index_id, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            chunk.get("chunk_id"),
            chunk.get("doc_id"),
            chunk.get("page"),
            chunk.get("chunk_type"),
            chunk.get("content_text"),
            json.dumps(chunk.get("bbox")) if chunk.get("bbox") else None,
            chunk.get("parent_chunk_id"),
            chunk.get("token_count"),
            faiss_index_id,
            json.dumps(chunk.get("metadata", {}))
        ))
        
        conn.commit()
    
    def add_chunks_batch(self, chunks: List[Dict], start_index: int = 0):
        """
        Add multiple chunks in batch.
        
        Args:
            chunks: List of chunk dictionaries
            start_index: Starting FAISS index
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        data = []
        for i, chunk in enumerate(chunks):
            data.append((
                chunk.get("chunk_id"),
                chunk.get("doc_id"),
                chunk.get("page"),
                chunk.get("chunk_type"),
                chunk.get("content_text"),
                json.dumps(chunk.get("bbox")) if chunk.get("bbox") else None,
                chunk.get("parent_chunk_id"),
                chunk.get("token_count"),
                start_index + i,
                json.dumps(chunk.get("metadata", {}))
            ))
        
        cursor.executemany('''
            INSERT OR REPLACE INTO chunks (
                chunk_id, doc_id, page, chunk_type, content_text,
                bbox, parent_chunk_id, token_count, faiss_index_id, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', data)
        
        conn.commit()
        self.logger.info(f"Added {len(chunks)} chunks to metadata store")
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict]:
        """
        Get chunk by ID.
        
        Args:
            chunk_id: Chunk ID
            
        Returns:
            Chunk dictionary or None
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM chunks WHERE chunk_id = ?', (chunk_id,))
        row = cursor.fetchone()
        
        return self._row_to_dict(row) if row else None
    
    def get_chunk_by_faiss_id(self, faiss_id: int) -> Optional[Dict]:
        """
        Get chunk by FAISS index ID.
        
        Args:
            faiss_id: FAISS index ID
            
        Returns:
            Chunk dictionary or None
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM chunks WHERE faiss_index_id = ?', (faiss_id,))
        row = cursor.fetchone()
        
        return self._row_to_dict(row) if row else None
    
    def get_chunks_by_doc_id(self, doc_id: str) -> List[Dict]:
        """
        Get all chunks for a document.
        
        Args:
            doc_id: Document ID
            
        Returns:
            List of chunks
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM chunks WHERE doc_id = ? ORDER BY page, chunk_id', (doc_id,))
        rows = cursor.fetchall()
        
        return [self._row_to_dict(row) for row in rows]
    
    def get_all_chunks(self) -> List[Dict]:
        """
        Get all chunks.
        
        Returns:
            List of all chunks
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM chunks ORDER BY doc_id, page, chunk_id')
        rows = cursor.fetchall()
        
        return [self._row_to_dict(row) for row in rows]
    
    def _row_to_dict(self, row: sqlite3.Row) -> Dict:
        """
        Convert SQLite row to dictionary.
        
        Args:
            row: SQLite row object
            
        Returns:
            Dictionary representation
        """
        chunk = dict(row)
        
        # Parse JSON fields
        if chunk.get("bbox"):
            chunk["bbox"] = json.loads(chunk["bbox"])
        if chunk.get("metadata"):
            chunk["metadata"] = json.loads(chunk["metadata"])
        
        return chunk
    
    def count_chunks(self) -> int:
        """
        Count total chunks.
        
        Returns:
            Number of chunks
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM chunks')
        return cursor.fetchone()[0]
    
    def close(self):
        """Close database connection."""
        if hasattr(self._local, 'conn') and self._local.conn is not None:
            self._local.conn.close()
            self._local.conn = None
