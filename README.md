# Multi-Modal Document Intelligence System

A production-quality RAG-based QA system that processes text, tables, and images from complex documents (PDFs) and provides accurate, citation-backed answers.

## Features

-  **Multi-modal Ingestion**: Text, tables, images, and OCR
-  **Unified Embeddings**: CLIP-based text and image embeddings in shared space
-  **Hybrid Retrieval**: BM25 + FAISS with RRF fusion
-  **Cross-modal Reranking**: CLIP similarity scoring
-  **Grounded Generation**: FLAN-T5 with citations
-  **Multi-document Briefing**: Clustered summarization
-  **Evaluation Dashboard**: Metrics visualization

## Architecture

```
PDF Documents → Ingestion → Chunking → CLIP Embeddings → FAISS + BM25 Index
                                                              ↓
User Query → Hybrid Retrieval → Cross-modal Reranking → FLAN-T5 → Answer
```

## Quick Start

### 1. Setup Environment

```powershell
# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data

Place your PDF documents in `data/raw/`

### 3. Ingest Documents

```powershell
python scripts/ingest_documents.py --input data/raw/
```

### 4. Build Indices

```powershell
python scripts/build_index.py
```

### 5. Run UI

```powershell
streamlit run src/ui/app.py
```

## Project Structure

```
├── data/                   # Data storage
│   ├── raw/               # Input PDFs
│   ├── processed/         # Parsed chunks
│   ├── indices/           # FAISS + BM25 indices
│   └── metadata/          # SQLite database
├── src/
│   ├── ingestion/         # PDF parsing, OCR, chunking
│   ├── embeddings/        # CLIP embeddings
│   ├── indexing/          # FAISS + BM25 + SQLite
│   ├── retrieval/         # Hybrid retrieval + reranking
│   ├── generation/        # FLAN-T5 generation
│   ├── evaluation/        # Metrics computation
│   └── ui/                # Streamlit interface
└── scripts/               # CLI utilities
```

## Configuration

Edit `config.yaml` to customize:
- Model selection
- Chunking parameters
- Retrieval settings
- Generation parameters

## Evaluation

Run evaluation on test queries:

```powershell
python scripts/run_evaluation.py
```

View results in the Streamlit dashboard (Metrics page).

## Tech Stack

- **Embeddings**: CLIP (laion/CLIP-ViT-B-32)
- **Generation**: FLAN-T5-base
- **Vector DB**: FAISS + SQLite
- **PDF Parsing**: PyMuPDF + pdfplumber
- **OCR**: EasyOCR
- **UI**: Streamlit

## Performance

- **Recall@10**: 65-75%
- **MRR**: 0.55-0.65
- **Latency**: <3s end-to-end (CPU)
- **Memory**: <2GB RAM


