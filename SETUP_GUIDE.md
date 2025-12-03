# Multi-Modal Document Intelligence - Setup Guide

## Complete Setup Instructions

### Prerequisites
- Python 3.8 or higher
- Git (optional)
- 8GB+ RAM recommended
- GPU (optional, for faster processing)

### Step 1: Environment Setup

```powershell
# Navigate to project directory
cd "e:\Multi-Modal Document Intelligence"

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip
```

### Step 2: Install Dependencies

```powershell
# Install all required packages
pip install -r requirements.txt
```

**Note:** This may take 10-15 minutes as it downloads several large models.

### Step 3: Prepare Your Documents

```powershell
# Place your PDF documents in the data/raw/ folder
# Example:
copy "C:\path\to\your\documents\*.pdf" "data\raw\"
```

### Step 4: Ingest Documents

```powershell
# Run the ingestion pipeline
python scripts\ingest_documents.py --input data\raw\
```

This will:
- Parse PDFs (text, tables, images)
- Extract tables using pdfplumber
- Run OCR on images using EasyOCR
- Create semantic chunks
- Save processed chunks to `data/processed/chunks.json`

**Expected time:** 2-5 minutes per document

### Step 5: Build Indices

```powershell
# Build FAISS and BM25 indices
python scripts\build_index.py
```

This will:
- Load processed chunks
- Generate CLIP embeddings for all chunks
- Build FAISS vector index
- Build BM25 lexical index
- Create SQLite metadata database

**Expected time:** 5-10 minutes for 1000 chunks

### Step 6: Run the UI

```powershell
# Start Streamlit interface
streamlit run src\ui\app.py
```

The UI will open in your default browser at `http://localhost:8501`

### Step 7: (Optional) Run Evaluation

```powershell
# Run evaluation on test queries
python scripts\run_evaluation.py
```

This will:
- Load test queries
- Run retrieval and generation
- Compute metrics (Recall, MRR, Precision, Faithfulness)
- Save results to `data/eval_results.json`

---

## Quick Test

To quickly test the system with sample data:

```powershell
# 1. Download a sample PDF (e.g., IMF report)
# Place it in data/raw/

# 2. Run ingestion
python scripts\ingest_documents.py

# 3. Build indices
python scripts\build_index.py

# 4. Launch UI
streamlit run src\ui\app.py

# 5. Ask a question like:
# "What are the key findings?"
# "Show the revenue breakdown"
```

---

## Troubleshooting

### Issue: "FAISS not found" or import errors

```powershell
# Reinstall faiss-cpu
pip uninstall faiss-cpu
pip install faiss-cpu
```

### Issue: "CUDA out of memory"

Edit `config.yaml` and ensure GPU is not used:
```yaml
# Use CPU instead
device: "cpu"
```

### Issue: OCR is slow

OCR runs on CPU by default. To speed up:
1. Install CUDA toolkit
2. Edit `src/ingestion/ocr.py` and set `gpu=True`

Or skip OCR for text-based PDFs.

### Issue: "Module not found"

Ensure you're in the virtual environment:
```powershell
.\venv\Scripts\Activate.ps1
```

And that you're running scripts from the project root:
```powershell
cd "e:\Multi-Modal Document Intelligence"
python scripts\ingest_documents.py
```

---

## Configuration

Edit `config.yaml` to customize:

- **Models:** Change CLIP or FLAN-T5 model versions
- **Chunk size:** Adjust `text_chunk_size` and `text_overlap`
- **Retrieval:** Modify `rrf_k`, `final_top_k`
- **Generation:** Set `max_new_tokens`, `temperature`

---

## Performance Tips

### Speed Up Ingestion
- Use smaller PDF files initially
- Disable OCR for text-based PDFs
- Process in batches

### Speed Up Retrieval
- Use smaller top_k values (50 instead of 100)
- Reduce final_top_k (5 instead of 10)

### Reduce Memory Usage
- Use smaller models:
  - CLIP: `openai/clip-vit-base-patch32`
  - FLAN-T5: `google/flan-t5-small`
- Reduce batch sizes in config

---

## Next Steps

1. **Add more documents:** Continue adding PDFs to `data/raw/` and re-run ingestion
2. **Create test queries:** Edit `data/test_queries.json` with your domain-specific queries
3. **Fine-tune:** Adjust RRF weights, chunking parameters, etc.
4. **Deploy:** Package the Streamlit app for deployment (Streamlit Cloud, Docker, etc.)

---

## Project Structure

```
Multi-Modal Document Intelligence/
├── config.yaml              # Configuration
├── requirements.txt         # Dependencies
├── README.md               # Project overview
│
├── data/
│   ├── raw/                # Input PDFs
│   ├── processed/          # Processed chunks
│   ├── indices/            # FAISS + BM25 indices
│   └── metadata/           # SQLite database
│
├── src/
│   ├── ingestion/          # PDF parsing, OCR, chunking
│   ├── embeddings/         # CLIP embeddings
│   ├── indexing/           # FAISS, BM25, SQLite
│   ├── retrieval/          # Hybrid retrieval + reranking
│   ├── generation/         # FLAN-T5 generation
│   ├── evaluation/         # Metrics computation
│   ├── ui/                 # Streamlit interface
│   └── utils/              # Config, logging, helpers
│
└── scripts/
    ├── ingest_documents.py # CLI: Ingest PDFs
    ├── build_index.py      # CLI: Build indices
    └── run_evaluation.py   # CLI: Run evaluation
```

---

## Support

For issues or questions:
1. Check troubleshooting section above
2. Review logs in terminal output
3. Check `config.yaml` settings
4. Ensure all dependencies are installed

---

## License

MIT License
