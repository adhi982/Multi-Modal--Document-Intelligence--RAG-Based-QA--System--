import yaml
from src.retrieval.retriever import HybridRetriever
from src.generation.generator import Generator
from src.generation.prompts import create_grounded_prompt
from src.indexing.faiss_index import FAISSIndex
from src.indexing.bm25_index import BM25Index
from src.indexing.metadata_store import MetadataStore
from src.embeddings.clip_embedder import CLIPEmbedder

# Load config
with open("config.yaml") as f:
    config = yaml.safe_load(f)

print("Loading indices...")
# Initialize components
embedder = CLIPEmbedder(config['models']['clip_model'])
faiss_index = FAISSIndex(dimension=512)
faiss_index = faiss_index.load(config['paths']['faiss_index'])
bm25_index = BM25Index()
bm25_index = bm25_index.load(config['paths']['bm25_index'])
metadata_store = MetadataStore(config['paths']['metadata_db'])

# Create retriever
retriever = HybridRetriever(
    faiss_index=faiss_index,
    bm25_index=bm25_index,
    metadata_store=metadata_store,
    embedder=embedder,
    bm25_weight=0.5,
    faiss_weight=0.5,
    rrf_k=config['retrieval']['rrf_k']
)

generator = Generator(**config['generation'])

# Test query
query = "What is Qatar's GDP growth for 2024-2025?"
print(f"Query: {query}\n")

# Retrieve
results = retriever.retrieve(query, top_k=5)
print(f"Retrieved {len(results)} chunks\n")

# Show what we're sending to the model
print("="*80)
print("CHUNKS BEING SENT:")
print("="*80)
for i, r in enumerate(results[:3], 1):
    print(f"\n[Chunk {i}] Page {r.get('page')}, Type: {r.get('chunk_type')}")
    if r.get('chunk_type') == 'table':
        print("HAS TABLE DATA:")
        table_data = r.get('table_data', {})
        print(f"  - Has markdown: {bool(table_data.get('markdown'))}")
        print(f"  - Has plain_text: {bool(table_data.get('plain_text'))}")
        if table_data.get('plain_text'):
            print(f"\nPlain text preview:\n{table_data['plain_text'][:300]}\n")
    print(f"Content text preview:\n{r.get('content_text', '')[:200]}")

# Show the prompt
print("\n" + "="*80)
print("PROMPT BEING SENT TO MODEL:")
print("="*80)
prompt = create_grounded_prompt(query, results[:3])
print(prompt[:1500])

# Generate
print("\n" + "="*80)
print("GENERATING ANSWER...")
print("="*80)
answer = generator.generate_answer(query, results)
print(f"\nGenerated Answer:\n{answer['answer']}")
