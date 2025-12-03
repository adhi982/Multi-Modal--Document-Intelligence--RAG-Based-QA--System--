"""Example queries page."""

import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils import get_config
from src.evaluation import EvalData
from src.embeddings import CLIPEmbedder
from src.indexing import FAISSIndex, BM25Index, MetadataStore
from src.retrieval import HybridRetriever, CLIPReranker, MultiStageRetriever
from src.generation import Generator


@st.cache_resource
def load_components():
    """Load retrieval and generation components (cached)."""
    config = get_config()
    
    embedder = CLIPEmbedder(model_name=config.clip_model)
    
    faiss_index = FAISSIndex(dimension=config.embedding_dimension)
    faiss_index.load(str(config.faiss_index_path))
    
    bm25_index = BM25Index()
    bm25_index.load(str(config.bm25_index_path))
    
    metadata_store = MetadataStore(db_path=str(config.metadata_db_path))
    
    hybrid_retriever = HybridRetriever(
        faiss_index=faiss_index,
        bm25_index=bm25_index,
        metadata_store=metadata_store,
        embedder=embedder,
        rrf_k=config.rrf_k,
        bm25_weight=config.get("retrieval.bm25_weight", 1.5)
    )
    
    reranker = CLIPReranker(embedder=embedder)
    
    multi_stage_retriever = MultiStageRetriever(
        hybrid_retriever=hybrid_retriever,
        reranker=reranker
    )
    
    generator = Generator(model_name=config.generation_model)
    
    return multi_stage_retriever, generator


def show():
    """Show example queries page."""
    st.title("Example Queries")
    st.write("Run predefined example queries and compare results with ground truth.")
    
    # Check if indices exist
    config = get_config()
    
    if not config.faiss_index_path.exists() or not config.bm25_index_path.exists():
        st.error("Indices not found. Please run build_index.py first.")
        return
    
    # Load test queries
    eval_data = EvalData(config.get("evaluation.test_queries_path"))
    
    if not eval_data.queries:
        st.warning("No test queries found. Creating sample queries...")
        eval_data.create_sample_queries()
    
    queries = eval_data.get_queries()
    
    # Query selector
    query_texts = [q["query"] for q in queries]
    selected_query_text = st.selectbox("Select an example query:", query_texts)
    
    # Find selected query data
    selected_query = next(q for q in queries if q["query"] == selected_query_text)
    
    # Display query details
    st.subheader("Query Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Category:** {selected_query.get('category', 'general')}")
        st.write(f"**Ground Truth Pages:** {selected_query.get('ground_truth_pages', [])}")
    
    with col2:
        st.write(f"**Ground Truth Docs:** {selected_query.get('ground_truth_docs', [])}")
        if selected_query.get('expected_answer_keywords'):
            st.write(f"**Expected Keywords:** {', '.join(selected_query['expected_answer_keywords'])}")
    
    # Run button
    if st.button("Run Query", type="primary"):
        try:
            with st.spinner("Loading models..."):
                retriever, generator = load_components()
            
            with st.spinner("Retrieving documents..."):
                results = retriever.retrieve(
                    selected_query_text,
                    stage1_top_k=50,
                    stage2_top_k=10
                )
            
            # Extract retrieved pages
            retrieved_pages = [r.get("page") for r in results]
            ground_truth_pages = selected_query.get("ground_truth_pages", [])
            
            # Compare results
            st.subheader("Comparison")
            
            col_a, col_b, col_c = st.columns(3)
            
            matches = set(retrieved_pages[:10]) & set(ground_truth_pages)
            
            with col_a:
                st.metric("Ground Truth Pages", len(ground_truth_pages))
            
            with col_b:
                st.metric("Retrieved Pages (Top 10)", len(retrieved_pages[:10]))
            
            with col_c:
                st.metric("Matches", len(matches))
            
            # Visualize matches
            st.subheader("Retrieved vs Ground Truth")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Ground Truth Pages:**")
                for page in ground_truth_pages:
                    if page in retrieved_pages[:10]:
                        st.success(f"Page {page} (Retrieved)")
                    else:
                        st.error(f"Page {page} (Missed)")
            
            with col2:
                st.write("**Retrieved Pages (Top 10):**")
                for i, page in enumerate(retrieved_pages[:10], start=1):
                    if page in ground_truth_pages:
                        st.success(f"Rank {i}: Page {page} (Correct)")
                    else:
                        st.warning(f"Rank {i}: Page {page} (Extra)")
            
            # Show retrieved chunks
            st.subheader("Retrieved Chunks")
            
            for i, result in enumerate(results[:5], start=1):
                with st.expander(f"Rank {i}: {result.get('doc_id')} - Page {result.get('page')}"):
                    st.write(result.get("content_text", ""))
                    
                    col_x, col_y = st.columns(2)
                    col_x.metric("RRF Score", f"{result.get('rrf_score', 0):.3f}")
                    col_y.metric("CLIP Score", f"{result.get('clip_score', 0):.3f}")
            
            # Generate answer
            st.subheader("Generated Answer")
            
            with st.spinner("Generating answer..."):
                answer_data = generator.generate_answer(selected_query_text, results)
                st.write(answer_data["answer"])
                
                # Check for expected keywords
                if selected_query.get("expected_answer_keywords"):
                    answer_lower = answer_data["answer"].lower()
                    found_keywords = [
                        kw for kw in selected_query["expected_answer_keywords"]
                        if kw.lower() in answer_lower
                    ]
                    
                    st.write(f"**Expected Keywords Found:** {len(found_keywords)}/{len(selected_query['expected_answer_keywords'])}")
                    st.write(f"Keywords: {', '.join(found_keywords) if found_keywords else 'None'}")
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.exception(e)
