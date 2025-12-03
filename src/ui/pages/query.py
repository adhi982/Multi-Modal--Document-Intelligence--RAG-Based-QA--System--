"""Query interface page."""

import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils import get_config, get_logger
from src.embeddings import CLIPEmbedder
from src.indexing import FAISSIndex, BM25Index, MetadataStore
from src.retrieval import HybridRetriever, CLIPReranker, MultiStageRetriever
from src.generation import UnifiedGenerator, Summarizer


@st.cache_resource
def load_models():
    """Load models and indices (cached)."""
    config = get_config()
    
    # Load embedder
    embedder = CLIPEmbedder(model_name=config.clip_model)
    
    # Load indices
    faiss_index = FAISSIndex(dimension=config.embedding_dimension)
    faiss_index.load(str(config.faiss_index_path))
    
    bm25_index = BM25Index()
    bm25_index.load(str(config.bm25_index_path))
    
    metadata_store = MetadataStore(db_path=str(config.metadata_db_path))
    
    # Create retrievers
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
    
    # Load generator (Mistral API with local fallback)
    generator = UnifiedGenerator(
        local_model_name=config.generation_model,
        use_local_fallback=True,
        max_new_tokens=config.get("generation.max_new_tokens", 512)
    )
    
    # Create summarizer
    summarizer = Summarizer(generator)
    
    return multi_stage_retriever, generator, summarizer


def show():
    """Show query interface page."""
    st.title("Multi-Modal Document QA")
    st.write("Ask questions about your documents and get accurate, citation-backed answers.")
    
    # Check if indices exist
    config = get_config()
    
    if not config.faiss_index_path.exists() or not config.bm25_index_path.exists():
        st.error("Indices not found. Please run `python scripts/build_index.py` first.")
        st.info("Steps:\n1. Place PDFs in `data/raw/`\n2. Run: `python scripts/ingest_documents.py`\n3. Run: `python scripts/build_index.py`")
        return
    
    # Initialize session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "example_used" not in st.session_state:
        st.session_state.example_used = False
    
    try:
        # Load models
        with st.spinner("Loading models..."):
            retriever, generator, summarizer = load_models()
        
        # Sidebar with settings and example queries
        with st.sidebar:
            st.header("Settings")
            
            top_k = st.slider("Number of chunks to retrieve", 5, 30, 10, 1,
                             help="More chunks = better context but slower")
            
            generation_temp = st.slider("Generation temperature", 0.0, 1.0, 0.3, 0.1,
                                       help="Higher = more creative, Lower = more focused")
            
            st.markdown("---")
            st.header("Example Queries")
            
            examples = [
                "What is Qatar's GDP growth for 2024-2025?",
                "What are the main economic risks mentioned?",
                "What is the fiscal balance outlook?",
                "How is the banking sector performing?",
                "What are the inflation projections?"
            ]
            
            for example in examples:
                if st.button(f"{example}", use_container_width=True):
                    st.session_state.example_query = example
                    st.session_state.example_used = True
            
            if st.button("Clear Chat History", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
        
        # Query input with example handling
        default_query = st.session_state.get("example_query", "")
        if st.session_state.example_used:
            query = default_query
            st.session_state.example_used = False
        else:
            query = st.text_input(
                "Enter your question:", 
                value=default_query,
                placeholder="What is the GDP growth rate in 2024?",
                help="Ask questions about the documents in natural language"
            )
        
        col1, col2, col3 = st.columns([2, 2, 3])
        
        with col1:
            search_button = st.button("Search", type="primary", use_container_width=True)
        
        with col2:
            briefing_mode = st.checkbox("Generate briefing", value=False,
                                       help="Create a comprehensive executive summary")
        
        with col3:
            show_retrieved = st.checkbox("Show retrieved chunks", value=True,
                                        help="Display the source documents used")
        
        if search_button and query:
            # Add to history
            st.session_state.chat_history.append({
                "query": query,
                "timestamp": "now"
            })
            
            with st.spinner("Searching documents..."):
                # Retrieve documents
                results = retriever.retrieve(
                    query,
                    stage1_top_k=50,
                    stage2_top_k=top_k
                )
                
                if not results:
                    st.warning("No results found for your query. Try rephrasing or using different keywords.")
                    return
                
                st.success(f"Found {len(results)} relevant chunks")
                
                # Display retrieved chunks (optional)
                if show_retrieved:
                    with st.expander(f"Retrieved Chunks ({len(results)} documents)", expanded=False):
                        for i, result in enumerate(results, start=1):
                            chunk_type = result.get('chunk_type', 'text')
                            
                            # Create tabs for better organization
                            col_a, col_b = st.columns([3, 1])
                            
                            with col_a:
                                st.markdown(f"**{i}. {result.get('doc_id', 'unknown')} - Page {result.get('page', '?')}** `{chunk_type}`")
                            
                            with col_b:
                                st.metric("Score", f"{result.get('rrf_score', 0):.3f}")
                            
                            # Special rendering for tables
                            if chunk_type == "table":
                                table_data = result.get("table_data", {})
                                if table_data.get("dataframe"):
                                    try:
                                        import pandas as pd
                                        df = pd.DataFrame(table_data["dataframe"])
                                        st.dataframe(df, use_container_width=True)
                                    except:
                                        st.code(table_data.get("markdown", result.get("content_text", "")), language=None)
                                elif table_data.get("markdown"):
                                    st.code(table_data["markdown"], language=None)
                                else:
                                    st.write(result.get("content_text", ""))
                            else:
                                # Truncate long text for preview
                                content = result.get("content_text", "")
                                if len(content) > 300:
                                    st.text(content[:300] + "...")
                                else:
                                    st.text(content)
                            
                            if i < len(results):
                                st.markdown("---")
                
                # Generate answer or briefing
                st.markdown("---")
                
                if briefing_mode:
                    with st.spinner("Generating comprehensive briefing..."):
                        briefing_data = summarizer.generate_briefing(query, results, n_clusters=3)
                        
                        # Display with nice formatting
                        st.markdown("### Executive Briefing")
                        
                        # Add query context
                        st.info(f"**Query:** {query}")
                        
                        st.markdown(briefing_data["briefing"])
                        
                        # Feedback section
                        st.markdown("---")
                        col_fb1, col_fb2, col_fb3 = st.columns(3)
                        with col_fb1:
                            if st.button("Helpful"):
                                st.success("Thank you for your feedback!")
                        with col_fb2:
                            if st.button("Not helpful"):
                                st.info("We'll improve our answers!")
                        with col_fb3:
                            if st.button("Copy to clipboard"):
                                st.code(briefing_data["briefing"])
                
                else:
                    with st.spinner("Generating answer..."):
                        answer_data = generator.generate_answer(query, results)
                        
                        # Display in chat bubble style
                        st.markdown("### Answer")
                        
                        # Query context
                        st.info(f"**Your Question:** {query}")
                        
                        # Answer in a nice container
                        with st.container():
                            st.markdown(answer_data["answer"])
                        
                        # Interactive elements
                        st.markdown("---")
                        
                        col_int1, col_int2 = st.columns([3, 1])
                        
                        with col_int1:
                            with st.expander("View Detailed Sources", expanded=False):
                                st.write(f"**{len(answer_data['sources'])} sources used:**")
                                for idx, source in enumerate(answer_data["sources"], 1):
                                    page = source.get('page_num') or source.get('page', 'N/A')
                                    doc_id = source.get('doc_id', 'unknown')
                                    chunk_type = source.get('chunk_type', 'text')
                                    score = source.get('score', 0)
                                    st.write(f"{idx}. **{doc_id}**, Page {page} ({chunk_type}) - Score: {score:.3f}")
                        
                        with col_int2:
                            st.metric("Sources", len(answer_data['sources']))
                            st.metric("Model", "Mistral" if "mistral" in str(type(generator)).lower() else "FLAN-T5")
                        
                        # Feedback section
                        st.markdown("---")
                        st.markdown("**Was this answer helpful?**")
                        col_fb1, col_fb2, col_fb3, col_fb4 = st.columns(4)
                        
                        with col_fb1:
                            if st.button("Yes, helpful", use_container_width=True):
                                st.success("Great! Thank you!")
                        
                        with col_fb2:
                            if st.button("Needs improvement", use_container_width=True):
                                st.info("We'll work on improving!")
                        
                        with col_fb3:
                            if st.button("Regenerate", use_container_width=True):
                                st.rerun()
                        
                        with col_fb4:
                            if st.button("Copy answer", use_container_width=True):
                                st.code(answer_data["answer"])
                
                # Follow-up suggestions
                st.markdown("---")
                st.markdown("**Suggested follow-up questions:**")
                
                # Generate context-aware follow-ups based on the query
                follow_ups = [
                    f"Can you elaborate on the key points about {query.split()[-1] if query else 'this'}?",
                    f"What are the implications of this?",
                    f"Are there any risks mentioned?",
                    f"What is the timeline for this?"
                ]
                
                cols = st.columns(len(follow_ups))
                for idx, followup in enumerate(follow_ups):
                    with cols[idx]:
                        if st.button(followup, use_container_width=True, key=f"followup_{idx}"):
                            # Add the follow-up to history and trigger new search
                            st.session_state.example_query = followup
                            st.session_state.example_used = True
                            st.info(f"Processing follow-up: {followup}")
                            st.rerun()
        
        # Display chat history in sidebar
        if st.session_state.chat_history:
            with st.sidebar:
                st.markdown("---")
                st.header("Recent Queries")
                for i, chat in enumerate(reversed(st.session_state.chat_history[-5:]), 1):
                    if st.button(f"{6-i}. {chat['query'][:40]}...", use_container_width=True, key=f"history_{i}"):
                        st.session_state.example_query = chat['query']
                        st.session_state.example_used = True
                        st.rerun()
    
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.exception(e)
