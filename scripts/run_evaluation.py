"""CLI script to run evaluation on test queries."""

import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from tqdm import tqdm

from src.utils import get_config, setup_logging, save_json
from src.embeddings import CLIPEmbedder
from src.indexing import FAISSIndex, BM25Index, MetadataStore
from src.retrieval import HybridRetriever, CLIPReranker, MultiStageRetriever
from src.generation import Generator
from src.evaluation import RetrievalMetrics, LatencyTracker, FaithfulnessChecker, EvalData

logger = setup_logging(level="INFO")


def run_evaluation():
    """Run evaluation on test queries."""
    config = get_config()
    
    # Check if indices exist
    if not config.faiss_index_path.exists():
        logger.error("FAISS index not found. Please run build_index.py first")
        return
    
    if not config.bm25_index_path.exists():
        logger.error("BM25 index not found. Please run build_index.py first")
        return
    
    # Load test queries
    eval_data = EvalData(config.get("evaluation.test_queries_path"))
    
    if not eval_data.queries:
        logger.warning("No test queries found. Creating sample queries...")
        eval_data.create_sample_queries()
    
    queries = eval_data.get_queries()
    logger.info(f"Loaded {len(queries)} test queries")
    
    # Initialize components
    logger.info("Loading models and indices...")
    
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
    
    generator = Generator(
        model_name=config.generation_model,
        max_new_tokens=config.get("generation.max_new_tokens", 256)
    )
    
    # Run evaluation
    logger.info("\nRunning evaluation...")
    
    latency_tracker = LatencyTracker()
    all_metrics = []
    per_query_results = []
    
    for query_data in tqdm(queries, desc="Evaluating"):
        query = query_data["query"]
        ground_truth_pages = query_data.get("ground_truth_pages", [])
        
        # Retrieve
        latency_tracker.start("retrieval")
        results = multi_stage_retriever.retrieve(query, stage1_top_k=50, stage2_top_k=10)
        latency_tracker.stop("retrieval")
        
        # Extract retrieved pages
        retrieved_pages = [r.get("page") for r in results]
        retrieved_ids = [f"p{page}" for page in retrieved_pages]
        ground_truth_ids = [f"p{page}" for page in ground_truth_pages]
        
        # Compute retrieval metrics
        metrics = RetrievalMetrics.compute_all_metrics(
            retrieved_ids,
            ground_truth_ids,
            k_values=[1, 5, 10]
        )
        
        # Generate answer
        latency_tracker.start("generation")
        answer_data = generator.generate_answer(query, results)
        latency_tracker.stop("generation")
        
        # Check faithfulness
        is_faithful, faith_score = FaithfulnessChecker.check_faithfulness(
            answer_data["answer"],
            results
        )
        
        # Store per-query results
        per_query_results.append({
            "query": query,
            "retrieved_pages": retrieved_pages,
            "ground_truth_pages": ground_truth_pages,
            "metrics": metrics,
            "answer": answer_data["answer"],
            "faithfulness": faith_score,
            "is_faithful": is_faithful
        })
        
        all_metrics.append(metrics)
    
    # Aggregate metrics
    logger.info("\nAggregating metrics...")
    
    aggregated_metrics = {}
    metric_keys = all_metrics[0].keys()
    
    for key in metric_keys:
        values = [m[key] for m in all_metrics]
        aggregated_metrics[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values))
        }
    
    # Latency stats
    latency_stats = latency_tracker.get_all_stats()
    
    # Faithfulness stats
    faithfulness_scores = [r["faithfulness"] for r in per_query_results]
    is_faithful_count = sum(1 for r in per_query_results if r["is_faithful"])
    
    # Compile results
    results_summary = {
        "num_queries": len(queries),
        "aggregated_metrics": aggregated_metrics,
        "latency_stats": latency_stats,
        "faithfulness": {
            "mean": float(np.mean(faithfulness_scores)),
            "std": float(np.std(faithfulness_scores)),
            "faithful_percentage": (is_faithful_count / len(queries)) * 100
        },
        "per_query_results": per_query_results
    }
    
    # Save results
    results_file = Path("data/eval_results.json")
    save_json(results_summary, results_file)
    logger.info(f"Saved evaluation results to {results_file}")
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*50)
    logger.info(f"Queries evaluated: {len(queries)}")
    logger.info(f"\nRetrieval Metrics (mean):")
    logger.info(f"  Recall@10: {aggregated_metrics['recall@10']['mean']:.2%}")
    logger.info(f"  Precision@5: {aggregated_metrics['precision@5']['mean']:.2%}")
    logger.info(f"  MRR: {aggregated_metrics['reciprocal_rank']['mean']:.3f}")
    
    logger.info(f"\nLatency (ms):")
    logger.info(f"  Retrieval: {latency_stats.get('retrieval', {}).get('mean', 0):.1f}")
    logger.info(f"  Generation: {latency_stats.get('generation', {}).get('mean', 0):.1f}")
    
    logger.info(f"\nFaithfulness:")
    logger.info(f"  Mean score: {results_summary['faithfulness']['mean']:.2f}")
    logger.info(f"  Faithful answers: {results_summary['faithfulness']['faithful_percentage']:.1f}%")
    
    logger.info("="*50)
    
    metadata_store.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run evaluation on test queries")
    args = parser.parse_args()
    
    run_evaluation()


if __name__ == "__main__":
    main()
