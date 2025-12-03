"""Metrics dashboard page."""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils import load_json


def show():
    """Show metrics dashboard page."""
    st.title("Evaluation Dashboard")
    st.write("View retrieval and generation metrics from evaluation runs.")
    
    # Load evaluation results
    results_file = Path("data/eval_results.json")
    
    if not results_file.exists():
        st.warning("No evaluation results found.")
        st.info("Run evaluation first: `python scripts/run_evaluation.py`")
        return
    
    try:
        results = load_json(results_file)
        
        # Summary metrics
        st.subheader("Summary Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        recall_10 = results["aggregated_metrics"]["recall@10"]["mean"]
        precision_5 = results["aggregated_metrics"]["precision@5"]["mean"]
        mrr = results["aggregated_metrics"]["reciprocal_rank"]["mean"]
        faithfulness = results["faithfulness"]["mean"]
        
        col1.metric("Recall@10", f"{recall_10:.2%}")
        col2.metric("Precision@5", f"{precision_5:.2%}")
        col3.metric("MRR", f"{mrr:.3f}")
        col4.metric("Faithfulness", f"{faithfulness:.2f}")
        
        # Retrieval metrics chart
        st.subheader("Retrieval Metrics")
        
        metrics_data = {
            "Metric": ["Recall@1", "Recall@5", "Recall@10", "Precision@1", "Precision@5", "Precision@10"],
            "Score": [
                results["aggregated_metrics"]["recall@1"]["mean"],
                results["aggregated_metrics"]["recall@5"]["mean"],
                results["aggregated_metrics"]["recall@10"]["mean"],
                results["aggregated_metrics"]["precision@1"]["mean"],
                results["aggregated_metrics"]["precision@5"]["mean"],
                results["aggregated_metrics"]["precision@10"]["mean"],
            ]
        }
        
        fig_metrics = px.bar(
            metrics_data,
            x="Metric",
            y="Score",
            title="Retrieval Metrics",
            color="Score",
            color_continuous_scale="Blues"
        )
        fig_metrics.update_layout(height=400)
        st.plotly_chart(fig_metrics, use_container_width=True)
        
        # Latency breakdown
        if "latency_stats" in results:
            st.subheader("Latency Breakdown")
            
            latency_data = results["latency_stats"]
            
            if latency_data:
                components = list(latency_data.keys())
                means = [latency_data[c].get("mean", 0) for c in components]
                
                fig_latency = go.Figure()
                fig_latency.add_trace(go.Bar(
                    x=components,
                    y=means,
                    text=[f"{m:.1f}ms" for m in means],
                    textposition='auto',
                ))
                fig_latency.update_layout(
                    title="Average Latency by Component",
                    yaxis_title="Time (ms)",
                    height=400
                )
                st.plotly_chart(fig_latency, use_container_width=True)
                
                # Latency percentiles
                col_a, col_b, col_c = st.columns(3)
                
                for component in components:
                    stats = latency_data[component]
                    with col_a if components.index(component) % 3 == 0 else (col_b if components.index(component) % 3 == 1 else col_c):
                        st.metric(
                            f"{component.title()} (p95)",
                            f"{stats.get('p95', 0):.1f}ms"
                        )
        
        # Per-query results table
        st.subheader("Per-Query Results")
        
        per_query = results.get("per_query_results", [])
        
        if per_query:
            table_data = []
            
            for i, qr in enumerate(per_query, start=1):
                table_data.append({
                    "Query": qr["query"][:50] + "..." if len(qr["query"]) > 50 else qr["query"],
                    "Recall@10": f"{qr['metrics']['recall@10']:.2%}",
                    "Precision@5": f"{qr['metrics']['precision@5']:.2%}",
                    "RR": f"{qr['metrics']['reciprocal_rank']:.3f}",
                    "Faithfulness": f"{qr['faithfulness']:.2f}",
                    "Faithful": "Yes" if qr["is_faithful"] else "No"
                })
            
            df = pd.DataFrame(table_data)
            st.dataframe(df, use_container_width=True, height=400)
            
            # Download results
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name="eval_results.csv",
                mime="text/csv"
            )
    
    except Exception as e:
        st.error(f"Error loading results: {str(e)}")
        st.exception(e)
