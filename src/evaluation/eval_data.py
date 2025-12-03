"""Test data and evaluation queries."""

import json
from pathlib import Path
from typing import List, Dict

from ..utils import get_logger, load_json, save_json

logger = get_logger(__name__)


class EvalData:
    """Manage evaluation test queries and ground truth."""
    
    def __init__(self, data_path: str = "data/test_queries.json"):
        """
        Initialize evaluation data.
        
        Args:
            data_path: Path to test queries JSON file
        """
        self.data_path = Path(data_path)
        self.logger = logger
        self.queries = []
        
        if self.data_path.exists():
            self.load()
    
    def load(self):
        """Load test queries from file."""
        try:
            self.queries = load_json(self.data_path)
            self.logger.info(f"Loaded {len(self.queries)} test queries")
        except Exception as e:
            self.logger.warning(f"Could not load test queries: {str(e)}")
            self.queries = []
    
    def save(self):
        """Save test queries to file."""
        try:
            save_json(self.queries, self.data_path)
            self.logger.info(f"Saved {len(self.queries)} test queries")
        except Exception as e:
            self.logger.error(f"Could not save test queries: {str(e)}")
    
    def add_query(
        self,
        query: str,
        ground_truth_docs: List[str],
        ground_truth_pages: List[int],
        expected_answer_keywords: List[str] = None,
        category: str = "general"
    ):
        """
        Add a test query.
        
        Args:
            query: Query text
            ground_truth_docs: List of relevant document IDs
            ground_truth_pages: List of relevant page numbers
            expected_answer_keywords: Expected keywords in answer
            category: Query category
        """
        query_data = {
            "query": query,
            "ground_truth_docs": ground_truth_docs,
            "ground_truth_pages": ground_truth_pages,
            "expected_answer_keywords": expected_answer_keywords or [],
            "category": category
        }
        
        self.queries.append(query_data)
        self.logger.info(f"Added query: {query[:50]}...")
    
    def get_queries(self, category: str = None) -> List[Dict]:
        """
        Get queries, optionally filtered by category.
        
        Args:
            category: Optional category filter
            
        Returns:
            List of queries
        """
        if category:
            return [q for q in self.queries if q.get("category") == category]
        return self.queries
    
    def create_sample_queries(self):
        """Create sample test queries for demonstration."""
        self.queries = [
            # General Understanding
            {
                "query": "What is the main topic of the document?",
                "ground_truth_docs": ["qatar_test_doc"],
                "ground_truth_pages": [1, 2],
                "expected_answer_keywords": ["Qatar", "economic", "IMF", "staff report"],
                "category": "general"
            },
            {
                "query": "What are the key findings of the report?",
                "ground_truth_docs": ["qatar_test_doc"],
                "ground_truth_pages": [1, 2, 3],
                "expected_answer_keywords": ["growth", "outlook", "reforms", "stability"],
                "category": "general"
            },
            {
                "query": "What is the overall economic outlook for Qatar?",
                "ground_truth_docs": ["qatar_test_doc"],
                "ground_truth_pages": [2, 3, 4],
                "expected_answer_keywords": ["growth", "GDP", "outlook", "forecast"],
                "category": "general"
            },
            {
                "query": "What are the major risks mentioned?",
                "ground_truth_docs": ["qatar_test_doc"],
                "ground_truth_pages": [5, 6],
                "expected_answer_keywords": ["risks", "vulnerabilities", "challenges"],
                "category": "general"
            },
            {
                "query": "What reforms are highlighted in the document?",
                "ground_truth_docs": ["qatar_test_doc"],
                "ground_truth_pages": [7, 8],
                "expected_answer_keywords": ["reforms", "structural", "fiscal", "policy"],
                "category": "general"
            },
            
            # Macroeconomic Summary
            {
                "query": "What is Qatar's projected GDP growth?",
                "ground_truth_docs": ["qatar_test_doc"],
                "ground_truth_pages": [3, 4],
                "expected_answer_keywords": ["GDP", "growth", "%", "projection"],
                "category": "macroeconomic"
            },
            {
                "query": "What is the expected inflation rate?",
                "ground_truth_docs": ["qatar_test_doc"],
                "ground_truth_pages": [4, 5],
                "expected_answer_keywords": ["inflation", "CPI", "%", "rate"],
                "category": "macroeconomic"
            },
            {
                "query": "What is the status of Qatar's fiscal balance?",
                "ground_truth_docs": ["qatar_test_doc"],
                "ground_truth_pages": [6, 7],
                "expected_answer_keywords": ["fiscal", "balance", "surplus", "deficit"],
                "category": "macroeconomic"
            },
            {
                "query": "What factors are driving Qatar's medium-term growth?",
                "ground_truth_docs": ["qatar_test_doc"],
                "ground_truth_pages": [3, 4, 5],
                "expected_answer_keywords": ["growth", "drivers", "LNG", "investment"],
                "category": "macroeconomic"
            },
            
            # Sector-Level Insights
            {
                "query": "How is the banking sector performing?",
                "ground_truth_docs": ["qatar_test_doc"],
                "ground_truth_pages": [10, 11],
                "expected_answer_keywords": ["banking", "financial", "sector", "credit"],
                "category": "sectoral"
            },
            {
                "query": "What is the NPL (non-performing loan) situation?",
                "ground_truth_docs": ["qatar_test_doc"],
                "ground_truth_pages": [11, 12],
                "expected_answer_keywords": ["NPL", "non-performing", "loan", "asset quality"],
                "category": "sectoral"
            },
            {
                "query": "How is credit growth trending?",
                "ground_truth_docs": ["qatar_test_doc"],
                "ground_truth_pages": [10, 11],
                "expected_answer_keywords": ["credit", "growth", "lending", "bank"],
                "category": "sectoral"
            },
            {
                "query": "How did the real estate sector perform?",
                "ground_truth_docs": ["qatar_test_doc"],
                "ground_truth_pages": [12, 13],
                "expected_answer_keywords": ["real estate", "property", "housing", "sector"],
                "category": "sectoral"
            },
            
            # Policy Insights
            {
                "query": "What fiscal reforms are recommended?",
                "ground_truth_docs": ["qatar_test_doc"],
                "ground_truth_pages": [15, 16],
                "expected_answer_keywords": ["fiscal", "reforms", "policy", "recommendation"],
                "category": "policy"
            },
            {
                "query": "What are the recommendations for monetary policy?",
                "ground_truth_docs": ["qatar_test_doc"],
                "ground_truth_pages": [16, 17],
                "expected_answer_keywords": ["monetary", "policy", "central bank", "interest"],
                "category": "policy"
            },
            {
                "query": "What structural reforms does the IMF suggest?",
                "ground_truth_docs": ["qatar_test_doc"],
                "ground_truth_pages": [17, 18],
                "expected_answer_keywords": ["structural", "reforms", "IMF", "recommendation"],
                "category": "policy"
            },
            {
                "query": "What are the priorities under NDS3?",
                "ground_truth_docs": ["qatar_test_doc"],
                "ground_truth_pages": [18, 19],
                "expected_answer_keywords": ["NDS3", "priorities", "development", "strategy"],
                "category": "policy"
            },
            {
                "query": "What tax reforms are proposed, such as VAT or CIT?",
                "ground_truth_docs": ["qatar_test_doc"],
                "ground_truth_pages": [20, 21],
                "expected_answer_keywords": ["tax", "VAT", "CIT", "reform", "revenue"],
                "category": "policy"
            },
            
            # Energy & LNG
            {
                "query": "What is Qatar's LNG expansion plan?",
                "ground_truth_docs": ["qatar_test_doc"],
                "ground_truth_pages": [22, 23],
                "expected_answer_keywords": ["LNG", "expansion", "capacity", "production"],
                "category": "energy"
            },
            {
                "query": "How will LNG production impact GDP?",
                "ground_truth_docs": ["qatar_test_doc"],
                "ground_truth_pages": [23, 24],
                "expected_answer_keywords": ["LNG", "GDP", "impact", "contribution"],
                "category": "energy"
            },
            {
                "query": "What are the risks related to hydrocarbon prices?",
                "ground_truth_docs": ["qatar_test_doc"],
                "ground_truth_pages": [25, 26],
                "expected_answer_keywords": ["hydrocarbon", "prices", "oil", "gas", "risk"],
                "category": "energy"
            },
            
            # Labor & Human Capital
            {
                "query": "What workforce reforms are discussed?",
                "ground_truth_docs": ["qatar_test_doc"],
                "ground_truth_pages": [30, 31],
                "expected_answer_keywords": ["workforce", "labor", "reform", "employment"],
                "category": "labor"
            },
            {
                "query": "How is Qatar improving skilled labor availability?",
                "ground_truth_docs": ["qatar_test_doc"],
                "ground_truth_pages": [31, 32],
                "expected_answer_keywords": ["skilled", "labor", "education", "training"],
                "category": "labor"
            },
            {
                "query": "What steps are being taken to boost private-sector employment?",
                "ground_truth_docs": ["qatar_test_doc"],
                "ground_truth_pages": [32, 33],
                "expected_answer_keywords": ["private sector", "employment", "jobs", "Qatarization"],
                "category": "labor"
            },
            
            # Climate & Sustainability
            {
                "query": "What climate-related initiatives are mentioned?",
                "ground_truth_docs": ["qatar_test_doc"],
                "ground_truth_pages": [35, 36],
                "expected_answer_keywords": ["climate", "sustainability", "environment", "green"],
                "category": "climate"
            },
            {
                "query": "How is Qatar addressing energy pricing reforms?",
                "ground_truth_docs": ["qatar_test_doc"],
                "ground_truth_pages": [36, 37],
                "expected_answer_keywords": ["energy", "pricing", "reform", "subsidy"],
                "category": "climate"
            },
            
            # Document-Level Summary
            {
                "query": "Summarize the key insights and conclusions from the entire document.",
                "ground_truth_docs": ["qatar_test_doc"],
                "ground_truth_pages": [1, 2, 3, 75, 76, 77, 78],
                "expected_answer_keywords": ["summary", "conclusion", "outlook", "recommendations"],
                "category": "summary"
            }
        ]
        
        self.save()
        self.logger.info("Created sample test queries")
