"""
HNSW Parameter Auto-Tuning Implementation

This module provides functionality to automatically tune the HNSW parameters
for ChromaDB to optimize recall and search performance based on a validation set.
"""

import os
import json
import time
import logging
import numpy as np
from typing import Dict, List, Any, Tuple
from pathlib import Path

# Setup logger
logger = logging.getLogger("hnsw_optimizer")

class HNSWOptimizer:
    def __init__(self, db_manager, validation_dir: str = None):
        """Initialize HNSW parameter optimizer.
        
        Args:
            db_manager: ChromaManager instance
            validation_dir: Directory to store validation data and results
        """
        self.db_manager = db_manager
        
        # Set default validation directory if not provided
        if validation_dir is None:
            base_dir = Path(db_manager.persist_directory).parent
            validation_dir = os.path.join(base_dir, "validation")
        
        self.validation_dir = validation_dir
        self.validation_file = os.path.join(validation_dir, "validation_set.json")
        self.results_file = os.path.join(validation_dir, "hnsw_tuning_results.json")
        
        # Create validation directory if it doesn't exist
        os.makedirs(validation_dir, exist_ok=True)
        
        # Initialize validation data
        self.validation_data = self._load_validation_data()
        
    def _load_validation_data(self) -> Dict[str, Any]:
        """Load or create validation data for parameter tuning."""
        if os.path.exists(self.validation_file):
            try:
                with open(self.validation_file, 'r') as f:
                    data = json.load(f)
                logger.info(f"Loaded validation data with {len(data.get('queries', []))} queries")
                return data
            except Exception as e:
                logger.warning(f"Failed to load validation data: {e}")
        
        # Default empty validation data structure
        return {
            "queries": [],
            "ground_truth": {},
            "created": time.time(),
            "last_updated": time.time()
        }
    
    def _save_validation_data(self) -> None:
        """Save validation data to disk."""
        try:
            self.validation_data["last_updated"] = time.time()
            with open(self.validation_file, 'w') as f:
                json.dump(self.validation_data, f, indent=2)
            logger.info(f"Saved validation data with {len(self.validation_data.get('queries', []))} queries")
        except Exception as e:
            logger.error(f"Failed to save validation data: {e}")
    
    def add_validation_query(self, query: str, relevant_docs: List[str]) -> None:
        """Add a query with ground truth relevant documents to the validation set.
        
        Args:
            query: The search query
            relevant_docs: List of document IDs/source paths that are relevant for this query
        """
        # Check if query already exists
        for existing_query in self.validation_data.get("queries", []):
            if existing_query == query:
                # Update ground truth for existing query
                self.validation_data["ground_truth"][query] = relevant_docs
                logger.info(f"Updated ground truth for existing query: {query}")
                self._save_validation_data()
                return
        
        # Add new query
        if "queries" not in self.validation_data:
            self.validation_data["queries"] = []
        
        self.validation_data["queries"].append(query)
        self.validation_data["ground_truth"][query] = relevant_docs
        
        logger.info(f"Added new validation query: {query}")
        self._save_validation_data()
    
    def add_validation_queries_from_file(self, file_path: str) -> int:
        """Load validation queries from a JSON file.
        
        Expected format:
        {
            "queries": [
                {
                    "query": "string query text",
                    "relevant_docs": ["doc1.py", "doc2.md", ...]
                },
                ...
            ]
        }
        
        Args:
            file_path: Path to the JSON file with validation queries
            
        Returns:
            Number of queries added
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            count = 0
            for item in data.get("queries", []):
                if "query" in item and "relevant_docs" in item:
                    self.add_validation_query(item["query"], item["relevant_docs"])
                    count += 1
            
            logger.info(f"Added {count} validation queries from {file_path}")
            return count
        except Exception as e:
            logger.error(f"Failed to load validation queries from {file_path}: {e}")
            return 0
    
    def calculate_recall(self, retrieved_docs: List[str], relevant_docs: List[str], k: int = 10) -> float:
        """Calculate recall@k for search results.
        
        Args:
            retrieved_docs: List of document IDs/sources returned by search
            relevant_docs: List of document IDs/sources that are relevant
            k: Number of top results to consider
            
        Returns:
            Recall@k score (0.0-1.0)
        """
        if not relevant_docs:
            return 1.0  # No relevant docs means nothing to recall
        
        hits = sum(1 for doc in retrieved_docs[:k] if doc in relevant_docs)
        return hits / len(relevant_docs)
    
    def calculate_ndcg(self, retrieved_docs: List[str], relevant_docs: List[str], k: int = 10) -> float:
        """Calculate nDCG@k (Normalized Discounted Cumulative Gain).
        
        Args:
            retrieved_docs: List of document IDs/sources returned by search
            relevant_docs: List of document IDs/sources that are relevant
            k: Number of top results to consider
            
        Returns:
            nDCG@k score (0.0-1.0)
        """
        if not relevant_docs:
            return 1.0  # No relevant docs means perfect ranking
        
        # Create relevance dictionary (1.0 for relevant, 0.0 for irrelevant)
        rel_dict = {doc: 1.0 for doc in relevant_docs}
        
        # Calculate DCG
        dcg = 0.0
        for i, doc in enumerate(retrieved_docs[:k]):
            if i >= k:
                break
            rel = rel_dict.get(doc, 0.0)
            # DCG formula: rel_i / log2(i+2)
            dcg += rel / np.log2(i + 2)  # i+2 because i is 0-indexed
        
        # Calculate ideal DCG (perfect ranking of relevant docs)
        ideal_rel = [1.0] * min(len(relevant_docs), k)
        idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_rel))
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def evaluate_parameters(self, parameters: Dict[str, Any], k: int = 10) -> Dict[str, float]:
        """Evaluate a set of HNSW parameters on the validation set.
        
        Args:
            parameters: HNSW parameters to evaluate
            k: Number of top results to consider for metrics
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.validation_data.get("queries"):
            logger.warning("No validation queries available for evaluation")
            return {"recall": 0.0, "ndcg": 0.0, "latency": 0.0}
        
        # Apply parameters to collection
        collection = self.db_manager.get_collection()
        try:
            collection.modify(metadata=parameters)
            logger.info(f"Applied parameters for evaluation: {parameters}")
        except Exception as e:
            logger.error(f"Failed to apply parameters: {e}")
            return {"recall": 0.0, "ndcg": 0.0, "latency": 0.0}
        
        # Initialize metrics
        recall_scores = []
        ndcg_scores = []
        latencies = []
        
        # Process each validation query
        from scripts.search.query_processor import QueryProcessor
        query_processor = QueryProcessor()
        
        for query in self.validation_data["queries"]:
            relevant_docs = self.validation_data["ground_truth"].get(query, [])
            if not relevant_docs:
                continue
            
            try:
                # Process query and get embedding
                start_time = time.time()
                query_data = query_processor.process_query(query)
                
                # Perform search with current parameters
                results = self.db_manager.query(query_data["embedding"], n_results=k)
                end_time = time.time()
                
                # Measure latency
                latency = end_time - start_time
                latencies.append(latency)
                
                # Extract retrieved documents
                if isinstance(results.get('metadatas'), list) and len(results['metadatas']) > 0:
                    # Flatten if needed (some ChromaDB versions return nested lists)
                    metadatas = results['metadatas'][0] if isinstance(results['metadatas'][0], list) else results['metadatas']
                    retrieved_docs = [meta.get("source", "") for meta in metadatas if isinstance(meta, dict)]
                else:
                    # Handle empty results
                    retrieved_docs = []
                
                # Calculate metrics
                recall = self.calculate_recall(retrieved_docs, relevant_docs, k)
                ndcg = self.calculate_ndcg(retrieved_docs, relevant_docs, k)
                
                recall_scores.append(recall)
                ndcg_scores.append(ndcg)
                
            except Exception as e:
                logger.error(f"Error processing validation query '{query}': {e}")
        
        # Calculate average metrics
        avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
        avg_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        
        logger.info(f"Evaluation results: recall@{k}={avg_recall:.4f}, nDCG@{k}={avg_ndcg:.4f}, latency={avg_latency:.4f}s")
        
        return {
            "recall": avg_recall,
            "ndcg": avg_ndcg,
            "latency": avg_latency,
            "k": k
        }
    
    def optimize_hnsw_parameters(self) -> Dict[str, Any]:
        """Find optimal HNSW parameters using the validation set.
        
        This function evaluates different search_ef values to find the
        best tradeoff between recall and latency.
        
        Returns:
            Dictionary with optimal parameters
        """
        if not self.validation_data.get("queries"):
            logger.warning("No validation queries available for optimization")
            return {}
        
        # Define parameter grid to search
        search_ef_values = [96, 128, 150, 175, 200, 225, 250, 300]
        
        # Track results
        results = {}
        
        # Evaluate each parameter combination
        for search_ef in search_ef_values:
            parameters = {
                "hnsw:space": "cosine",
                "hnsw:construction_ef": 200,  # Fixed during search
                "hnsw:search_ef": search_ef,
                "hnsw:M": 16               # Fixed during search
            }
            
            # Evaluate and record metrics
            metrics = self.evaluate_parameters(parameters)
            
            # Store results with parameters
            results[search_ef] = {
                "parameters": parameters,
                "metrics": metrics
            }
        
        # Find optimal parameters based on recall and latency tradeoff
        # We prioritize recall but penalize excessive latency
        def score_fn(result):
            recall = result["metrics"]["recall"]
            latency = result["metrics"]["latency"]
            # Penalize latency over a threshold
            latency_penalty = max(0, (latency - 0.1) / 0.1) if latency > 0.1 else 0
            return recall - (0.05 * latency_penalty)  # Focus more on recall
        
        # Sort by score function
        sorted_results = sorted(
            [(search_ef, result) for search_ef, result in results.items()],
            key=lambda x: score_fn(x[1]),
            reverse=True
        )
        
        # Get best parameters
        if sorted_results:
            best_search_ef, best_result = sorted_results[0]
            optimal_parameters = best_result["parameters"]
            
            # Save optimization results
            self._save_optimization_results(results, best_search_ef)
            
            logger.info(f"Optimal parameters found: search_ef={best_search_ef}")
            logger.info(f"Metrics: recall={best_result['metrics']['recall']:.4f}, "
                       f"ndcg={best_result['metrics']['ndcg']:.4f}, "
                       f"latency={best_result['metrics']['latency']:.4f}s")
            
            # Apply optimal parameters to collection
            collection = self.db_manager.get_collection()
            try:
                collection.modify(metadata=optimal_parameters)
                logger.info("Applied optimal parameters to collection")
            except Exception as e:
                logger.error(f"Failed to apply optimal parameters: {e}")
            
            return optimal_parameters
        else:
            logger.warning("No valid results from parameter optimization")
            return {}
    
    def _save_optimization_results(self, results: Dict[str, Any], best_search_ef: int) -> None:
        """Save optimization results to disk."""
        try:
            output = {
                "timestamp": time.time(),
                "best_search_ef": best_search_ef,
                "results": results
            }
            
            with open(self.results_file, 'w') as f:
                json.dump(output, f, indent=2)
            
            logger.info(f"Saved optimization results to {self.results_file}")
        except Exception as e:
            logger.error(f"Failed to save optimization results: {e}")
    
    def create_validation_set_from_test_queries(self, count: int = 50) -> int:
        """Create a validation set from the test query suite.
        
        This uses the test query suite to create a validation set,
        simulating ground truth by running searches and using top results.
        
        Args:
            count: Number of queries to include in validation set
            
        Returns:
            Number of queries added to validation set
        """
        # Import test queries
        from scripts.search.query_processor import QueryProcessor
        
        try:
            # Try to import test suite queries
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            from docs.prompt_test_suite import (
                TENSOR_CODE_QUERIES, TENSOR_CONCEPT_QUERIES,
                NN_MODULE_CODE_QUERIES, NN_MODULE_CONCEPT_QUERIES,
                OPTIM_CODE_QUERIES, OPTIM_CONCEPT_QUERIES,
                DATA_QUERIES, DEPLOYMENT_QUERIES
            )
            
            # Sample queries from different categories
            import random
            all_queries = (
                TENSOR_CODE_QUERIES + TENSOR_CONCEPT_QUERIES +
                NN_MODULE_CODE_QUERIES + NN_MODULE_CONCEPT_QUERIES +
                OPTIM_CODE_QUERIES + OPTIM_CONCEPT_QUERIES +
                DATA_QUERIES + DEPLOYMENT_QUERIES
            )
            
            # Shuffle and select a subset
            random.shuffle(all_queries)
            selected_queries = all_queries[:min(count, len(all_queries))]
            
            query_processor = QueryProcessor()
            added_count = 0
            
            # For each query, run a search and use top results as "ground truth"
            for query in selected_queries:
                try:
                    # Process query
                    query_data = query_processor.process_query(query)
                    
                    # Search with default parameters (temporarily)
                    results = self.db_manager.query(query_data["embedding"], n_results=5)
                    
                    # Extract relevant documents from results
                    if isinstance(results.get('metadatas'), list) and len(results['metadatas']) > 0:
                        # Flatten if needed
                        metadatas = results['metadatas'][0] if isinstance(results['metadatas'][0], list) else results['metadatas']
                        relevant_docs = [meta.get("source", "") for meta in metadatas if isinstance(meta, dict)]
                    else:
                        # Handle empty results
                        relevant_docs = []
                    
                    # Only add if we have some relevant docs
                    if relevant_docs:
                        self.add_validation_query(query, relevant_docs)
                        added_count += 1
                
                except Exception as e:
                    logger.error(f"Error processing query '{query}': {e}")
            
            logger.info(f"Added {added_count} queries to validation set")
            return added_count
            
        except ImportError as e:
            logger.error(f"Failed to import test queries: {e}")
            return 0

# For command-line usage
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    from scripts.database.chroma_manager import ChromaManager
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimize HNSW parameters for ChromaDB")
    parser.add_argument("--create-validation", action="store_true", help="Create validation set from test queries")
    parser.add_argument("--validation-file", type=str, help="JSON file with validation queries")
    parser.add_argument("--optimize", action="store_true", help="Run optimization")
    parser.add_argument("--collection", type=str, default="pytorch_docs", help="ChromaDB collection name")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize components
    db_manager = ChromaManager(collection_name=args.collection)
    optimizer = HNSWOptimizer(db_manager)
    
    if args.create_validation:
        optimizer.create_validation_set_from_test_queries(50)
    
    if args.validation_file:
        optimizer.add_validation_queries_from_file(args.validation_file)
    
    if args.optimize:
        optimal_params = optimizer.optimize_hnsw_parameters()
        print(f"Optimal parameters: {optimal_params}")
    
    # If no arguments provided, show help
    if not (args.create_validation or args.validation_file or args.optimize):
        parser.print_help()