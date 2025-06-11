"""
Retrieval Evaluator

Evaluates embedding models on information retrieval tasks using
ranking metrics like precision@k, recall@k, MAP, and NDCG.
"""

import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

from evaluators.base_embedding_evaluator import BaseEmbeddingEvaluator, EvaluationResult
from datasets.dataset_loader import DatasetSample
from utils.vector_operations import VectorOperations

class RetrievalEvaluator(BaseEmbeddingEvaluator):
    """Evaluator for information retrieval tasks"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.vector_ops = VectorOperations()
        
        # Default configuration
        self.default_config = {
            'k_values': [1, 3, 5, 10, 20],
            'metrics': ['precision_at_k', 'recall_at_k', 'map_at_k', 'ndcg_at_k'],
            'similarity_metric': 'cosine',
            'normalize_embeddings': True,
            'max_corpus_size': 10000,  # Limit corpus size for performance
            'relevance_threshold': 0.5  # For binary relevance
        }
        
        # Merge with provided config
        self.config = {**self.default_config, **self.config}
    
    def evaluate(self, 
                 embeddings: List[np.ndarray], 
                 samples: List[DatasetSample],
                 model_name: str,
                 dataset_name: str,
                 **kwargs) -> EvaluationResult:
        """Evaluate retrieval performance"""
        
        start_time = time.time()
        self._log_evaluation_start(model_name, dataset_name, len(samples))
        
        try:
            # Validate inputs
            if not self._validate_inputs(embeddings, samples):
                raise ValueError("Input validation failed")
            
            # Prepare embeddings and filter samples
            valid_embeddings, valid_indices = self._prepare_embeddings(embeddings)
            valid_samples = self._filter_samples(samples, valid_indices)
            
            # Organize data for retrieval evaluation
            queries, corpus, relevance_judgments = self._organize_retrieval_data(valid_embeddings, valid_samples)
            
            if len(queries) == 0:
                raise ValueError("No valid queries found for retrieval evaluation")
            
            # Perform retrieval evaluation
            metrics = self._evaluate_retrieval(queries, corpus, relevance_judgments)
            
            # Generate metadata
            metadata = self._generate_metadata(queries, corpus, relevance_judgments, valid_samples)
            
            execution_time = time.time() - start_time
            self._log_evaluation_end(metrics, execution_time)
            
            return self._create_result(
                task_type="retrieval",
                model_name=model_name,
                dataset_name=dataset_name,
                metrics=metrics,
                metadata=metadata,
                execution_time=execution_time,
                num_samples=len(valid_samples)
            )
            
        except Exception as e:
            self.logger.error(f"Retrieval evaluation failed: {e}")
            raise
    
    def _organize_retrieval_data(self, embeddings: np.ndarray, samples: List[DatasetSample]) -> Tuple[Dict, Dict, Dict]:
        """Organize samples into queries, corpus, and relevance judgments"""
        queries = {}
        corpus = {}
        relevance_judgments = defaultdict(dict)
        
        # Strategy 1: Use samples with query_id and doc_id metadata
        for i, sample in enumerate(samples):
            if sample.metadata:
                query_id = sample.metadata.get('query_id')
                doc_id = sample.metadata.get('doc_id') or sample.metadata.get('passage_id')
                
                if query_id and doc_id:
                    # This is a query-document pair
                    if query_id not in queries:
                        queries[query_id] = {
                            'text': sample.text1,
                            'embedding': embeddings[i]
                        }
                    
                    if doc_id not in corpus:
                        corpus[doc_id] = {
                            'text': sample.text2 or sample.text1,
                            'embedding': embeddings[i]
                        }
                    
                    # Store relevance judgment
                    relevance = sample.label if sample.label is not None else 1
                    relevance_judgments[query_id][doc_id] = float(relevance)
        
        # Strategy 2: If no query/doc structure, create artificial retrieval setup
        if not queries:
            self.logger.info("No query-document structure found, creating artificial retrieval setup")
            
            # Use first half as queries, second half as corpus
            mid_point = len(embeddings) // 2
            
            for i in range(mid_point):
                query_id = f"q_{i}"
                queries[query_id] = {
                    'text': samples[i].text1,
                    'embedding': embeddings[i]
                }
                
                # Create relevance judgments based on similarity or labels
                for j in range(mid_point, len(embeddings)):
                    doc_id = f"d_{j}"
                    
                    if doc_id not in corpus:
                        corpus[doc_id] = {
                            'text': samples[j].text1,
                            'embedding': embeddings[j]
                        }
                    
                    # Use label similarity or domain matching for relevance
                    relevance = self._calculate_artificial_relevance(samples[i], samples[j])
                    if relevance > 0:
                        relevance_judgments[query_id][doc_id] = relevance
        
        # Limit corpus size for performance
        if len(corpus) > self.config['max_corpus_size']:
            corpus_items = list(corpus.items())[:self.config['max_corpus_size']]
            corpus = dict(corpus_items)
            self.logger.info(f"Limited corpus to {self.config['max_corpus_size']} documents")
        
        return queries, corpus, dict(relevance_judgments)
    
    def _calculate_artificial_relevance(self, query_sample: DatasetSample, doc_sample: DatasetSample) -> float:
        """Calculate artificial relevance for samples without explicit query-doc structure"""
        relevance = 0.0
        
        # Domain matching
        if (query_sample.metadata and doc_sample.metadata and 
            query_sample.metadata.get('domain') == doc_sample.metadata.get('domain')):
            relevance += 0.5
        
        # Label similarity (for similarity datasets)
        if query_sample.label is not None and doc_sample.label is not None:
            label_diff = abs(float(query_sample.label) - float(doc_sample.label))
            if label_diff < 0.2:  # Similar labels
                relevance += 0.5
        
        # Text overlap (simple heuristic)
        if query_sample.text1 and doc_sample.text1:
            query_words = set(query_sample.text1.lower().split())
            doc_words = set(doc_sample.text1.lower().split())
            overlap = len(query_words.intersection(doc_words))
            if overlap > 2:
                relevance += 0.3
        
        return min(relevance, 1.0)
    
    def _evaluate_retrieval(self, queries: Dict, corpus: Dict, relevance_judgments: Dict) -> Dict[str, float]:
        """Evaluate retrieval performance using various metrics"""
        metrics = {}
        
        # Prepare corpus embeddings
        corpus_ids = list(corpus.keys())
        corpus_embeddings = np.array([corpus[doc_id]['embedding'] for doc_id in corpus_ids])
        
        # Normalize embeddings if configured
        if self.config['normalize_embeddings']:
            corpus_embeddings = self.vector_ops.normalize_vectors(corpus_embeddings)
        
        all_precisions = {k: [] for k in self.config['k_values']}
        all_recalls = {k: [] for k in self.config['k_values']}
        all_aps = []  # Average Precision scores
        all_ndcgs = {k: [] for k in self.config['k_values']}
        
        for query_id, query_data in queries.items():
            if query_id not in relevance_judgments:
                continue
            
            query_embedding = query_data['embedding']
            if self.config['normalize_embeddings']:
                query_embedding = self.vector_ops.normalize_vectors(query_embedding.reshape(1, -1))[0]
            
            # Calculate similarities
            similarities = cosine_similarity([query_embedding], corpus_embeddings)[0]
            
            # Rank documents by similarity
            ranked_indices = np.argsort(similarities)[::-1]  # Descending order
            ranked_doc_ids = [corpus_ids[i] for i in ranked_indices]
            
            # Get relevance judgments for this query
            query_relevances = relevance_judgments[query_id]
            
            # Calculate metrics for different k values
            for k in self.config['k_values']:
                if k <= len(ranked_doc_ids):
                    # Precision@k and Recall@k
                    precision_k, recall_k = self._calculate_precision_recall_at_k(
                        ranked_doc_ids[:k], query_relevances
                    )
                    all_precisions[k].append(precision_k)
                    all_recalls[k].append(recall_k)
                    
                    # NDCG@k
                    ndcg_k = self._calculate_ndcg_at_k(ranked_doc_ids[:k], query_relevances)
                    all_ndcgs[k].append(ndcg_k)
            
            # Average Precision (for MAP calculation)
            ap = self._calculate_average_precision(ranked_doc_ids, query_relevances)
            all_aps.append(ap)
        
        # Aggregate metrics
        for k in self.config['k_values']:
            if all_precisions[k]:
                metrics[f'precision_at_{k}'] = float(np.mean(all_precisions[k]))
                metrics[f'recall_at_{k}'] = float(np.mean(all_recalls[k]))
                metrics[f'ndcg_at_{k}'] = float(np.mean(all_ndcgs[k]))
        
        if all_aps:
            metrics['map'] = float(np.mean(all_aps))  # Mean Average Precision
        
        # Additional metrics
        metrics['num_queries'] = len(queries)
        metrics['num_documents'] = len(corpus)
        
        return metrics
    
    def _calculate_precision_recall_at_k(self, ranked_docs: List[str], relevances: Dict[str, float]) -> Tuple[float, float]:
        """Calculate Precision@k and Recall@k"""
        relevant_retrieved = 0
        total_relevant = sum(1 for rel in relevances.values() if rel > self.config['relevance_threshold'])
        
        for doc_id in ranked_docs:
            if doc_id in relevances and relevances[doc_id] > self.config['relevance_threshold']:
                relevant_retrieved += 1
        
        precision = relevant_retrieved / len(ranked_docs) if ranked_docs else 0.0
        recall = relevant_retrieved / total_relevant if total_relevant > 0 else 0.0
        
        return precision, recall
    
    def _calculate_ndcg_at_k(self, ranked_docs: List[str], relevances: Dict[str, float]) -> float:
        """Calculate Normalized Discounted Cumulative Gain at k"""
        def dcg(relevance_scores):
            return sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance_scores))
        
        # Actual DCG
        actual_relevances = [relevances.get(doc_id, 0.0) for doc_id in ranked_docs]
        actual_dcg = dcg(actual_relevances)
        
        # Ideal DCG (best possible ranking)
        ideal_relevances = sorted(relevances.values(), reverse=True)[:len(ranked_docs)]
        ideal_dcg = dcg(ideal_relevances)
        
        return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0
    
    def _calculate_average_precision(self, ranked_docs: List[str], relevances: Dict[str, float]) -> float:
        """Calculate Average Precision for a single query"""
        relevant_count = 0
        precision_sum = 0.0
        
        for i, doc_id in enumerate(ranked_docs):
            if doc_id in relevances and relevances[doc_id] > self.config['relevance_threshold']:
                relevant_count += 1
                precision_at_i = relevant_count / (i + 1)
                precision_sum += precision_at_i
        
        total_relevant = sum(1 for rel in relevances.values() if rel > self.config['relevance_threshold'])
        return precision_sum / total_relevant if total_relevant > 0 else 0.0
    
    def _generate_metadata(self, queries: Dict, corpus: Dict, relevance_judgments: Dict, 
                          samples: List[DatasetSample]) -> Dict[str, Any]:
        """Generate retrieval evaluation metadata"""
        metadata = {
            'num_queries': len(queries),
            'num_documents': len(corpus),
            'num_query_doc_pairs': sum(len(rels) for rels in relevance_judgments.values()),
            'k_values_evaluated': self.config['k_values'],
            'similarity_metric': self.config['similarity_metric'],
            'config': self.config.copy()
        }
        
        # Relevance statistics
        all_relevances = []
        for rels in relevance_judgments.values():
            all_relevances.extend(rels.values())
        
        if all_relevances:
            metadata['relevance_stats'] = {
                'min': float(np.min(all_relevances)),
                'max': float(np.max(all_relevances)),
                'mean': float(np.mean(all_relevances)),
                'std': float(np.std(all_relevances))
            }
        
        # Query length statistics
        query_lengths = [len(q['text'].split()) for q in queries.values()]
        if query_lengths:
            metadata['query_length_stats'] = {
                'min': int(np.min(query_lengths)),
                'max': int(np.max(query_lengths)),
                'mean': float(np.mean(query_lengths))
            }
        
        # Document length statistics
        doc_lengths = [len(d['text'].split()) for d in corpus.values()]
        if doc_lengths:
            metadata['document_length_stats'] = {
                'min': int(np.min(doc_lengths)),
                'max': int(np.max(doc_lengths)),
                'mean': float(np.mean(doc_lengths))
            }
        
        return metadata
    
    def get_required_metrics(self) -> List[str]:
        """Get list of metrics this evaluator provides"""
        base_metrics = ['precision_at_k', 'recall_at_k', 'ndcg_at_k', 'map']
        k_specific_metrics = []
        
        for k in self.config['k_values']:
            k_specific_metrics.extend([
                f'precision_at_{k}',
                f'recall_at_{k}',
                f'ndcg_at_{k}'
            ])
        
        return base_metrics + k_specific_metrics
