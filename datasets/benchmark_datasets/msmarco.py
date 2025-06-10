"""
MS MARCO Dataset Loader

Loads MS MARCO passage ranking dataset for evaluating retrieval performance.
"""

import os
import requests
import gzip
import json
from typing import List, Tuple, Dict
from collections import defaultdict

from ..dataset_loader import BenchmarkDatasetLoader, DatasetSample, DatasetInfo
from utils.embedding_logger import EmbeddingLogger

logger = EmbeddingLogger.get_logger(__name__)

class MSMARCOLoader(BenchmarkDatasetLoader):
    """Loader for MS MARCO passage ranking dataset"""
    
    BASE_URL = "https://msmarco.blob.core.windows.net/msmarcoranking/"
    DATASET_NAME = "msmarco"
    
    # File URLs
    FILES = {
        "queries_train": "queries.train.tsv",
        "queries_dev": "queries.dev.tsv", 
        "queries_eval": "queries.eval.tsv",
        "collection": "collection.tsv",
        "qrels_train": "qrels.train.tsv",
        "qrels_dev": "qrels.dev.tsv",
        "top1000_dev": "top1000.dev.tsv",
        "top1000_eval": "top1000.eval.tsv"
    }
    
    def __init__(self, cache_dir: str = "./cache/datasets/", subset_size: int = None):
        super().__init__(cache_dir)
        self.dataset_path = self._get_cache_path(self.DATASET_NAME)
        self.subset_size = subset_size  # Limit dataset size for testing
        
    def download(self) -> bool:
        """Download MS MARCO dataset files"""
        try:
            if self._check_cache(self.DATASET_NAME):
                logger.info("MS MARCO already cached")
                return True
                
            logger.info("Downloading MS MARCO dataset...")
            
            # Download essential files
            essential_files = ["queries_dev", "collection", "qrels_dev", "top1000_dev"]
            
            for file_key in essential_files:
                filename = self.FILES[file_key]
                url = self.BASE_URL + filename
                file_path = os.path.join(self.dataset_path, filename)
                
                if os.path.exists(file_path):
                    logger.info(f"File {filename} already exists")
                    continue
                
                logger.info(f"Downloading {filename}...")
                response = requests.get(url, stream=True)
                response.raise_for_status()
                
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                logger.info(f"Downloaded {filename}")
            
            logger.info("MS MARCO dataset downloaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading MS MARCO: {e}")
            return False
    
    def load(self, split: str = "dev", **kwargs) -> Tuple[List[DatasetSample], DatasetInfo]:
        """Load MS MARCO dataset"""
        try:
            # Download if not cached
            if not self.download():
                raise RuntimeError("Failed to download MS MARCO")
            
            # Load queries
            queries = self._load_queries(split)
            
            # Load collection (passages)
            collection = self._load_collection()
            
            # Load relevance judgments
            qrels = self._load_qrels(split)
            
            # Create samples for retrieval evaluation
            samples = []
            
            for query_id, query_text in queries.items():
                if query_id in qrels:
                    relevant_passages = qrels[query_id]
                    
                    # Create positive samples
                    for passage_id in relevant_passages:
                        if passage_id in collection:
                            samples.append(DatasetSample(
                                text1=query_text,
                                text2=collection[passage_id],
                                label=1,  # Relevant
                                metadata={
                                    'query_id': query_id,
                                    'passage_id': passage_id,
                                    'task': 'retrieval'
                                }
                            ))
                
                # Limit dataset size if specified
                if self.subset_size and len(samples) >= self.subset_size:
                    break
            
            # Validate samples
            samples = self._validate_samples(samples)
            
            dataset_info = DatasetInfo(
                name="MS MARCO Passage Ranking",
                task_type="retrieval",
                num_samples=len(samples),
                description="Large-scale information retrieval dataset with real user queries",
                source="https://microsoft.github.io/msmarco/",
                citation="@article{bajaj2016ms, title={MS MARCO: A human generated machine reading comprehension dataset}, author={Bajaj, Payal and Campos, Daniel and Craswell, Nick and Deng, Li and Gao, Jianfeng and Liu, Xiaodong and Mittal, Rangan and Nouri, Paul and Pantel, Patrick and Sauper, Christina and others}, journal={arXiv preprint arXiv:1611.09268}, year={2016}}",
                languages=["en"],
                domains=["web", "general"]
            )
            
            logger.info(f"Loaded {len(samples)} MS MARCO samples ({split} split)")
            return samples, dataset_info
            
        except Exception as e:
            logger.error(f"Error loading MS MARCO: {e}")
            raise
    
    def _load_queries(self, split: str) -> Dict[str, str]:
        """Load queries from TSV file"""
        queries = {}
        filename = f"queries.{split}.tsv"
        file_path = os.path.join(self.dataset_path, filename)
        
        if not os.path.exists(file_path):
            logger.warning(f"Queries file not found: {filename}")
            return queries
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    query_id, query_text = parts[0], parts[1]
                    queries[query_id] = query_text
        
        logger.info(f"Loaded {len(queries)} queries from {filename}")
        return queries
    
    def _load_collection(self) -> Dict[str, str]:
        """Load passage collection from TSV file"""
        collection = {}
        file_path = os.path.join(self.dataset_path, "collection.tsv")
        
        if not os.path.exists(file_path):
            logger.warning("Collection file not found")
            return collection
        
        count = 0
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    passage_id, passage_text = parts[0], parts[1]
                    collection[passage_id] = passage_text
                    count += 1
                    
                    # Limit collection size for memory efficiency
                    if self.subset_size and count >= self.subset_size * 10:
                        break
        
        logger.info(f"Loaded {len(collection)} passages from collection")
        return collection
    
    def _load_qrels(self, split: str) -> Dict[str, List[str]]:
        """Load relevance judgments"""
        qrels = defaultdict(list)
        filename = f"qrels.{split}.tsv"
        file_path = os.path.join(self.dataset_path, filename)
        
        if not os.path.exists(file_path):
            logger.warning(f"Qrels file not found: {filename}")
            return dict(qrels)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 4:
                    query_id, _, passage_id, relevance = parts
                    if int(relevance) > 0:  # Only relevant passages
                        qrels[query_id].append(passage_id)
        
        logger.info(f"Loaded qrels for {len(qrels)} queries from {filename}")
        return dict(qrels)
    
    def get_info(self) -> DatasetInfo:
        """Get MS MARCO dataset information"""
        return DatasetInfo(
            name="MS MARCO Passage Ranking",
            task_type="retrieval",
            num_samples=0,  # Will be updated when loaded
            description="Large-scale information retrieval dataset with real user queries",
            source="https://microsoft.github.io/msmarco/",
            languages=["en"],
            domains=["web", "general"]
        )
