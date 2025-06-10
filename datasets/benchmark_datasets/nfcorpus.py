"""
NFCorpus Dataset Loader

Loads the NFCorpus (Nutrition Facts Corpus) dataset for domain-specific
information retrieval evaluation in the nutrition domain.
"""

import os
import requests
import json
from typing import List, Tuple, Dict

from ..dataset_loader import BenchmarkDatasetLoader, DatasetSample, DatasetInfo
from utils.embedding_logger import EmbeddingLogger

logger = EmbeddingLogger.get_logger(__name__)

class NFCorpusLoader(BenchmarkDatasetLoader):
    """Loader for NFCorpus dataset"""
    
    BASE_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/"
    DATASET_NAME = "nfcorpus"
    
    def __init__(self, cache_dir: str = "./cache/datasets/"):
        super().__init__(cache_dir)
        self.dataset_path = self._get_cache_path(self.DATASET_NAME)
        
    def download(self) -> bool:
        """Download NFCorpus dataset"""
        try:
            if self._check_cache(self.DATASET_NAME):
                logger.info("NFCorpus already cached")
                return True
                
            logger.info("Downloading NFCorpus dataset...")
            
            # Download the zip file
            zip_url = f"{self.BASE_URL}nfcorpus.zip"
            response = requests.get(zip_url, stream=True)
            response.raise_for_status()
            
            zip_path = os.path.join(self.dataset_path, "nfcorpus.zip")
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Extract the archive
            import zipfile
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.dataset_path)
            
            # Clean up zip file
            os.remove(zip_path)
            
            logger.info("NFCorpus downloaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading NFCorpus: {e}")
            return False
    
    def load(self, split: str = "test", **kwargs) -> Tuple[List[DatasetSample], DatasetInfo]:
        """Load NFCorpus dataset"""
        try:
            # Download if not cached
            if not self.download():
                raise RuntimeError("Failed to download NFCorpus")
            
            # Find the dataset directory
            dataset_dir = None
            for root, dirs, files in os.walk(self.dataset_path):
                if 'queries.jsonl' in files and 'corpus.jsonl' in files:
                    dataset_dir = root
                    break
            
            if not dataset_dir:
                raise FileNotFoundError("NFCorpus dataset files not found")
            
            # Load queries
            queries = self._load_jsonl(os.path.join(dataset_dir, 'queries.jsonl'))
            
            # Load corpus
            corpus = self._load_jsonl(os.path.join(dataset_dir, 'corpus.jsonl'))
            
            # Load qrels
            qrels_file = os.path.join(dataset_dir, f'qrels/{split}.tsv')
            if not os.path.exists(qrels_file):
                qrels_file = os.path.join(dataset_dir, 'qrels/test.tsv')  # Fallback
            
            qrels = self._load_qrels(qrels_file)
            
            # Create samples
            samples = []
            
            for query_id, query_data in queries.items():
                if query_id in qrels:
                    query_text = query_data.get('text', '')
                    
                    for doc_id, relevance in qrels[query_id].items():
                        if doc_id in corpus and relevance > 0:
                            doc_text = corpus[doc_id].get('text', '')
                            
                            samples.append(DatasetSample(
                                text1=query_text,
                                text2=doc_text,
                                label=relevance,
                                metadata={
                                    'query_id': query_id,
                                    'doc_id': doc_id,
                                    'task': 'retrieval',
                                    'domain': 'nutrition'
                                }
                            ))
            
            # Validate samples
            samples = self._validate_samples(samples)
            
            dataset_info = DatasetInfo(
                name="NFCorpus",
                task_type="retrieval",
                num_samples=len(samples),
                description="Nutrition Facts Corpus for domain-specific information retrieval",
                source="https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/",
                citation="@inproceedings{boteva2016full, title={A full-text learning to rank dataset for medical information retrieval}, author={Boteva, Vera and Gholipour, Demian and Sokolov, Artem and Riezler, Stefan}, booktitle={European Conference on Information Retrieval}, pages={716--722}, year={2016}, organization={Springer}}",
                languages=["en"],
                domains=["nutrition", "medical", "health"]
            )
            
            logger.info(f"Loaded {len(samples)} NFCorpus samples ({split} split)")
            return samples, dataset_info
            
        except Exception as e:
            logger.error(f"Error loading NFCorpus: {e}")
            raise
    
    def _load_jsonl(self, file_path: str) -> Dict[str, Dict]:
        """Load JSONL file into dictionary"""
        data = {}
        
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            return data
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                data[item['_id']] = item
        
        logger.info(f"Loaded {len(data)} items from {os.path.basename(file_path)}")
        return data
    
    def _load_qrels(self, file_path: str) -> Dict[str, Dict[str, int]]:
        """Load qrels file"""
        qrels = {}
        
        if not os.path.exists(file_path):
            logger.warning(f"Qrels file not found: {file_path}")
            return qrels
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 4:
                    query_id, _, doc_id, relevance = parts
                    if query_id not in qrels:
                        qrels[query_id] = {}
                    qrels[query_id][doc_id] = int(relevance)
        
        logger.info(f"Loaded qrels for {len(qrels)} queries")
        return qrels
    
    def get_info(self) -> DatasetInfo:
        """Get NFCorpus dataset information"""
        return DatasetInfo(
            name="NFCorpus",
            task_type="retrieval",
            num_samples=0,  # Will be updated when loaded
            description="Nutrition Facts Corpus for domain-specific information retrieval",
            source="https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/",
            languages=["en"],
            domains=["nutrition", "medical", "health"]
        )
