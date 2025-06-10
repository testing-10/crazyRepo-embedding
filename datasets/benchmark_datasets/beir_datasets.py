"""
BEIR (Benchmarking IR) Dataset Loader

Loads datasets from the BEIR benchmark suite for comprehensive
information retrieval evaluation across multiple domains.
"""

import os
import requests
import json
from typing import List, Tuple, Dict, Optional

from ..dataset_loader import BenchmarkDatasetLoader, DatasetSample, DatasetInfo
from utils.embedding_logger import EmbeddingLogger

logger = EmbeddingLogger.get_logger(__name__)

class BEIRDatasetLoader(BenchmarkDatasetLoader):
    """Loader for BEIR benchmark datasets"""
    
    BASE_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/"
    DATASET_NAME = "beir"
    
    # Available BEIR datasets
    AVAILABLE_DATASETS = {
        'trec-covid': 'COVID-19 related scientific articles',
        'nfcorpus': 'Nutrition facts corpus',
        'nq': 'Natural Questions',
        'hotpotqa': 'HotpotQA multi-hop reasoning',
        'fiqa': 'Financial opinion mining',
        'arguana': 'Argument retrieval',
        'touche-2020': 'Argument retrieval (Touché 2020)',
        'cqadupstack': 'CQADupStack community question answering',
        'quora': 'Quora duplicate questions',
        'dbpedia-entity': 'DBpedia entity retrieval',
        'scidocs': 'Scientific document retrieval',
        'fever': 'Fact extraction and verification',
        'climate-fever': 'Climate change fact verification',
        'scifact': 'Scientific claim verification'
    }
    
    def __init__(self, dataset_name: str = 'nfcorpus', cache_dir: str = "./cache/datasets/"):
        super().__init__(cache_dir)
        self.beir_dataset = dataset_name
        self.dataset_path = self._get_cache_path(f"{self.DATASET_NAME}_{dataset_name}")
        
        if dataset_name not in self.AVAILABLE_DATASETS:
            logger.warning(f"Dataset {dataset_name} not in known BEIR datasets")
    
    def download(self) -> bool:
        """Download specified BEIR dataset"""
        try:
            if self._check_cache(f"{self.DATASET_NAME}_{self.beir_dataset}"):
                logger.info(f"BEIR {self.beir_dataset} already cached")
                return True
                
            logger.info(f"Downloading BEIR {self.beir_dataset} dataset...")
            
            # Download the zip file
            zip_url = f"{self.BASE_URL}{self.beir_dataset}.zip"
            response = requests.get(zip_url, stream=True)
            response.raise_for_status()
            
            zip_path = os.path.join(self.dataset_path, f"{self.beir_dataset}.zip")
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Extract the archive
            import zipfile
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.dataset_path)
            
            # Clean up zip file
            os.remove(zip_path)
            
            logger.info(f"BEIR {self.beir_dataset} downloaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading BEIR {self.beir_dataset}: {e}")
            return False
    
    def load(self, split: str = "test", **kwargs) -> Tuple[List[DatasetSample], DatasetInfo]:
        """Load BEIR dataset"""
        try:
            # Download if not cached
            if not self.download():
                raise RuntimeError(f"Failed to download BEIR {self.beir_dataset}")
            
            # Find the dataset directory
            dataset_dir = None
            for root, dirs, files in os.walk(self.dataset_path):
                if 'queries.jsonl' in files and 'corpus.jsonl' in files:
                    dataset_dir = root
                    break
            
            if not dataset_dir:
                raise FileNotFoundError(f"BEIR {self.beir_dataset} dataset files not found")
            
            # Load queries
            queries = self._load_jsonl(os.path.join(dataset_dir, 'queries.jsonl'))
            
            # Load corpus
            corpus = self._load_jsonl(os.path.join(dataset_dir, 'corpus.jsonl'))
            
            # Load qrels
            qrels_file = os.path.join(dataset_dir, f'qrels/{split}.tsv')
            if not os.path.exists(qrels_file):
                # Try alternative paths
                for alt_split in ['test', 'dev', 'train']:
                    alt_file = os.path.join(dataset_dir, f'qrels/{alt_split}.tsv')
                    if os.path.exists(alt_file):
                        qrels_file = alt_file
                        logger.info(f"Using {alt_split} split instead of {split}")
                        break
            
            qrels = self._load_qrels(qrels_file)
            
            # Create samples
            samples = []
            
            for query_id, query_data in queries.items():
                if query_id in qrels:
                    query_text = query_data.get('text', '')
                    
                    for doc_id, relevance in qrels[query_id].items():
                        if doc_id in corpus and relevance > 0:
                            doc_data = corpus[doc_id]
                            doc_text = doc_data.get('text', '')
                            
                            # Combine title and text if available
                            if 'title' in doc_data and doc_data['title']:
                                doc_text = f"{doc_data['title']} {doc_text}"
                            
                            samples.append(DatasetSample(
                                text1=query_text,
                                text2=doc_text,
                                label=relevance,
                                metadata={
                                    'query_id': query_id,
                                    'doc_id': doc_id,
                                    'task': 'retrieval',
                                    'dataset': self.beir_dataset,
                                    'beir_suite': True
                                }
                            ))
            
            # Validate samples
            samples = self._validate_samples(samples)
            
            dataset_info = DatasetInfo(
                name=f"BEIR {self.beir_dataset}",
                task_type="retrieval",
                num_samples=len(samples),
                description=f"BEIR benchmark: {self.AVAILABLE_DATASETS.get(self.beir_dataset, 'Unknown dataset')}",
                source="https://github.com/beir-cellar/beir",
                citation="@inproceedings{thakur2021beir, title={BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models}, author={Thakur, Nandan and Reimers, Nils and Rücklé, Andreas and Srivastava, Abhishek and Gurevych, Iryna}, booktitle={Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2)}, year={2021}}",
                languages=["en"],
                domains=[self.beir_dataset]
            )
            
            logger.info(f"Loaded {len(samples)} BEIR {self.beir_dataset} samples ({split} split)")
            return samples, dataset_info
            
        except Exception as e:
            logger.error(f"Error loading BEIR {self.beir_dataset}: {e}")
            raise
    
    def _load_jsonl(self, file_path: str) -> Dict[str, Dict]:
        """Load JSONL file into dictionary"""
        data = {}
        
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            return data
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    data[item['_id']] = item
                except json.JSONDecodeError as e:
                    logger.warning(f"Error parsing JSON line: {e}")
                    continue
        
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
        """Get BEIR dataset information"""
        return DatasetInfo(
            name=f"BEIR {self.beir_dataset}",
            task_type="retrieval",
            num_samples=0,  # Will be updated when loaded
            description=f"BEIR benchmark: {self.AVAILABLE_DATASETS.get(self.beir_dataset, 'Unknown dataset')}",
            source="https://github.com/beir-cellar/beir",
            languages=["en"],
            domains=[self.beir_dataset]
        )
    
    @classmethod
    def list_available_datasets(cls) -> Dict[str, str]:
        """List all available BEIR datasets"""
        return cls.AVAILABLE_DATASETS.copy()
