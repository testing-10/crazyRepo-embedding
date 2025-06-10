"""
STS (Semantic Textual Similarity) Benchmark Dataset Loader

Loads the STS benchmark dataset for evaluating semantic similarity
between sentence pairs with human-annotated similarity scores.
"""

import os
import requests
import zipfile
import pandas as pd
from typing import List, Tuple
from urllib.parse import urljoin

from ..dataset_loader import BenchmarkDatasetLoader, DatasetSample, DatasetInfo
from utils.embedding_logger import EmbeddingLogger

logger = EmbeddingLogger.get_logger(__name__)

class STSBenchmarkLoader(BenchmarkDatasetLoader):
    """Loader for STS Benchmark dataset"""
    
    DATASET_URL = "https://ixa2.si.ehu.eus/stswiki/images/4/48/Stsbenchmark.tar.gz"
    DATASET_NAME = "sts_benchmark"
    
    def __init__(self, cache_dir: str = "./cache/datasets/"):
        super().__init__(cache_dir)
        self.dataset_path = self._get_cache_path(self.DATASET_NAME)
        
    def download(self) -> bool:
        """Download STS benchmark dataset"""
        try:
            if self._check_cache(self.DATASET_NAME):
                logger.info("STS benchmark already cached")
                return True
                
            logger.info("Downloading STS benchmark dataset...")
            
            # Download the tar.gz file
            response = requests.get(self.DATASET_URL, stream=True)
            response.raise_for_status()
            
            tar_path = os.path.join(self.dataset_path, "stsbenchmark.tar.gz")
            with open(tar_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Extract the archive
            import tarfile
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(self.dataset_path)
            
            # Clean up tar file
            os.remove(tar_path)
            
            logger.info("STS benchmark downloaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading STS benchmark: {e}")
            return False
    
    def load(self, split: str = "test", **kwargs) -> Tuple[List[DatasetSample], DatasetInfo]:
        """Load STS benchmark dataset"""
        try:
            # Download if not cached
            if not self.download():
                raise RuntimeError("Failed to download STS benchmark")
            
            # Find the data file
            data_file = None
            for root, dirs, files in os.walk(self.dataset_path):
                for file in files:
                    if f"sts-{split}.csv" in file:
                        data_file = os.path.join(root, file)
                        break
                if data_file:
                    break
            
            if not data_file:
                raise FileNotFoundError(f"STS {split} split not found")
            
            # Load the data
            df = pd.read_csv(data_file, sep='\t', header=None, 
                           names=['genre', 'filename', 'year', 'id', 'score', 'sentence1', 'sentence2'])
            
            samples = []
            for _, row in df.iterrows():
                if pd.notna(row['sentence1']) and pd.notna(row['sentence2']):
                    samples.append(DatasetSample(
                        text1=str(row['sentence1']).strip(),
                        text2=str(row['sentence2']).strip(),
                        label=float(row['score']) if pd.notna(row['score']) else None,
                        metadata={
                            'genre': row['genre'],
                            'filename': row['filename'],
                            'year': row['year'],
                            'id': row['id']
                        }
                    ))
            
            # Validate samples
            samples = self._validate_samples(samples)
            
            dataset_info = DatasetInfo(
                name="STS Benchmark",
                task_type="similarity",
                num_samples=len(samples),
                description="Semantic Textual Similarity benchmark with human-annotated similarity scores (0-5)",
                source="https://ixa2.si.ehu.eus/stswiki/index.php/STSbenchmark",
                citation="@inproceedings{cer2017semeval, title={SemEval-2017 Task 1: Semantic Textual Similarity Multilingual and Crosslingual Focused Evaluation}, author={Cer, Daniel and Diab, Mona and Agirre, Eneko and Lopez-Gazpio, Inigo and Specia, Lucia}, booktitle={Proceedings of the 11th International Workshop on Semantic Evaluation (SemEval-2017)}, pages={1--14}, year={2017}}",
                languages=["en"],
                domains=["news", "captions", "forums", "headlines", "plagiarism"]
            )
            
            logger.info(f"Loaded {len(samples)} STS benchmark samples ({split} split)")
            return samples, dataset_info
            
        except Exception as e:
            logger.error(f"Error loading STS benchmark: {e}")
            raise
    
    def get_info(self) -> DatasetInfo:
        """Get STS benchmark dataset information"""
        return DatasetInfo(
            name="STS Benchmark",
            task_type="similarity", 
            num_samples=0,  # Will be updated when loaded
            description="Semantic Textual Similarity benchmark with human-annotated similarity scores (0-5)",
            source="https://ixa2.si.ehu.eus/stswiki/index.php/STSbenchmark",
            languages=["en"],
            domains=["news", "captions", "forums", "headlines", "plagiarism"]
        )
