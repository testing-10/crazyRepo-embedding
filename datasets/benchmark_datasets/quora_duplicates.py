"""
Quora Question Pairs Dataset Loader

Loads the Quora Question Pairs dataset for evaluating semantic similarity
and duplicate detection between question pairs.
"""

import os
import requests
import pandas as pd
from typing import List, Tuple

from datasets.dataset_loader import BenchmarkDatasetLoader, DatasetSample, DatasetInfo
from utils.embedding_logger import EmbeddingLogger

logger = EmbeddingLogger.get_logger(__name__)

class QuoraDuplicatesLoader(BenchmarkDatasetLoader):
    """Loader for Quora Question Pairs dataset"""
    
    # Note: Direct download requires Kaggle API or manual download
    DATASET_NAME = "quora_duplicates"
    KAGGLE_DATASET = "c1udxj/quora-question-pairs"
    
    def __init__(self, cache_dir: str = "./cache/datasets/", subset_size: int = None):
        super().__init__(cache_dir)
        self.dataset_path = self._get_cache_path(self.DATASET_NAME)
        self.subset_size = subset_size
        
    def download(self) -> bool:
        """Download Quora dataset (requires manual setup or Kaggle API)"""
        try:
            if self._check_cache(self.DATASET_NAME):
                logger.info("Quora Question Pairs already cached")
                return True
            
            # Check if file exists in cache directory
            train_file = os.path.join(self.dataset_path, "train.csv")
            if os.path.exists(train_file):
                logger.info("Quora dataset file found")
                return True
            
            # Try to download using kaggle API if available
            try:
                import kaggle
                logger.info("Downloading Quora dataset using Kaggle API...")
                kaggle.api.dataset_download_files(
                    self.KAGGLE_DATASET, 
                    path=self.dataset_path, 
                    unzip=True
                )
                logger.info("Quora dataset downloaded successfully")
                return True
                
            except ImportError:
                logger.warning("Kaggle API not available. Please download manually.")
                logger.info(f"Download from: https://www.kaggle.com/datasets/{self.KAGGLE_DATASET}")
                logger.info(f"Place train.csv in: {self.dataset_path}")
                return False
                
            except Exception as e:
                logger.warning(f"Kaggle download failed: {e}")
                logger.info("Please download manually and place train.csv in cache directory")
                return False
                
        except Exception as e:
            logger.error(f"Error setting up Quora dataset: {e}")
            return False
    
    def load(self, split: str = "train", **kwargs) -> Tuple[List[DatasetSample], DatasetInfo]:
        """Load Quora Question Pairs dataset"""
        try:
            # Check for dataset availability
            if not self.download():
                # Create a small sample dataset for testing
                return self._create_sample_dataset()
            
            # Load the CSV file
            csv_file = os.path.join(self.dataset_path, f"{split}.csv")
            if not os.path.exists(csv_file):
                csv_file = os.path.join(self.dataset_path, "train.csv")  # Fallback
            
            if not os.path.exists(csv_file):
                logger.warning("Quora CSV file not found, creating sample dataset")
                return self._create_sample_dataset()
            
            logger.info(f"Loading Quora dataset from {csv_file}")
            df = pd.read_csv(csv_file)
            
            # Apply subset limit if specified
            if self.subset_size:
                df = df.head(self.subset_size)
            
            samples = []
            
            for _, row in df.iterrows():
                if pd.notna(row.get('question1')) and pd.notna(row.get('question2')):
                    samples.append(DatasetSample(
                        text1=str(row['question1']).strip(),
                        text2=str(row['question2']).strip(),
                        label=int(row.get('is_duplicate', 0)),
                        metadata={
                            'id': row.get('id'),
                            'qid1': row.get('qid1'),
                            'qid2': row.get('qid2'),
                            'task': 'similarity'
                        }
                    ))
            
            # Validate samples
            samples = self._validate_samples(samples)
            
            dataset_info = DatasetInfo(
                name="Quora Question Pairs",
                task_type="similarity",
                num_samples=len(samples),
                description="Question pairs from Quora with binary duplicate labels",
                source="https://www.kaggle.com/c/quora-question-pairs",
                citation="@misc{quora2017, title={Quora Question Pairs}, author={Quora}, year={2017}, url={https://www.kaggle.com/c/quora-question-pairs}}",
                languages=["en"],
                domains=["questions", "general"]
            )
            
            logger.info(f"Loaded {len(samples)} Quora question pairs ({split} split)")
            return samples, dataset_info
            
        except Exception as e:
            logger.error(f"Error loading Quora dataset: {e}")
            # Fallback to sample dataset
            return self._create_sample_dataset()
    
    def _create_sample_dataset(self) -> Tuple[List[DatasetSample], DatasetInfo]:
        """Create a small sample dataset for testing when main dataset unavailable"""
        logger.info("Creating sample Quora-style dataset for testing")
        
        sample_pairs = [
            ("What is machine learning?", "Can you explain machine learning?", 1),
            ("How do I learn Python?", "What's the best way to learn Python programming?", 1),
            ("What is the capital of France?", "How do I cook pasta?", 0),
            ("Why is the sky blue?", "What causes the blue color of the sky?", 1),
            ("How to lose weight fast?", "What are quick weight loss methods?", 1),
            ("What is quantum computing?", "How does quantum computing work?", 1),
            ("Best restaurants in NYC", "Where to eat in New York City?", 1),
            ("How to fix a car engine?", "What is artificial intelligence?", 0),
            ("Python vs Java programming", "Comparison between Python and Java", 1),
            ("Climate change effects", "Global warming impact on environment", 1)
        ]
        
        samples = []
        for i, (q1, q2, label) in enumerate(sample_pairs):
            samples.append(DatasetSample(
                text1=q1,
                text2=q2,
                label=label,
                metadata={
                    'id': i,
                    'task': 'similarity',
                    'source': 'sample'
                }
            ))
        
        dataset_info = DatasetInfo(
            name="Quora Question Pairs (Sample)",
            task_type="similarity",
            num_samples=len(samples),
            description="Sample question pairs for testing (Quora-style)",
            source="generated_sample",
            languages=["en"],
            domains=["questions", "general"]
        )
        
        return samples, dataset_info
    
    def get_info(self) -> DatasetInfo:
        """Get Quora dataset information"""
        return DatasetInfo(
            name="Quora Question Pairs",
            task_type="similarity",
            num_samples=0,  # Will be updated when loaded
            description="Question pairs from Quora with binary duplicate labels",
            source="https://www.kaggle.com/c/quora-question-pairs",
            languages=["en"],
            domains=["questions", "general"]
        )
