"""
Dataset Loader Base Classes and Registry

Provides base interfaces for loading benchmark and custom datasets
for embedding model evaluation.
"""

import json
import os
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass
import logging

from utils.embedding_logger import EmbeddingLogger
from utils.embedding_file_utils import FileUtils


logger = EmbeddingLogger.get_logger(__name__)


@dataclass
class DatasetSample:
    """Represents a single dataset sample"""
    text1: str
    text2: Optional[str] = None
    label: Optional[Union[float, int, str]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'text1': self.text1,
            'text2': self.text2,
            'label': self.label,
            'metadata': self.metadata or {}
        }

@dataclass 
class DatasetInfo:
    """Dataset metadata and information"""
    name: str
    task_type: str  # similarity, retrieval, clustering, classification
    num_samples: int
    description: str
    source: str
    license: Optional[str] = None
    citation: Optional[str] = None
    languages: Optional[List[str]] = None
    domains: Optional[List[str]] = None

class BaseDatasetLoader(ABC):
    """Base class for all dataset loaders"""
    
    def __init__(self, cache_dir: str = "./cache/datasets/"):
        self.cache_dir = cache_dir
        self.file_utils = FileUtils()
        os.makedirs(cache_dir, exist_ok=True)
        
    @abstractmethod
    def load(self, split: str = "test", **kwargs) -> Tuple[List[DatasetSample], DatasetInfo]:
        """
        Load dataset samples and metadata
        
        Args:
            split: Dataset split (train/dev/test)
            **kwargs: Additional loader-specific parameters
            
        Returns:
            Tuple of (samples, dataset_info)
        """
        pass
    
    @abstractmethod
    def get_info(self) -> DatasetInfo:
        """Get dataset information and metadata"""
        pass
    
    def _validate_samples(self, samples: List[DatasetSample]) -> List[DatasetSample]:
        """Validate and clean dataset samples"""
        valid_samples = []
        
        for i, sample in enumerate(samples):
            try:
                # Basic validation
                if not sample.text1 or not sample.text1.strip():
                    logger.warning(f"Sample {i}: Empty text1, skipping")
                    continue
                    
                # Clean text
                sample.text1 = sample.text1.strip()
                if sample.text2:
                    sample.text2 = sample.text2.strip()
                    
                valid_samples.append(sample)
                
            except Exception as e:
                logger.warning(f"Sample {i}: Validation error - {e}")
                continue
                
        logger.info(f"Validated {len(valid_samples)}/{len(samples)} samples")
        return valid_samples

class BenchmarkDatasetLoader(BaseDatasetLoader):
    """Base class for standard benchmark dataset loaders"""
    
    def __init__(self, cache_dir: str = "./cache/datasets/"):
        super().__init__(cache_dir)
        self.download_required = True
        
    @abstractmethod
    def download(self) -> bool:
        """Download dataset if not cached"""
        pass
    
    def _check_cache(self, dataset_name: str) -> bool:
        """Check if dataset is already cached"""
        cache_path = os.path.join(self.cache_dir, dataset_name)
        return os.path.exists(cache_path) and os.listdir(cache_path)
    
    def _get_cache_path(self, dataset_name: str) -> str:
        """Get cache directory path for dataset"""
        cache_path = os.path.join(self.cache_dir, dataset_name)
        os.makedirs(cache_path, exist_ok=True)
        return cache_path

class CustomDatasetLoader(BaseDatasetLoader):
    """Loader for custom JSON/CSV datasets"""
    
    def __init__(self, file_path: str, cache_dir: str = "./cache/datasets/"):
        super().__init__(cache_dir)
        self.file_path = file_path
        
    def load(self, split: str = "test", **kwargs) -> Tuple[List[DatasetSample], DatasetInfo]:
        """Load custom dataset from JSON or CSV file"""
        try:
            if not os.path.exists(self.file_path):
                raise FileNotFoundError(f"Dataset file not found: {self.file_path}")
                
            # Determine file format
            file_ext = os.path.splitext(self.file_path)[1].lower()
            
            if file_ext == '.json':
                samples = self._load_json()
            elif file_ext == '.csv':
                samples = self._load_csv()
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
                
            # Validate samples
            samples = self._validate_samples(samples)
            
            # Create dataset info
            dataset_info = DatasetInfo(
                name=os.path.basename(self.file_path),
                task_type=kwargs.get('task_type', 'similarity'),
                num_samples=len(samples),
                description=f"Custom dataset from {self.file_path}",
                source="custom"
            )
            
            logger.info(f"Loaded {len(samples)} samples from {self.file_path}")
            return samples, dataset_info
            
        except Exception as e:
            logger.error(f"Error loading custom dataset: {e}")
            raise
    
    def _load_json(self) -> List[DatasetSample]:
        """Load samples from JSON file"""
        with open(self.file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        samples = []
        
        # Handle different JSON structures
        if isinstance(data, list):
            # List of samples
            for item in data:
                samples.append(self._parse_json_item(item))
        elif isinstance(data, dict):
            # Dictionary with samples under a key
            if 'samples' in data:
                for item in data['samples']:
                    samples.append(self._parse_json_item(item))
            else:
                # Single sample
                samples.append(self._parse_json_item(data))
                
        return samples
    
    def _parse_json_item(self, item: Dict[str, Any]) -> DatasetSample:
        """Parse a single JSON item into DatasetSample"""
        # Flexible field mapping
        text1 = item.get('text1') or item.get('text') or item.get('sentence1') or item.get('query')
        text2 = item.get('text2') or item.get('sentence2') or item.get('document')
        label = item.get('label') or item.get('score') or item.get('similarity')
        
        # Extract metadata (all other fields)
        metadata = {k: v for k, v in item.items() 
                if k not in ['text1', 'text2', 'text', 'sentence1', 'sentence2', 
                            'query', 'document', 'label', 'score', 'similarity']}
        
        return DatasetSample(
            text1=str(text1) if text1 is not None else "",
            text2=str(text2) if text2 is not None else None,
            label=label,
            metadata=metadata if metadata else None
        )
    
    def _load_csv(self) -> List[DatasetSample]:
        """Load samples from CSV file"""
        df = pd.read_csv(self.file_path)
        samples = []
        
        for _, row in df.iterrows():
            # Flexible column mapping
            text1 = None
            text2 = None
            label = None
            
            # Try different column names
            for col in df.columns:
                col_lower = col.lower()
                if col_lower in ['text1', 'text', 'sentence1', 'query'] and text1 is None:
                    text1 = row[col]
                elif col_lower in ['text2', 'sentence2', 'document'] and text2 is None:
                    text2 = row[col]
                elif col_lower in ['label', 'score', 'similarity'] and label is None:
                    label = row[col]
            
            # Extract metadata
            metadata = {col: row[col] for col in df.columns 
                    if col.lower() not in ['text1', 'text2', 'text', 'sentence1', 
                                            'sentence2', 'query', 'document', 'label', 
                                            'score', 'similarity']}
            
            samples.append(DatasetSample(
                text1=str(text1) if pd.notna(text1) else "",
                text2=str(text2) if pd.notna(text2) else None,
                label=label if pd.notna(label) else None,
                metadata=metadata if metadata else None
            ))
            
        return samples
    
    def get_info(self) -> DatasetInfo:
        """Get dataset information"""
        return DatasetInfo(
            name=os.path.basename(self.file_path),
            task_type="custom",
            num_samples=0,  # Will be updated when loaded
            description=f"Custom dataset from {self.file_path}",
            source="custom"
        )

class DatasetRegistry:
    """Registry for managing dataset loaders"""
    
    def __init__(self):
        self._loaders: Dict[str, type] = {}
        
    def register(self, name: str, loader_class: type):
        """Register a dataset loader"""
        if not issubclass(loader_class, BaseDatasetLoader):
            raise ValueError(f"Loader must inherit from BaseDatasetLoader")
            
        self._loaders[name] = loader_class
        logger.info(f"Registered dataset loader: {name}")
        
    def get_loader(self, name: str, **kwargs) -> BaseDatasetLoader:
        """Get a dataset loader instance"""
        if name not in self._loaders:
            raise ValueError(f"Unknown dataset loader: {name}")
            
        return self._loaders[name](**kwargs)
    
    def list_loaders(self) -> List[str]:
        """List all registered loaders"""
        return list(self._loaders.keys())

def load_dataset(dataset_name: str, split: str = "test", **kwargs) -> Tuple[List[DatasetSample], DatasetInfo]:
    """
    Convenience function to load a dataset
    
    Args:
        dataset_name: Name of registered dataset or path to custom file
        split: Dataset split to load
        **kwargs: Additional parameters for the loader
        
    Returns:
        Tuple of (samples, dataset_info)
    """
    from datasets import DATASET_REGISTRY
    
    # Check if it's a registered dataset
    if dataset_name in DATASET_REGISTRY.list_loaders():
        loader = DATASET_REGISTRY.get_loader(dataset_name, **kwargs)
        return loader.load(split=split, **kwargs)
    
    # Check if it's a file path
    elif os.path.exists(dataset_name):
        loader = CustomDatasetLoader(dataset_name, **kwargs)
        return loader.load(split=split, **kwargs)
    
    else:
        raise ValueError(f"Unknown dataset or file not found: {dataset_name}")

class DatasetLoader:
    pass