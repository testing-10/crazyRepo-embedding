"""
Classification Evaluator

Evaluates embedding models on classification tasks using embeddings as features
for downstream classification with various classifiers and metrics.
"""

import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline

from evaluators.base_embedding_evaluator import BaseEmbeddingEvaluator, EvaluationResult
from datasets.dataset_loader import DatasetSample
from utils.vector_operations import VectorOperations

class ClassificationEvaluator(BaseEmbeddingEvaluator):
    """Evaluator for classification tasks using embeddings as features"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.vector_ops = VectorOperations()
        
        # Default configuration
        self.default_config = {
            'classifiers': ['logistic_regression', 'svm', 'random_forest', 'knn'],
            'metrics': ['accuracy', 'precision', 'recall', 'f1_score', 'auc'],
            'cross_validation_folds': 5,
            'test_size': 0.2,
            'normalize_embeddings': True,
            'random_state': 42,
            'max_samples': 5000,  # Limit for performance
            'multiclass_average': 'weighted'  # For multiclass metrics
        }
        
        # Merge with provided config
        self.config = {**self.default_config, **self.config}
    
    def evaluate(self, 
                 embeddings: List[np.ndarray], 
                 samples: List[DatasetSample],
                 model_name: str,
                 dataset_name: str,
                 **kwargs) -> EvaluationResult:
        """Evaluate classification performance using embeddings as features"""
        
        start_time = time.time()
        self._log_evaluation_start(model_name, dataset_name, len(samples))
        
        try:
            # Validate inputs
            if not self._validate_inputs(embeddings, samples):
                raise ValueError("Input validation failed")
            
            # Prepare embeddings and filter samples
            valid_embeddings, valid_indices = self._prepare_embeddings(embeddings)
            valid_samples = self._filter_samples(samples, valid_indices)
            
            # Extract labels for classification
            X, y, label_info = self._prepare_classification_data(valid_embeddings, valid_samples)
            
            if len(X) == 0 or len(np.unique(y)) < 2:
                raise ValueError("Insufficient data or classes for classification")
            
            # Limit samples for performance if needed
            if len(X) > self.config['max_samples']:
                indices = np.random.choice(len(X), self.config['max_samples'], replace=False)
                X = X[indices]
                y = y[indices]
                self.logger.info(f"Subsampled to {self.config['max_samples']} samples for classification")
            
            # Perform classification evaluation
            metrics = self._evaluate_classification(X, y, label_info)
            
            # Generate metadata
            metadata = self._generate_metadata(X, y, label_info, valid_samples)
            
            execution_time = time.time() - start_time
            self._log_evaluation_end(metrics, execution_time)
            
            return self._create_result(
                task_type="classification",
                model_name=model_name,
                dataset_name=dataset_name,
                metrics=metrics,
                metadata=metadata,
                execution_time=execution_time,
                num_samples=len(X)
            )
            
        except Exception as e:
            self.logger.error(f"Classification evaluation failed: {e}")
            raise
    
    def _prepare_classification_data(self, embeddings: np.ndarray, samples: List[DatasetSample]) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Prepare embeddings and labels for classification"""
        X = embeddings
        labels = []
        
        # Extract labels from samples
        for sample in samples:
            if sample.label is not None:
                labels.append(sample.label)
            elif sample.metadata and 'category' in sample.metadata:
                labels.append(sample.metadata['category'])
            elif sample.metadata and 'domain' in sample.metadata:
                labels.append(sample.metadata['domain'])
            elif sample.metadata and 'class' in sample.metadata:
                labels.append(sample.metadata['class'])
            else:
                labels.append('unknown')
        
        # Encode labels
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(labels)
        
        # Filter out unknown labels if they exist
        known_mask = np.array(labels) != 'unknown'
        if np.sum(known_mask) < len(labels):
            X = X[known_mask]
            y = y[known_mask]
            self.logger.info(f"Filtered out {len(labels) - np.sum(known_mask)} samples with unknown labels")
        
        # Label information
        label_info = {
            'classes': label_encoder.classes_.tolist(),
            'n_classes': len(label_encoder.classes_),
            'class_distribution': {
                str(cls): int(np.sum(y == i)) 
                for i, cls in enumerate(label_encoder.classes_)
            }
        }
        
        return X, y, label_info
    
    def _evaluate_classification(self, X: np.ndarray, y: np.ndarray, label_info: Dict) -> Dict[str, float]:
        """Evaluate classification performance with different classifiers"""
        metrics = {}
        
        # Normalize embeddings if configured
        if self.config['normalize_embeddings']:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        
        # Determine if binary or multiclass
        is_binary = label_info['n_classes'] == 2
        
        # Evaluate each classifier
        for classifier_name in self.config['classifiers']:
            try:
                classifier_metrics = self._evaluate_single_classifier(
                    X, y, classifier_name, is_binary, label_info
                )
                metrics.update(classifier_metrics)
                
            except Exception as e:
                self.logger.warning(f"Error evaluating {classifier_name}: {e}")
        
        return metrics
    
    def _evaluate_single_classifier(self, X: np.ndarray, y: np.ndarray, 
                                   classifier_name: str, is_binary: bool, 
                                   label_info: Dict) -> Dict[str, float]:
        """Evaluate a single classifier"""
        metrics = {}
        prefix = f"{classifier_name}_"
        
        # Get classifier
        classifier = self._get_classifier(classifier_name)
        
        # Cross-validation evaluation
        cv = StratifiedKFold(n_splits=self.config['cross_validation_folds'], 
                            shuffle=True, random_state=self.config['random_state'])
        
        # Cross-validation scores
        cv_scores = cross_val_score(classifier, X, y, cv=cv, scoring='accuracy')
        metrics[f'{prefix}cv_accuracy_mean'] = float(np.mean(cv_scores))
        metrics[f'{prefix}cv_accuracy_std'] = float(np.std(cv_scores))
        
        # Train-test split evaluation
        from sklearn.model_selection import train_test_split
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config['test_size'], 
            random_state=self.config['random_state'], stratify=y
        )
        
        # Fit classifier
        classifier.fit(X_train, y_train)
        
        # Predictions
        y_pred = classifier.predict(X_test)
        y_pred_proba = None
        
        if hasattr(classifier, 'predict_proba'):
            try:
                y_pred_proba = classifier.predict_proba(X_test)
            except:
                pass
        
        # Calculate metrics
        metrics[f'{prefix}accuracy'] = float(accuracy_score(y_test, y_pred))
        
        # Precision, Recall, F1
        average = 'binary' if is_binary else self.config['multiclass_average']
        
        metrics[f'{prefix}precision'] = float(precision_score(y_test, y_pred, average=average, zero_division=0))
        metrics[f'{prefix}recall'] = float(recall_score(y_test, y_pred, average=average, zero_division=0))
        metrics[f'{prefix}f1_score'] = float(f1_score(y_test, y_pred, average=average, zero_division=0))
        
        # AUC (if probabilities available)
        if y_pred_proba is not None and 'auc' in self.config['metrics']:
            try:
                if is_binary:
                    auc = roc_auc_score(y_test, y_pred_proba[:, 1])
                else:
                    auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
                metrics[f'{prefix}auc'] = float(auc)
            except Exception as e:
                self.logger.warning(f"Could not calculate AUC for {classifier_name}: {e}")
        
        # Per-class metrics for multiclass
        if not is_binary and label_info['n_classes'] <= 10:  # Limit to avoid too many metrics
            precision_per_class = precision_score(y_test, y_pred, average=None, zero_division=0)
            recall_per_class = recall_score(y_test, y_pred, average=None, zero_division=0)
            f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0)
            
            for i, class_name in enumerate(label_info['classes']):
                metrics[f'{prefix}precision_class_{class_name}'] = float(precision_per_class[i])
                metrics[f'{prefix}recall_class_{class_name}'] = float(recall_per_class[i])
                metrics[f'{prefix}f1_class_{class_name}'] = float(f1_per_class[i])
        
        return metrics
    
    def _get_classifier(self, classifier_name: str):
        """Get classifier instance by name"""
        classifiers = {
            'logistic_regression': LogisticRegression(
                random_state=self.config['random_state'],
                max_iter=1000
            ),
            'svm': SVC(
                random_state=self.config['random_state'],
                probability=True
            ),
            'random_forest': RandomForestClassifier(
                random_state=self.config['random_state'],
                n_estimators=100
            ),
            'knn': KNeighborsClassifier(n_neighbors=5)
        }
        
        if classifier_name not in classifiers:
            raise ValueError(f"Unknown classifier: {classifier_name}")
        
        return classifiers[classifier_name]
    
    def _generate_metadata(self, X: np.ndarray, y: np.ndarray, 
                          label_info: Dict, samples: List[DatasetSample]) -> Dict[str, Any]:
        """Generate classification evaluation metadata"""
        metadata = {
            'embedding_dimension': int(X.shape[1]),
            'n_samples': len(X),
            'n_classes': label_info['n_classes'],
            'class_names': label_info['classes'],
            'class_distribution': label_info['class_distribution'],
            'classifiers_used': self.config['classifiers'],
            'is_binary_classification': label_info['n_classes'] == 2,
            'config': self.config.copy()
        }
        
        # Class balance analysis
        class_counts = list(label_info['class_distribution'].values())
        metadata['class_balance'] = {
            'min_class_size': int(np.min(class_counts)),
            'max_class_size': int(np.max(class_counts)),
            'balance_ratio': float(np.min(class_counts) / np.max(class_counts))
        }
        
        # Domain analysis if available
        domains = [sample.metadata.get('domain') for sample in samples if sample.metadata]
        if domains:
            unique_domains = list(set([d for d in domains if d is not None]))
            metadata['domains'] = unique_domains
            metadata['cross_domain_classification'] = len(unique_domains) > 1
        
        return metadata
    
    def get_required_metrics(self) -> List[str]:
        """Get list of metrics this evaluator provides"""
        base_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
        classifier_metrics = []
        
        for classifier in self.config['classifiers']:
            for metric in base_metrics:
                classifier_metrics.append(f'{classifier}_{metric}')
            classifier_metrics.append(f'{classifier}_cv_accuracy_mean')
        
        return base_metrics + classifier_metrics
