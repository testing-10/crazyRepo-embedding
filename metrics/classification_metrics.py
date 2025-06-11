"""
Classification metrics for embedding evaluation.

Provides lightweight, reusable classification evaluation metrics with proper edge case handling.
"""

import numpy as np
from typing import List, Dict, Union, Optional
from collections import Counter
import warnings

class ClassificationMetrics:
    def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate accuracy score for classification.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Accuracy score (0 to 1)
        """
        if len(y_true) == 0 or len(y_pred) == 0:
            return 0.0
        
        if len(y_true) != len(y_pred):
            raise ValueError("True and predicted labels must have the same length")
        
        return np.mean(y_true == y_pred)

    def precision_score(y_true: np.ndarray, y_pred: np.ndarray, 
                    average: str = 'binary', pos_label: Union[str, int] = 1) -> float:
        """
        Calculate precision score for classification.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            average: Averaging strategy ('binary', 'macro', 'micro', 'weighted')
            pos_label: Positive class label for binary classification
            
        Returns:
            Precision score (0 to 1)
        """
        if len(y_true) == 0 or len(y_pred) == 0:
            return 0.0
        
        if len(y_true) != len(y_pred):
            raise ValueError("True and predicted labels must have the same length")
        
        if average == 'binary':
            return _binary_precision(y_true, y_pred, pos_label)
        elif average == 'macro':
            return _macro_precision(y_true, y_pred)
        elif average == 'micro':
            return _micro_precision(y_true, y_pred)
        elif average == 'weighted':
            return _weighted_precision(y_true, y_pred)
        else:
            raise ValueError(f"Unknown average method: {average}")

    def recall_score(y_true: np.ndarray, y_pred: np.ndarray, 
                    average: str = 'binary', pos_label: Union[str, int] = 1) -> float:
        """
        Calculate recall score for classification.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            average: Averaging strategy ('binary', 'macro', 'micro', 'weighted')
            pos_label: Positive class label for binary classification
            
        Returns:
            Recall score (0 to 1)
        """
        if len(y_true) == 0 or len(y_pred) == 0:
            return 0.0
        
        if len(y_true) != len(y_pred):
            raise ValueError("True and predicted labels must have the same length")
        
        if average == 'binary':
            return _binary_recall(y_true, y_pred, pos_label)
        elif average == 'macro':
            return _macro_recall(y_true, y_pred)
        elif average == 'micro':
            return _micro_recall(y_true, y_pred)
        elif average == 'weighted':
            return _weighted_recall(y_true, y_pred)
        else:
            raise ValueError(f"Unknown average method: {average}")

    def f1_score(y_true: np.ndarray, y_pred: np.ndarray, 
                average: str = 'binary', pos_label: Union[str, int] = 1) -> float:
        """
        Calculate F1 score for classification.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            average: Averaging strategy ('binary', 'macro', 'micro', 'weighted')
            pos_label: Positive class label for binary classification
            
        Returns:
            F1 score (0 to 1)
        """
        precision = precision_score(y_true, y_pred, average, pos_label)
        recall = recall_score(y_true, y_pred, average, pos_label)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)

    def confusion_matrix_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Union[np.ndarray, Dict]]:
        """
        Calculate confusion matrix and related metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary containing confusion matrix and per-class metrics
        """
        if len(y_true) == 0 or len(y_pred) == 0:
            return {
                'confusion_matrix': np.array([]),
                'per_class_metrics': {},
                'overall_metrics': {}
            }
        
        if len(y_true) != len(y_pred):
            raise ValueError("True and predicted labels must have the same length")
        
        # Get unique labels
        labels = sorted(list(set(y_true) | set(y_pred)))
        n_labels = len(labels)
        
        # Create label to index mapping
        label_to_idx = {label: idx for idx, label in enumerate(labels)}
        
        # Initialize confusion matrix
        cm = np.zeros((n_labels, n_labels), dtype=int)
        
        # Fill confusion matrix
        for true_label, pred_label in zip(y_true, y_pred):
            true_idx = label_to_idx[true_label]
            pred_idx = label_to_idx[pred_label]
            cm[true_idx, pred_idx] += 1
        
        # Calculate per-class metrics
        per_class_metrics = {}
        for i, label in enumerate(labels):
            tp = cm[i, i]
            fp = np.sum(cm[:, i]) - tp
            fn = np.sum(cm[i, :]) - tp
            tn = np.sum(cm) - tp - fp - fn
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            per_class_metrics[label] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'support': np.sum(cm[i, :])
            }
        
        # Calculate overall metrics
        overall_metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'macro_precision': np.mean([metrics['precision'] for metrics in per_class_metrics.values()]),
            'macro_recall': np.mean([metrics['recall'] for metrics in per_class_metrics.values()]),
            'macro_f1': np.mean([metrics['f1_score'] for metrics in per_class_metrics.values()]),
            'weighted_precision': _weighted_precision(y_true, y_pred),
            'weighted_recall': _weighted_recall(y_true, y_pred),
            'weighted_f1': f1_score(y_true, y_pred, average='weighted')
        }
        
        return {
            'confusion_matrix': cm,
            'labels': labels,
            'per_class_metrics': per_class_metrics,
            'overall_metrics': overall_metrics
        }

    def classification_report_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Generate a comprehensive classification report as a dictionary.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary containing detailed classification metrics
        """
        return confusion_matrix_metrics(y_true, y_pred)

    # Helper functions for different averaging strategies

    def _binary_precision(y_true: np.ndarray, y_pred: np.ndarray, pos_label: Union[str, int]) -> float:
        """Calculate precision for binary classification."""
        tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
        fp = np.sum((y_true != pos_label) & (y_pred == pos_label))
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    def _binary_recall(y_true: np.ndarray, y_pred: np.ndarray, pos_label: Union[str, int]) -> float:
        """Calculate recall for binary classification."""
        tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
        fn = np.sum((y_true == pos_label) & (y_pred != pos_label))
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    def _macro_precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate macro-averaged precision."""
        labels = list(set(y_true) | set(y_pred))
        precisions = []
        
        for label in labels:
            precision = _binary_precision(y_true, y_pred, label)
            precisions.append(precision)
        
        return np.mean(precisions) if precisions else 0.0

    def _macro_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate macro-averaged recall."""
        labels = list(set(y_true) | set(y_pred))
        recalls = []
        
        for label in labels:
            recall = _binary_recall(y_true, y_pred, label)
            recalls.append(recall)
        
        return np.mean(recalls) if recalls else 0.0

    def _micro_precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate micro-averaged precision (same as accuracy for multiclass)."""
        return accuracy_score(y_true, y_pred)

    def _micro_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate micro-averaged recall (same as accuracy for multiclass)."""
        return accuracy_score(y_true, y_pred)

    def _weighted_precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate weighted precision."""
        labels = list(set(y_true))
        total_precision = 0.0
        total_support = 0
        
        for label in labels:
            precision = _binary_precision(y_true, y_pred, label)
            support = np.sum(y_true == label)
            total_precision += precision * support
            total_support += support
        
        return total_precision / total_support if total_support > 0 else 0.0

    def _weighted_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate weighted recall."""
        labels = list(set(y_true))
        total_recall = 0.0
        total_support = 0
        
        for label in labels:
            recall = _binary_recall(y_true, y_pred, label)
            support = np.sum(y_true == label)
            total_recall += recall * support
            total_support += support
        
        return total_recall / total_support if total_support > 0 else 0.0

    def top_k_accuracy(y_true: np.ndarray, y_pred_proba: np.ndarray, k: int = 5) -> float:
        """
        Calculate top-k accuracy for multi-class classification.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities (n_samples, n_classes)
            k: Number of top predictions to consider
            
        Returns:
            Top-k accuracy score (0 to 1)
        """
        if len(y_true) == 0 or len(y_pred_proba) == 0:
            return 0.0
        
        if len(y_true) != len(y_pred_proba):
            raise ValueError("True labels and predicted probabilities must have the same length")
        
        if k <= 0:
            return 0.0
        
        # Get top k predictions for each sample
        top_k_preds = np.argsort(y_pred_proba, axis=1)[:, -k:]
        
        # Check if true label is in top k predictions
        correct = 0
        for i, true_label in enumerate(y_true):
            if true_label in top_k_preds[i]:
                correct += 1
        
        return correct / len(y_true)
