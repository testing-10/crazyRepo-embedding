"""
Embedding Logger - Centralized logging for embedding model testing framework
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import json


class EmbeddingLogger:
    """
    Centralized logging utility for embedding model testing framework.
    Provides structured logging with multiple output formats and levels.
    """
    
    _instances: Dict[str, 'EmbeddingLogger'] = {}
    
    def __init__(self, 
                 name: str = "embedding_framework",
                 log_level: str = "INFO",
                 log_dir: Optional[str] = None,
                 console_output: bool = True,
                 file_output: bool = True,
                 json_format: bool = False):
        """
        Initialize the embedding logger.
        
        Args:
            name: Logger name/identifier
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_dir: Directory for log files (defaults to ./logs)
            console_output: Enable console logging
            file_output: Enable file logging
            json_format: Use JSON format for structured logging
        """
        self.name = name
        self.log_level = getattr(logging, log_level.upper(), logging.INFO)
        self.log_dir = Path(log_dir) if log_dir else Path("logs")
        self.console_output = console_output
        self.file_output = file_output
        self.json_format = json_format
        
        # Create log directory
        if self.file_output:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logger
        self.logger = self._setup_logger()
        
    @classmethod
    def get_logger(cls, 
                   name: str = "embedding_framework",
                   **kwargs) -> 'EmbeddingLogger':
        """
        Get or create a logger instance (singleton pattern per name).
        
        Args:
            name: Logger name
            **kwargs: Logger configuration parameters
            
        Returns:
            EmbeddingLogger instance
        """
        if name not in cls._instances:
            cls._instances[name] = cls(name=name, **kwargs)
        return cls._instances[name]
    
    def _setup_logger(self) -> logging.Logger:
        """Setup and configure the logger with handlers."""
        logger = logging.getLogger(self.name)
        logger.setLevel(self.log_level)
        
        # Clear existing handlers to avoid duplicates
        logger.handlers.clear()
        
        # Create formatters
        if self.json_format:
            formatter = self._get_json_formatter()
        else:
            formatter = self._get_standard_formatter()
        
        # Console handler
        if self.console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.log_level)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # File handler
        if self.file_output:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = self.log_dir / f"{self.name}_{timestamp}.log"
            
            file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
            file_handler.setLevel(self.log_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def _get_standard_formatter(self) -> logging.Formatter:
        """Get standard text formatter."""
        return logging.Formatter(
            fmt='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    def _get_json_formatter(self) -> logging.Formatter:
        """Get JSON formatter for structured logging."""
        class JsonFormatter(logging.Formatter):
            def format(self, record):
                log_entry = {
                    'timestamp': datetime.fromtimestamp(record.created).isoformat(),
                    'logger': record.name,
                    'level': record.levelname,
                    'message': record.getMessage(),
                    'module': record.module,
                    'function': record.funcName,
                    'line': record.lineno
                }
                
                # Add extra fields if present
                if hasattr(record, 'extra_data'):
                    log_entry.update(record.extra_data)
                
                return json.dumps(log_entry)
        
        return JsonFormatter()
    
    def debug(self, message: str, extra_data: Optional[Dict[str, Any]] = None):
        """Log debug message."""
        self._log(logging.DEBUG, message, extra_data)
    
    def info(self, message: str, extra_data: Optional[Dict[str, Any]] = None):
        """Log info message."""
        self._log(logging.INFO, message, extra_data)
    
    def warning(self, message: str, extra_data: Optional[Dict[str, Any]] = None):
        """Log warning message."""
        self._log(logging.WARNING, message, extra_data)
    
    def error(self, message: str, extra_data: Optional[Dict[str, Any]] = None):
        """Log error message."""
        self._log(logging.ERROR, message, extra_data)
    
    def critical(self, message: str, extra_data: Optional[Dict[str, Any]] = None):
        """Log critical message."""
        self._log(logging.CRITICAL, message, extra_data)
    
    def _log(self, level: int, message: str, extra_data: Optional[Dict[str, Any]] = None):
        """Internal logging method with extra data support."""
        if extra_data:
            # Create a custom LogRecord with extra data
            record = self.logger.makeRecord(
                self.logger.name, level, "", 0, message, (), None
            )
            record.extra_data = extra_data
            self.logger.handle(record)
        else:
            self.logger.log(level, message)
    
    def log_model_evaluation(self, 
                           model_name: str,
                           test_type: str,
                           metrics: Dict[str, Any],
                           duration: float):
        """Log model evaluation results."""
        extra_data = {
            'model_name': model_name,
            'test_type': test_type,
            'metrics': metrics,
            'duration_seconds': duration,
            'event_type': 'model_evaluation'
        }
        self.info(f"Model evaluation completed: {model_name} - {test_type}", extra_data)
    
    def log_api_call(self, 
                     provider: str,
                     endpoint: str,
                     tokens_used: int,
                     cost: float,
                     duration: float):
        """Log API call details."""
        extra_data = {
            'provider': provider,
            'endpoint': endpoint,
            'tokens_used': tokens_used,
            'cost_usd': cost,
            'duration_seconds': duration,
            'event_type': 'api_call'
        }
        self.info(f"API call: {provider} - {endpoint}", extra_data)
    
    def log_error_with_context(self, 
                              error: Exception,
                              context: Dict[str, Any]):
        """Log error with additional context."""
        extra_data = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context,
            'event_type': 'error'
        }
        self.error(f"Error occurred: {type(error).__name__}", extra_data)
    
    def set_level(self, level: str):
        """Change logging level dynamically."""
        new_level = getattr(logging, level.upper(), logging.INFO)
        self.logger.setLevel(new_level)
        for handler in self.logger.handlers:
            handler.setLevel(new_level)
    
    def get_log_files(self) -> list:
        """Get list of current log files."""
        if not self.file_output or not self.log_dir.exists():
            return []
        
        return [str(f) for f in self.log_dir.glob(f"{self.name}_*.log")]


# Convenience function for quick logger access
def get_logger(name: str = "embedding_framework", **kwargs) -> EmbeddingLogger:
    """Get a logger instance with the specified name and configuration."""
    return EmbeddingLogger.get_logger(name=name, **kwargs)
