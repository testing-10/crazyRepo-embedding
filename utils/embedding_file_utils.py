"""
File Utilities - File operations for embedding model testing framework
"""

import json
import yaml
import csv
import os
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import pickle
import gzip


class FileUtils:
    """
    Utility class for file operations in the embedding testing framework.
    Handles JSON, YAML, CSV, and other file formats with error handling.
    """
    
    @staticmethod
    def ensure_directory(path: Union[str, Path]) -> Path:
        """
        Ensure directory exists, create if it doesn't.
        
        Args:
            path: Directory path
            
        Returns:
            Path object of the directory
            
        Raises:
            OSError: If directory cannot be created
        """
        path_obj = Path(path)
        try:
            path_obj.mkdir(parents=True, exist_ok=True)
            return path_obj
        except OSError as e:
            raise OSError(f"Failed to create directory {path}: {e}")
    
    @staticmethod
    def load_json(file_path: Union[str, Path], 
                  default: Optional[Any] = None) -> Any:
        """
        Load JSON file with error handling.
        
        Args:
            file_path: Path to JSON file
            default: Default value if file doesn't exist or is invalid
            
        Returns:
            Parsed JSON data or default value
            
        Raises:
            FileNotFoundError: If file doesn't exist and no default provided
            json.JSONDecodeError: If JSON is invalid and no default provided
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            if default is not None:
                return default
            raise FileNotFoundError(f"JSON file not found: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            if default is not None:
                return default
            raise json.JSONDecodeError(f"Invalid JSON in {file_path}: {e}")
    
    @staticmethod
    def save_json(data: Any, 
                  file_path: Union[str, Path],
                  indent: int = 2,
                  ensure_ascii: bool = False) -> bool:
        """
        Save data to JSON file with error handling.
        
        Args:
            data: Data to save
            file_path: Output file path
            indent: JSON indentation
            ensure_ascii: Whether to escape non-ASCII characters
            
        Returns:
            True if successful, False otherwise
            
        Raises:
            OSError: If file cannot be written
        """
        file_path = Path(file_path)
        FileUtils.ensure_directory(file_path.parent)
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii, default=str)
            return True
        except (OSError, TypeError) as e:
            raise OSError(f"Failed to save JSON to {file_path}: {e}")
    
    @staticmethod
    def load_yaml(file_path: Union[str, Path], 
                  default: Optional[Any] = None) -> Any:
        """
        Load YAML file with error handling.
        
        Args:
            file_path: Path to YAML file
            default: Default value if file doesn't exist or is invalid
            
        Returns:
            Parsed YAML data or default value
            
        Raises:
            FileNotFoundError: If file doesn't exist and no default provided
            yaml.YAMLError: If YAML is invalid and no default provided
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            if default is not None:
                return default
            raise FileNotFoundError(f"YAML file not found: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except yaml.YAMLError as e:
            if default is not None:
                return default
            raise yaml.YAMLError(f"Invalid YAML in {file_path}: {e}")
    
    @staticmethod
    def save_yaml(data: Any, 
                  file_path: Union[str, Path],
                  default_flow_style: bool = False) -> bool:
        """
        Save data to YAML file with error handling.
        
        Args:
            data: Data to save
            file_path: Output file path
            default_flow_style: YAML flow style
            
        Returns:
            True if successful, False otherwise
            
        Raises:
            OSError: If file cannot be written
        """
        file_path = Path(file_path)
        FileUtils.ensure_directory(file_path.parent)
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=default_flow_style, 
                         allow_unicode=True, sort_keys=False)
            return True
        except (OSError, yaml.YAMLError) as e:
            raise OSError(f"Failed to save YAML to {file_path}: {e}")
    
    @staticmethod
    def load_csv(file_path: Union[str, Path], 
                 delimiter: str = ',',
                 has_header: bool = True) -> List[Dict[str, Any]]:
        """
        Load CSV file as list of dictionaries.
        
        Args:
            file_path: Path to CSV file
            delimiter: CSV delimiter
            has_header: Whether CSV has header row
            
        Returns:
            List of dictionaries representing CSV rows
            
        Raises:
            FileNotFoundError: If file doesn't exist
            csv.Error: If CSV is invalid
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if has_header:
                    reader = csv.DictReader(f, delimiter=delimiter)
                    return list(reader)
                else:
                    reader = csv.reader(f, delimiter=delimiter)
                    return [{"col_" + str(i): val for i, val in enumerate(row)} 
                           for row in reader]
        except csv.Error as e:
            raise csv.Error(f"Invalid CSV in {file_path}: {e}")
    
    @staticmethod
    def save_csv(data: List[Dict[str, Any]], 
                 file_path: Union[str, Path],
                 delimiter: str = ',') -> bool:
        """
        Save list of dictionaries to CSV file.
        
        Args:
            data: List of dictionaries to save
            file_path: Output file path
            delimiter: CSV delimiter
            
        Returns:
            True if successful, False otherwise
            
        Raises:
            OSError: If file cannot be written
            ValueError: If data is empty or invalid
        """
        if not data:
            raise ValueError("Cannot save empty data to CSV")
        
        file_path = Path(file_path)
        FileUtils.ensure_directory(file_path.parent)
        
        try:
            fieldnames = data[0].keys()
            with open(file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter)
                writer.writeheader()
                writer.writerows(data)
            return True
        except (OSError, csv.Error) as e:
            raise OSError(f"Failed to save CSV to {file_path}: {e}")
    
    @staticmethod
    def load_text_file(file_path: Union[str, Path], 
                       encoding: str = 'utf-8') -> str:
        """
        Load text file content.
        
        Args:
            file_path: Path to text file
            encoding: File encoding
            
        Returns:
            File content as string
            
        Raises:
            FileNotFoundError: If file doesn't exist
            UnicodeDecodeError: If encoding is incorrect
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Text file not found: {file_path}")
        
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError as e:
            raise UnicodeDecodeError(f"Failed to decode {file_path} with {encoding}: {e}")
    
    @staticmethod
    def save_text_file(content: str, 
                       file_path: Union[str, Path],
                       encoding: str = 'utf-8') -> bool:
        """
        Save text content to file.
        
        Args:
            content: Text content to save
            file_path: Output file path
            encoding: File encoding
            
        Returns:
            True if successful, False otherwise
            
        Raises:
            OSError: If file cannot be written
        """
        file_path = Path(file_path)
        FileUtils.ensure_directory(file_path.parent)
        
        try:
            with open(file_path, 'w', encoding=encoding) as f:
                f.write(content)
            return True
        except OSError as e:
            raise OSError(f"Failed to save text to {file_path}: {e}")
    
    @staticmethod
    def copy_file(source: Union[str, Path], 
                  destination: Union[str, Path]) -> bool:
        """
        Copy file from source to destination.
        
        Args:
            source: Source file path
            destination: Destination file path
            
        Returns:
            True if successful, False otherwise
            
        Raises:
            FileNotFoundError: If source file doesn't exist
            OSError: If copy operation fails
        """
        source_path = Path(source)
        dest_path = Path(destination)
        
        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")
        
        FileUtils.ensure_directory(dest_path.parent)
        
        try:
            shutil.copy2(source_path, dest_path)
            return True
        except OSError as e:
            raise OSError(f"Failed to copy {source_path} to {dest_path}: {e}")
    
    @staticmethod
    def delete_file(file_path: Union[str, Path]) -> bool:
        """
        Delete file if it exists.
        
        Args:
            file_path: Path to file to delete
            
        Returns:
            True if file was deleted or didn't exist, False otherwise
        """
        file_path = Path(file_path)
        
        try:
            if file_path.exists():
                file_path.unlink()
            return True
        except OSError:
            return False
    
    @staticmethod
    def get_file_size(file_path: Union[str, Path]) -> int:
        """
        Get file size in bytes.
        
        Args:
            file_path: Path to file
            
        Returns:
            File size in bytes
            
        Raises:
            FileNotFoundError: If file doesn't exist
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        return file_path.stat().st_size
    
    @staticmethod
    def get_file_modification_time(file_path: Union[str, Path]) -> datetime:
        """
        Get file modification time.
        
        Args:
            file_path: Path to file
            
        Returns:
            File modification time as datetime
            
        Raises:
            FileNotFoundError: If file doesn't exist
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        return datetime.fromtimestamp(file_path.stat().st_mtime)
    
    @staticmethod
    def list_files(directory: Union[str, Path], 
                   pattern: str = "*",
                   recursive: bool = False) -> List[Path]:
        """
        List files in directory matching pattern.
        
        Args:
            directory: Directory to search
            pattern: File pattern (glob style)
            recursive: Whether to search recursively
            
        Returns:
            List of matching file paths
            
        Raises:
            FileNotFoundError: If directory doesn't exist
        """
        dir_path = Path(directory)
        
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")
        
        if recursive:
            return list(dir_path.rglob(pattern))
        else:
            return list(dir_path.glob(pattern))
    
    @staticmethod
    def create_timestamped_filename(base_name: str, 
                                   extension: str,
                                   timestamp_format: str = "%Y%m%d_%H%M%S") -> str:
        """
        Create filename with timestamp.
        
        Args:
            base_name: Base filename without extension
            extension: File extension (with or without dot)
            timestamp_format: Timestamp format string
            
        Returns:
            Timestamped filename
        """
        timestamp = datetime.now().strftime(timestamp_format)
        extension = extension if extension.startswith('.') else f'.{extension}'
        return f"{base_name}_{timestamp}{extension}"
    
    @staticmethod
    def save_pickle(data: Any, 
                    file_path: Union[str, Path],
                    compress: bool = False) -> bool:
        """
        Save data using pickle with optional compression.
        
        Args:
            data: Data to pickle
            file_path: Output file path
            compress: Whether to use gzip compression
            
        Returns:
            True if successful, False otherwise
            
        Raises:
            OSError: If file cannot be written
        """
        file_path = Path(file_path)
        FileUtils.ensure_directory(file_path.parent)
        
        try:
            if compress:
                with gzip.open(file_path, 'wb') as f:
                    pickle.dump(data, f)
            else:
                with open(file_path, 'wb') as f:
                    pickle.dump(data, f)
            return True
        except (OSError, pickle.PickleError) as e:
            raise OSError(f"Failed to save pickle to {file_path}: {e}")
    
    @staticmethod
    def load_pickle(file_path: Union[str, Path],
                    compressed: bool = False) -> Any:
        """
        Load data from pickle file.
        
        Args:
            file_path: Path to pickle file
            compressed: Whether file is gzip compressed
            
        Returns:
            Unpickled data
            
        Raises:
            FileNotFoundError: If file doesn't exist
            pickle.PickleError: If pickle is invalid
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Pickle file not found: {file_path}")
        
        try:
            if compressed:
                with gzip.open(file_path, 'rb') as f:
                    return pickle.load(f)
            else:
                with open(file_path, 'rb') as f:
                    return pickle.load(f)
        except (pickle.PickleError, gzip.BadGzipFile) as e:
            raise pickle.PickleError(f"Failed to load pickle from {file_path}: {e}")

    @staticmethod
    def calculate_file_hash(file_path: Union[str, Path], algorithm: str = 'md5') -> str:
        """
        Calculate hash of a file.
        
        Args:
            file_path: Path to file
            algorithm: Hash algorithm ('md5', 'sha1', 'sha256')
        
        Returns:
            Hex digest of file hash
        
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If algorithm is not supported
        """
        import hashlib
        
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Get hash function
        if algorithm.lower() == 'md5':
            hash_func = hashlib.md5()
        elif algorithm.lower() == 'sha1':
            hash_func = hashlib.sha1()
        elif algorithm.lower() == 'sha256':
            hash_func = hashlib.sha256()
        else:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")
        
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_func.update(chunk)
            return hash_func.hexdigest()
        except OSError as e:
            raise OSError(f"Failed to calculate hash for {file_path}: {e}")