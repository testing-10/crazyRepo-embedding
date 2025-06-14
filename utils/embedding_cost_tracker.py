"""
Cost Tracker - API usage and cost tracking for embedding model testing framework
"""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from collections import defaultdict
import threading
from pathlib import Path
import json


@dataclass
class APICall:
    """Data class representing a single API call."""
    timestamp: datetime
    provider: str
    model: str
    endpoint: str
    tokens_used: int
    cost_usd: float
    duration_seconds: float
    success: bool
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class CostSummary:
    """Data class representing cost summary statistics."""
    total_calls: int
    successful_calls: int
    failed_calls: int
    total_tokens: int
    total_cost_usd: float
    total_duration_seconds: float
    average_cost_per_call: float
    average_tokens_per_call: float
    average_duration_per_call: float


class CostTracker:
    """
    Thread-safe cost and usage tracker for embedding model API calls.
    Tracks tokens, costs, timing, and provides detailed analytics.
    """

    # Default pricing per 1K tokens (USD) - can be overridden via config
    DEFAULT_PRICING = {
        'openai': {
            'text-embedding-3-small': 0.00002,
            'text-embedding-3-large': 0.00013,
            'text-embedding-ada-002': 0.0001,
        },
        'cohere': {
            'embed-english-v3.0': 0.0001,
            'embed-multilingual-v3.0': 0.0001,
        },
        'anthropic': {
            'claude-3-haiku': 0.00025,
            'claude-3-sonnet': 0.003,
        },
        'huggingface': {
            'default': 0.0,  # Usually free for open models
        },
        'sentence-transformers': {
            'default': 0.0,  # Local models, no API cost
        },
        'jina': {
            'jina-embeddings-v2-base-en': 0.00002,
            'jina-embeddings-v2-small-en': 0.00001,
        }
    }

    def __init__(self, 
                 pricing_config: Optional[Dict[str, Dict[str, float]]] = None,
                 auto_save: bool = True,
                 save_interval: int = 300):  # 5 minutes
        """
        Initialize the cost tracker.

        Args:
            pricing_config: Custom pricing configuration
            auto_save: Whether to auto-save tracking data
            save_interval: Auto-save interval in seconds
        """
        self.pricing = pricing_config or self.DEFAULT_PRICING
        self.auto_save = auto_save
        self.save_interval = save_interval

        # Thread-safe storage
        self._lock = threading.Lock()
        self._api_calls: List[APICall] = []
        self._session_start = datetime.now()
        self._session_cost = 0.0

        # Auto-save setup
        self._last_save = time.time()

    def add_usage(self, 
                  provider: str,
                  model: str,
                  tokens_used: int,
                  cost: Optional[float] = None) -> None:
        """
        Add usage tracking (simplified method for compatibility).

        Args:
            provider: API provider name
            model: Model name
            tokens_used: Number of tokens used
            cost: Optional cost override
        """
        calculated_cost = cost if cost is not None else self._calculate_cost(provider, model, tokens_used)

        with self._lock:
            self._session_cost += calculated_cost

        # Also track as API call for full tracking
        self.track_api_call(
            provider=provider,
            model=model,
            endpoint="embed",
            tokens_used=tokens_used,
            duration_seconds=0.0,
            success=True
        )

    def get_session_cost(self) -> float:
        """Get the current session cost."""
        with self._lock:
            return self._session_cost

    def track_api_call(self,
                       provider: str,
                       model: str,
                       endpoint: str,
                       tokens_used: int,
                       duration_seconds: float,
                       success: bool = True,
                       error_message: Optional[str] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> APICall:
        """
        Track a single API call.

        Args:
            provider: API provider name (e.g., 'openai', 'cohere')
            model: Model name
            endpoint: API endpoint called
            tokens_used: Number of tokens consumed
            duration_seconds: Call duration in seconds
            success: Whether the call was successful
            error_message: Error message if call failed
            metadata: Additional metadata

        Returns:
            APICall object representing the tracked call
        """
        cost_usd = self._calculate_cost(provider, model, tokens_used)

        api_call = APICall(
            timestamp=datetime.now(),
            provider=provider,
            model=model,
            endpoint=endpoint,
            tokens_used=tokens_used,
            cost_usd=cost_usd,
            duration_seconds=duration_seconds,
            success=success,
            error_message=error_message,
            metadata=metadata or {}
        )

        with self._lock:
            self._api_calls.append(api_call)
            self._session_cost += cost_usd

        # Auto-save if enabled
        if self.auto_save and time.time() - self._last_save > self.save_interval:
            self._auto_save()

        return api_call

    def _calculate_cost(self, provider: str, model: str, tokens: int) -> float:
        """
        Calculate cost for API call based on provider and model.

        Args:
            provider: API provider
            model: Model name
            tokens: Number of tokens

        Returns:
            Cost in USD
        """
        provider_pricing = self.pricing.get(provider.lower(), {})

        # Try exact model match first
        cost_per_1k = provider_pricing.get(model)

        # Fall back to default for provider
        if cost_per_1k is None:
            cost_per_1k = provider_pricing.get('default', 0.0)

        return (tokens / 1000.0) * cost_per_1k

    def get_summary(self, 
                    provider: Optional[str] = None,
                    model: Optional[str] = None,
                    time_range: Optional[tuple] = None) -> CostSummary:
        """
        Get cost summary with optional filtering.

        Args:
            provider: Filter by provider
            model: Filter by model
            time_range: Tuple of (start_time, end_time) for filtering

        Returns:
            CostSummary object
        """
        with self._lock:
            filtered_calls = self._filter_calls(provider, model, time_range)

        if not filtered_calls:
            return CostSummary(0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)

        total_calls = len(filtered_calls)
        successful_calls = sum(1 for call in filtered_calls if call.success)
        failed_calls = total_calls - successful_calls
        total_tokens = sum(call.tokens_used for call in filtered_calls)
        total_cost = sum(call.cost_usd for call in filtered_calls)
        total_duration = sum(call.duration_seconds for call in filtered_calls)

        return CostSummary(
            total_calls=total_calls,
            successful_calls=successful_calls,
            failed_calls=failed_calls,
            total_tokens=total_tokens,
            total_cost_usd=total_cost,
            total_duration_seconds=total_duration,
            average_cost_per_call=total_cost / total_calls if total_calls > 0 else 0.0,
            average_tokens_per_call=total_tokens / total_calls if total_calls > 0 else 0.0,
            average_duration_per_call=total_duration / total_calls if total_calls > 0 else 0.0
        )

    def get_provider_breakdown(self) -> Dict[str, CostSummary]:
        """
        Get cost breakdown by provider.

        Returns:
            Dictionary mapping provider names to CostSummary objects
        """
        with self._lock:
            providers = set(call.provider for call in self._api_calls)

        return {
            provider: self.get_summary(provider=provider)
            for provider in providers
        }

    def get_model_breakdown(self) -> Dict[str, CostSummary]:
        """
        Get cost breakdown by model.

        Returns:
            Dictionary mapping model names to CostSummary objects
        """
        with self._lock:
            models = set(call.model for call in self._api_calls)

        return {
            model: self.get_summary(model=model)
            for model in models
        }

    def get_hourly_usage(self, hours: int = 24) -> Dict[str, CostSummary]:
        """
        Get usage breakdown by hour for the last N hours.

        Args:
            hours: Number of hours to look back

        Returns:
            Dictionary mapping hour strings to CostSummary objects
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)

        hourly_data = {}

        for i in range(hours):
            hour_start = start_time + timedelta(hours=i)
            hour_end = hour_start + timedelta(hours=1)
            hour_key = hour_start.strftime("%Y-%m-%d %H:00")

            hourly_data[hour_key] = self.get_summary(
                time_range=(hour_start, hour_end)
            )

        return hourly_data

    def get_recent_calls(self, limit: int = 100) -> List[APICall]:
        """
        Get most recent API calls.

        Args:
            limit: Maximum number of calls to return

        Returns:
            List of recent APICall objects
        """
        with self._lock:
            return sorted(self._api_calls, key=lambda x: x.timestamp, reverse=True)[:limit]

    def get_failed_calls(self) -> List[APICall]:
        """
        Get all failed API calls.

        Returns:
            List of failed APICall objects
        """
        with self._lock:
            return [call for call in self._api_calls if not call.success]

    def _filter_calls(self,
                      provider: Optional[str] = None,
                      model: Optional[str] = None,
                      time_range: Optional[tuple] = None) -> List[APICall]:
        """Filter API calls based on criteria."""
        filtered = self._api_calls

        if provider:
            filtered = [call for call in filtered if call.provider.lower() == provider.lower()]

        if model:
            filtered = [call for call in filtered if call.model == model]

        if time_range:
            start_time, end_time = time_range
            filtered = [call for call in filtered 
                       if start_time <= call.timestamp <= end_time]

        return filtered

    def export_to_dict(self) -> Dict[str, Any]:
        """
        Export all tracking data to dictionary.

        Returns:
            Dictionary containing all tracking data
        """
        with self._lock:
            return {
                'session_start': self._session_start.isoformat(),
                'export_time': datetime.now().isoformat(),
                'total_calls': len(self._api_calls),
                'session_cost': self._session_cost,
                'api_calls': [asdict(call) for call in self._api_calls],
                'summary': asdict(self.get_summary()),
                'provider_breakdown': {
                    provider: asdict(summary) 
                    for provider, summary in self.get_provider_breakdown().items()
                },
                'model_breakdown': {
                    model: asdict(summary)
                    for model, summary in self.get_model_breakdown().items()
                }
            }

    def save_to_file(self, file_path: Union[str, Path]) -> bool:
        """
        Save tracking data to JSON file.

        Args:
            file_path: Output file path

        Returns:
            True if successful, False otherwise
        """
        try:
            from .embedding_file_utils import FileUtils
            data = self.export_to_dict()

            # Convert datetime objects to strings for JSON serialization
            def convert_datetime(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                return obj

            # Process the data to convert datetime objects
            json_data = json.loads(json.dumps(data, default=convert_datetime))

            FileUtils.save_json(json_data, file_path)
            self._last_save = time.time()
            return True
        except Exception:
            return False

    def load_from_file(self, file_path: Union[str, Path]) -> bool:
        """
        Load tracking data from JSON file.

        Args:
            file_path: Input file path

        Returns:
            True if successful, False otherwise
        """
        try:
            from .embedding_file_utils import FileUtils
            data = FileUtils.load_json(file_path)

            with self._lock:
                self._api_calls = []
                self._session_cost = data.get('session_cost', 0.0)
                for call_data in data.get('api_calls', []):
                    # Convert timestamp string back to datetime
                    call_data['timestamp'] = datetime.fromisoformat(call_data['timestamp'])
                    self._api_calls.append(APICall(**call_data))

            return True
        except Exception:
            return False

    def _auto_save(self):
        """Auto-save tracking data."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"cost_tracking_{timestamp}.json"
        self.save_to_file(Path("logs") / filename)

    def reset(self):
        """Reset all tracking data."""
        with self._lock:
            self._api_calls.clear()
            self._session_start = datetime.now()
            self._session_cost = 0.0

    def update_pricing(self, provider: str, model: str, cost_per_1k_tokens: float):
        """
        Update pricing for a specific provider/model combination.

        Args:
            provider: Provider name
            model: Model name
            cost_per_1k_tokens: Cost per 1000 tokens in USD
        """
        if provider not in self.pricing:
            self.pricing[provider] = {}
        self.pricing[provider][model] = cost_per_1k_tokens

    def get_pricing_info(self) -> Dict[str, Dict[str, float]]:
        """
        Get current pricing configuration.

        Returns:
            Current pricing dictionary
        """
        return self.pricing.copy()


# Context manager for tracking API calls
class APICallTracker:
    """Context manager for tracking individual API calls."""

    def __init__(self, 
                 cost_tracker: CostTracker,
                 provider: str,
                 model: str,
                 endpoint: str):
        self.cost_tracker = cost_tracker
        self.provider = provider
        self.model = model
        self.endpoint = endpoint
        self.start_time = None
        self.tokens_used = 0
        self.success = True
        self.error_message = None
        self.metadata = {}

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time

        if exc_type is not None:
            self.success = False
            self.error_message = str(exc_val)

        self.cost_tracker.track_api_call(
            provider=self.provider,
            model=self.model,
            endpoint=self.endpoint,
            tokens_used=self.tokens_used,
            duration_seconds=duration,
            success=self.success,
            error_message=self.error_message,
            metadata=self.metadata
        )

    def set_tokens_used(self, tokens: int):
        """Set the number of tokens used in this call."""
        self.tokens_used = tokens

    def add_metadata(self, key: str, value: Any):
        """Add metadata to this call."""
        self.metadata[key] = value
