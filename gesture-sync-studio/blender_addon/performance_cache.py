"""
Performance optimizations and caching utilities for gesture generation.

Provides:
- LRU cache for audio features
- Model inference batching
- Memory-efficient processing
- Feature normalization caching
"""

import hashlib
import pickle
import time
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from functools import wraps
import logging
import numpy as np

logger = logging.getLogger(__name__)


class FeatureCache:
    """
    LRU cache for audio features to avoid recomputing.

    Uses file hash and processing parameters as cache key.
    """

    def __init__(self, max_size_mb: int = 500, cache_dir: Optional[str] = None):
        """
        Initialize feature cache.

        Args:
            max_size_mb: Maximum cache size in megabytes
            cache_dir: Directory for cache files (None = in-memory only)
        """
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.memory_cache = {}
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0
        }

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Feature cache initialized: {cache_dir} (max {max_size_mb} MB)")
        else:
            logger.info(f"Feature cache initialized: in-memory only (max {max_size_mb} MB)")

    def _compute_cache_key(self, filepath: str, params: Dict) -> str:
        """
        Compute cache key from file and parameters.

        Args:
            filepath: Path to audio file
            params: Processing parameters

        Returns:
            Cache key (hash string)
        """
        # Hash file contents
        hasher = hashlib.sha256()

        try:
            with open(filepath, 'rb') as f:
                # Read in chunks to handle large files
                for chunk in iter(lambda: f.read(8192), b''):
                    hasher.update(chunk)
        except Exception as e:
            logger.warning(f"Could not hash file {filepath}: {e}")
            # Fallback to filepath + mtime
            hasher.update(filepath.encode())
            hasher.update(str(Path(filepath).stat().st_mtime).encode())

        # Hash parameters
        param_str = str(sorted(params.items()))
        hasher.update(param_str.encode())

        return hasher.hexdigest()

    def get(self, filepath: str, params: Dict) -> Optional[Dict]:
        """
        Get cached features if available.

        Args:
            filepath: Path to audio file
            params: Processing parameters

        Returns:
            Cached features or None if not found
        """
        cache_key = self._compute_cache_key(filepath, params)

        # Check memory cache first
        if cache_key in self.memory_cache:
            self.cache_stats['hits'] += 1
            logger.debug(f"Cache hit (memory): {filepath}")
            return self.memory_cache[cache_key]['data']

        # Check disk cache if enabled
        if self.cache_dir:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        data = pickle.load(f)

                    # Load into memory cache
                    size = len(pickle.dumps(data))
                    self._add_to_memory(cache_key, data, size)

                    self.cache_stats['hits'] += 1
                    logger.debug(f"Cache hit (disk): {filepath}")
                    return data

                except Exception as e:
                    logger.warning(f"Failed to load cache file {cache_file}: {e}")

        self.cache_stats['misses'] += 1
        return None

    def put(self, filepath: str, params: Dict, features: Dict):
        """
        Store features in cache.

        Args:
            filepath: Path to audio file
            params: Processing parameters
            features: Features to cache
        """
        cache_key = self._compute_cache_key(filepath, params)

        try:
            # Serialize features
            serialized = pickle.dumps(features)
            size = len(serialized)

            # Add to memory cache
            self._add_to_memory(cache_key, features, size)

            # Save to disk if enabled
            if self.cache_dir:
                cache_file = self.cache_dir / f"{cache_key}.pkl"
                with open(cache_file, 'wb') as f:
                    f.write(serialized)

                logger.debug(f"Cached features to disk: {cache_file.name} ({size / 1024:.1f} KB)")

        except Exception as e:
            logger.warning(f"Failed to cache features: {e}")

    def _add_to_memory(self, key: str, data: Dict, size: int):
        """Add item to memory cache with LRU eviction."""
        # Evict old entries if needed
        current_size = sum(item['size'] for item in self.memory_cache.values())

        while current_size + size > self.max_size_bytes and self.memory_cache:
            # Evict oldest entry
            oldest_key = min(
                self.memory_cache.keys(),
                key=lambda k: self.memory_cache[k]['timestamp']
            )
            evicted_size = self.memory_cache[oldest_key]['size']
            del self.memory_cache[oldest_key]
            current_size -= evicted_size
            self.cache_stats['evictions'] += 1

        # Add new entry
        self.memory_cache[key] = {
            'data': data,
            'size': size,
            'timestamp': time.time()
        }

    def clear(self):
        """Clear all caches."""
        self.memory_cache.clear()

        if self.cache_dir:
            for cache_file in self.cache_dir.glob("*.pkl"):
                try:
                    cache_file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to delete cache file {cache_file}: {e}")

        logger.info("Cache cleared")

    def get_stats(self) -> Dict:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
        hit_rate = (
            self.cache_stats['hits'] / total_requests * 100
            if total_requests > 0 else 0
        )

        memory_size = sum(item['size'] for item in self.memory_cache.values())

        return {
            'hits': self.cache_stats['hits'],
            'misses': self.cache_stats['misses'],
            'evictions': self.cache_stats['evictions'],
            'hit_rate': hit_rate,
            'memory_entries': len(self.memory_cache),
            'memory_size_mb': memory_size / 1024 / 1024
        }


class BatchProcessor:
    """
    Batch processor for efficient ML inference.

    Processes multiple inputs together to maximize GPU/CPU utilization.
    """

    def __init__(self, batch_size: int = 32, timeout_ms: int = 100):
        """
        Initialize batch processor.

        Args:
            batch_size: Maximum batch size
            timeout_ms: Maximum wait time for batch accumulation
        """
        self.batch_size = batch_size
        self.timeout_ms = timeout_ms
        logger.info(f"Batch processor initialized: batch_size={batch_size}, timeout={timeout_ms}ms")

    def process_batches(
        self,
        inputs: np.ndarray,
        process_fn: Callable,
        chunk_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Process inputs in batches.

        Args:
            inputs: Input array (can be large)
            process_fn: Function to process each batch
            chunk_size: Override batch size for this call

        Returns:
            Processed outputs
        """
        batch_size = chunk_size or self.batch_size
        num_samples = len(inputs)

        if num_samples <= batch_size:
            # Small enough to process in one go
            return process_fn(inputs)

        # Process in batches
        outputs = []
        num_batches = (num_samples + batch_size - 1) // batch_size

        logger.debug(f"Processing {num_samples} samples in {num_batches} batches")

        for i in range(0, num_samples, batch_size):
            batch = inputs[i:i + batch_size]
            batch_output = process_fn(batch)
            outputs.append(batch_output)

        # Concatenate results
        return np.concatenate(outputs, axis=0)


class NormalizationCache:
    """
    Cache normalization statistics for consistent feature scaling.
    """

    def __init__(self):
        """Initialize normalization cache."""
        self.stats = {}
        logger.debug("Normalization cache initialized")

    def get_stats(self, key: str) -> Optional[Dict[str, np.ndarray]]:
        """
        Get cached normalization statistics.

        Args:
            key: Cache key (e.g., model name)

        Returns:
            Dictionary with 'mean' and 'std' arrays, or None
        """
        return self.stats.get(key)

    def put_stats(self, key: str, mean: np.ndarray, std: np.ndarray):
        """
        Cache normalization statistics.

        Args:
            key: Cache key
            mean: Mean values
            std: Standard deviation values
        """
        self.stats[key] = {
            'mean': mean.copy(),
            'std': std.copy()
        }
        logger.debug(f"Cached normalization stats: {key}")

    def clear(self):
        """Clear cache."""
        self.stats.clear()


def timed_operation(operation_name: str):
    """
    Decorator to log operation timing.

    Args:
        operation_name: Name of operation for logging
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start

            logger.debug(f"{operation_name} took {elapsed*1000:.1f}ms")

            return result
        return wrapper
    return decorator


def memory_efficient_process(
    data: np.ndarray,
    process_fn: Callable,
    chunk_size: int = 1000,
    axis: int = 0
) -> np.ndarray:
    """
    Process large arrays in chunks to reduce memory usage.

    Args:
        data: Large input array
        process_fn: Function to process each chunk
        chunk_size: Size of each chunk
        axis: Axis along which to chunk

    Returns:
        Processed array
    """
    if data.shape[axis] <= chunk_size:
        return process_fn(data)

    # Process in chunks
    num_chunks = (data.shape[axis] + chunk_size - 1) // chunk_size
    logger.debug(f"Processing array of shape {data.shape} in {num_chunks} chunks")

    results = []

    for i in range(0, data.shape[axis], chunk_size):
        if axis == 0:
            chunk = data[i:i + chunk_size]
        elif axis == 1:
            chunk = data[:, i:i + chunk_size]
        else:
            raise ValueError(f"Unsupported axis: {axis}")

        result = process_fn(chunk)
        results.append(result)

    # Concatenate results
    return np.concatenate(results, axis=axis)


class PerformanceMonitor:
    """
    Monitor and report performance metrics.
    """

    def __init__(self):
        """Initialize performance monitor."""
        self.metrics = {}
        self.start_times = {}

    def start(self, operation: str):
        """
        Start timing an operation.

        Args:
            operation: Operation name
        """
        self.start_times[operation] = time.time()

    def end(self, operation: str):
        """
        End timing an operation.

        Args:
            operation: Operation name
        """
        if operation not in self.start_times:
            logger.warning(f"No start time for operation: {operation}")
            return

        elapsed = time.time() - self.start_times[operation]

        if operation not in self.metrics:
            self.metrics[operation] = []

        self.metrics[operation].append(elapsed)
        del self.start_times[operation]

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get performance summary.

        Returns:
            Dictionary mapping operation names to statistics
        """
        summary = {}

        for operation, times in self.metrics.items():
            if not times:
                continue

            summary[operation] = {
                'count': len(times),
                'total': sum(times),
                'mean': np.mean(times),
                'std': np.std(times),
                'min': min(times),
                'max': max(times)
            }

        return summary

    def print_summary(self):
        """Print performance summary to log."""
        summary = self.get_summary()

        if not summary:
            logger.info("No performance metrics recorded")
            return

        logger.info("=" * 60)
        logger.info("PERFORMANCE SUMMARY")
        logger.info("=" * 60)

        for operation, stats in summary.items():
            logger.info(
                f"{operation:30s}: {stats['mean']*1000:7.1f}ms ± {stats['std']*1000:6.1f}ms "
                f"({stats['count']} calls, total: {stats['total']:.2f}s)"
            )

        logger.info("=" * 60)

    def clear(self):
        """Clear all metrics."""
        self.metrics.clear()
        self.start_times.clear()


# Global instances
_feature_cache = None
_normalization_cache = NormalizationCache()
_performance_monitor = PerformanceMonitor()


def get_feature_cache(max_size_mb: int = 500, cache_dir: Optional[str] = None) -> FeatureCache:
    """
    Get or create global feature cache.

    Args:
        max_size_mb: Maximum cache size in MB
        cache_dir: Cache directory (None = in-memory only)

    Returns:
        FeatureCache instance
    """
    global _feature_cache

    if _feature_cache is None:
        _feature_cache = FeatureCache(max_size_mb, cache_dir)

    return _feature_cache


def get_normalization_cache() -> NormalizationCache:
    """Get global normalization cache."""
    return _normalization_cache


def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor."""
    return _performance_monitor
