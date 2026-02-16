import threading
import time
import pickle
import os
from typing import Any, Optional

from config import STOCK_DATA_CACHE_TTL, PREDICTION_CACHE_TTL


class TTLCache:
    """Thread-safe in-memory cache with TTL expiration."""

    def __init__(self):
        self._cache: dict[str, tuple[Any, float]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        """Get value if exists and not expired, else None."""
        with self._lock:
            if key not in self._cache:
                return None
            value, expiry = self._cache[key]
            if time.time() > expiry:
                del self._cache[key]
                return None
            return value

    def set(self, key: str, value: Any, ttl: int) -> None:
        """Set value with TTL in seconds."""
        with self._lock:
            expiry = time.time() + ttl
            self._cache[key] = (value, expiry)


class PersistentTTLCache(TTLCache):
    """
    TTL cache with file-based persistence for stock data.

    Survives server restarts by saving to disk.
    Uses pickle for serialization to support DataFrames and other complex types.
    """

    def __init__(self, cache_file: str = ".cache/stock_data_cache.pkl"):
        super().__init__()
        self._cache_file = cache_file
        self._dirty = False
        self._save_lock = threading.Lock()
        self._load_from_disk()

    def _load_from_disk(self):
        """Load cache from disk on startup."""
        try:
            if os.path.exists(self._cache_file):
                with open(self._cache_file, 'rb') as f:
                    data = pickle.load(f)
                    now = time.time()
                    # Only load non-expired entries
                    for key, (value, expiry) in data.items():
                        if expiry > now:
                            self._cache[key] = (value, expiry)
                    print(f"[Cache] Loaded {len(self._cache)} entries from disk")
        except Exception as e:
            print(f"[Cache] Could not load cache from disk: {e}")

    def _save_to_disk(self):
        """Save cache to disk using pickle."""
        with self._save_lock:
            try:
                # Ensure directory exists
                os.makedirs(os.path.dirname(self._cache_file), exist_ok=True)

                # Only save non-expired entries
                now = time.time()
                to_save = {k: v for k, v in self._cache.items() if v[1] > now}

                with open(self._cache_file, 'wb') as f:
                    pickle.dump(to_save, f)
                self._dirty = False
            except Exception as e:
                print(f"[Cache] Could not save cache to disk: {e}")

    def set(self, key: str, value: Any, ttl: int) -> None:
        """Set value with TTL and mark for persistence."""
        super().set(key, value, ttl)
        self._dirty = True
        # Save to disk periodically (every 10 new entries)
        if len(self._cache) % 10 == 0:
            self._save_to_disk()

    def save(self):
        """Force save to disk."""
        if self._dirty:
            self._save_to_disk()


# Global cache instances
# Use persistent cache for stock data (survives restarts, supports DataFrames)
stock_data_cache = PersistentTTLCache(".cache/stock_data_cache.pkl")
# Regular in-memory cache for predictions (computed quickly)
prediction_cache = TTLCache()
