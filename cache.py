import threading
import time
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


# Global cache instances
stock_data_cache = TTLCache()
prediction_cache = TTLCache()
