from functools import lru_cache

# Placeholder cache (can be replaced by diskcache or others)
@lru_cache(maxsize=256)
def cache_key(key: str) -> str:
    return key