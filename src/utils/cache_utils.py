import os
import pickle
from diskcache import Cache
from collections import defaultdict
import numpy as np

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Configure cache with better settings for large objects
cache = Cache(
    CACHE_DIR,
    size_limit=2 ** 30,  # 1GB
    disk_min_file_size=2 ** 20,  # 1MB min file size
    sqlite_journal_mode='WAL',  # Better write performance
    timeout=1
)


def save_cache(key, obj):
    """Store an object in the cache with smart handling for large objects"""
    try:
        # Handle large dictionaries
        if isinstance(obj, (dict, defaultdict)) and len(obj) > 5000:
            return _save_large_dict(key, obj)

        # Handle numpy arrays or embeddings
        elif isinstance(obj, np.ndarray):
            return _save_numpy_array(key, obj)

        # Default case for smaller objects
        else:
            cache[key] = obj

    except (MemoryError, pickle.PicklingError):
        # Fallback to disk storage if memory fails
        _save_to_disk(key, obj)


def _save_large_dict(key, data_dict, chunk_size=5000):
    """Save large dictionaries in chunks"""
    items = list(data_dict.items())
    for i in range(0, len(items), chunk_size):
        chunk_key = f"{key}_chunk{i // chunk_size}"
        cache[chunk_key] = dict(items[i:i + chunk_size])


def _save_numpy_array(key, array):
    """Save numpy arrays efficiently"""
    path = f"{CACHE_DIR}/{key}.npy"
    np.save(path, array, allow_pickle=False)
    cache[key] = {'__numpy_array__': path}


def _save_to_disk(key, obj):
    """Fallback storage for very large objects"""
    path = f"{CACHE_DIR}/{key}.pkl"
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    cache[key] = {'__disk_file__': path}


def load_cache(key):
    """Load an object with smart handling of large objects"""
    # First check for special storage formats
    val = cache.get(key)

    if isinstance(val, dict):
        # Handle numpy arrays
        if '__numpy_array__' in val:
            path = val['__numpy_array__']
            return np.load(path, mmap_mode='r')

        # Handle disk-stored objects
        elif '__disk_file__' in val:
            path = val['__disk_file__']
            with open(path, 'rb') as f:
                return pickle.load(f)

    # Handle chunked dictionaries
    if isinstance(val, dict) and any(
            k.startswith(f"{key}_chunk") for k in cache):
        return _load_large_dict(key)

    # Default case
    return val


def _load_large_dict(key):
    """Load chunked dictionary"""
    result = {}
    i = 0
    while True:
        chunk_key = f"{key}_chunk{i}"
        if chunk_key not in cache:
            break
        result.update(cache[chunk_key])
        i += 1
    return result or None


def cache_exists(key):
    """Check if cached object exists"""
    # Check for regular cache entry
    if key in cache:
        return True

    # Check for chunked dictionary
    if any(k.startswith(f"{key}_chunk") for k in cache):
        return True

    # Check for disk files
    for ext in ['.json', '.npy', '.pkl']:
        if os.path.exists(f"{CACHE_DIR}/{key}{ext}"):
            return True

    return False