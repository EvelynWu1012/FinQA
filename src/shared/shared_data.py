
import os
# === Global constants ===
MAX_SAMPLES = 3037
url = "https://github.com/czyssrs/ConvFinQA/raw/main/data.zip"
# Cache directory (diskcache handles it automatically)
CACHE_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "shared", "cache"))

