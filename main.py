import numpy as np

from src.data_loader import download_data
from src.preprocessing import preprocess_dataset
from src.prompt_LLM import initialize_question_clusters, generate_answer, \
    generate_ground_truth
from src.shared import shared_data
from src.utils import cache_exists, load_cache, save_cache
import psutil

# Global variable to hold the max_samples value
MAX_SAMPLES = 3037

# Cache directory (diskcache handles it automatically)
CACHE_DIR = "cache"

url = "https://github.com/czyssrs/ConvFinQA/raw/main/data.zip"


def print_memory_usage():
    """Helper function to log memory usage"""
    mem = psutil.virtual_memory()
    print(
        f"Memory - Available: {mem.available / 1024 ** 3:.2f}GB | Used: {mem.used / 1024 ** 3:.2f}GB")


# === Load or preprocess dataset ===
print_memory_usage()
if cache_exists('processed_dataset') and cache_exists('questions'):
    print("Loading cached data...")
    shared_data.processed_dataset = load_cache('processed_dataset')
    shared_data.questions = load_cache('questions')
    print("Cached processed dataset loaded")
else:
    print("Downloading and preprocessing data...")
    raw = download_data(url)
    shared_data.processed_dataset = preprocess_dataset(raw, MAX_SAMPLES)
    shared_data.questions = list(shared_data.processed_dataset.keys())
    save_cache('processed_dataset', shared_data.processed_dataset)
    save_cache('questions', shared_data.questions)

print_memory_usage()

# === Load or initialize clustering ===
if (cache_exists('question_to_cluster_label') and
        cache_exists('cluster_idx_to_questions') and
        cache_exists('question_embeddings')):
    print("Loading cached clustering...")
    shared_data.question_to_cluster_label = load_cache(
        'question_to_cluster_label')
    shared_data.cluster_idx_to_questions = load_cache(
        'cluster_idx_to_questions')

    # Special handling for embeddings
    embeddings = load_cache('question_embeddings')
    if isinstance(embeddings, np.ndarray):
        shared_data.question_embeddings = embeddings
    else:
        # Handle case where embeddings might be memory-mapped
        shared_data.question_embeddings = np.load(embeddings, mmap_mode='r')
else:
    print("Initializing question clustering...")
    print_memory_usage()

    # Initialize with smaller batch size if memory is low
    mem = psutil.virtual_memory()
    batch_size = 512 if mem.available < 4 * 1024 ** 3 else 1024  # 4GB threshold

    initialize_question_clusters()

    # Save with new caching system
    save_cache('question_to_cluster_label',
               shared_data.question_to_cluster_label)
    save_cache('cluster_idx_to_questions',
               shared_data.cluster_idx_to_questions)

    # Special handling for embeddings
    if isinstance(shared_data.question_embeddings, np.ndarray):
        save_cache('question_embeddings', shared_data.question_embeddings)
    else:
        # Convert to numpy array if needed
        emb = shared_data.question_embeddings.numpy() if hasattr(
            shared_data.question_embeddings,
            'numpy') else shared_data.question_embeddings
        save_cache('question_embeddings', emb)

print_memory_usage()

# === Run Inference ===
question_text = ("what was the percent of the growth in the revenues from "
                 "2007 to 2008")
print("\n------ GPT-3.5 Response ------\n")
generate_answer(question_text)
print("\n------ Ground Truth ------")
ground_truth = generate_ground_truth(question_text)
print("Expected Program:", ground_truth["Program"])
print("Expected Answer:", ground_truth["Answer"])

# Then run evaluation
# evaluate_answer_program(url=url, num_samples=3)