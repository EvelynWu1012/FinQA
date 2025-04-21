from src.utils import cache_exists, load_cache, save_cache
from src.data_loader import download_data
import os
import faiss
import numpy as np
import psutil
from src.shared.shared_data import MAX_SAMPLES, url, CACHE_DIR

class SharedData:
    """A class to manage and store shared data for the application."""

    def __init__(self):
        self.processed_dataset = {}
        self.questions = []
        self.question_embeddings = None
        self.faiss_index = None

    # Methods for dataset, FAISS index, and embeddings remain unchanged
    def get_dataset(self, url, max_samples=MAX_SAMPLES):
        # Delay the import to avoid circular dependencies
        from src.preprocessing import preprocess_dataset
        if cache_exists('processed_dataset') and cache_exists('questions'):
            print("Loading cached data...")
            self.processed_dataset = load_cache('processed_dataset')
            self.questions = load_cache('questions')
        else:
            print("Downloading and preprocessing data...")
            raw = download_data(url)
            self.processed_dataset = preprocess_dataset(raw, max_samples)
            self.questions = list(self.processed_dataset.keys())
            save_cache('processed_dataset', self.processed_dataset)
            save_cache('questions', self.questions)

    def set_faiss_index(self, faiss_index):
        self.faiss_index = faiss_index

    def set_question_embeddings(self, embeddings):
        self.question_embeddings = embeddings

    def get_search_index(self):
        from src.prompt_LLM.prompt_shots_selector import initialize_faiss_index
        def save_faiss_index(index):
            faiss_path = os.path.join(CACHE_DIR, "faiss.index")
            faiss.write_index(index, faiss_path)
            save_cache('faiss_index', faiss_path)
            return faiss_path


        if cache_exists('question_embeddings') and cache_exists('faiss_index'):
            print("Loading cached FAISS index and question embeddings...")
            # Load question embeddings
            embeddings = load_cache('question_embeddings')
            self.question_embeddings = (
                embeddings if isinstance(embeddings, np.ndarray)
                else np.load(embeddings, mmap_mode='r')
            )

            # Load faiss index
            index_path = load_cache('faiss_index')
            if index_path and os.path.exists(index_path):
                self.faiss_index = faiss.read_index(index_path)
            else:
                print("⚠️ FAISS index path is missing. Reinitializing...")
                self.faiss_index, self.question_embeddings = initialize_faiss_index()
                save_faiss_index(self.faiss_index)
        else:
            print("Initializing FAISS index...")
            self.faiss_index, self.question_embeddings = initialize_faiss_index()

            if isinstance(self.question_embeddings, np.ndarray):
                save_cache('question_embeddings', self.question_embeddings)

            save_faiss_index(self.faiss_index)


# Initialize shared_data instance
shared_data = SharedData()
