from data_loader import download_data
from evaluation_metrics import evaluate_exact_match
from preprocessor import preprocess_dataset
import shared_data
from prompt_answer_gen_inference import MAX_SAMPLES
from prompt_example_selector import prepare_questions, \
    initialize_question_clusters


url = "https://github.com/czyssrs/ConvFinQA/raw/main/data.zip"

if not shared_data.processed_dataset:
    print("Loading and preprocessing data...")
    raw = download_data(url)
    shared_data.processed_dataset = preprocess_dataset(raw,
                                                       MAX_SAMPLES)  # Adjust
    # MAX_SAMPLES as necessary
    shared_data.questions = list(
        shared_data.processed_dataset.keys())  # Explicitly update questions

    # print to check if it's populated
else:
    print("Data already loaded and preprocessed. Skipping...")
    print(
        f"Questions in shared_data: {len(shared_data.questions)}")  # Debug
    # print

initialize_question_clusters()

# Then run evaluation
evaluate_exact_match(url=url, num_samples=3)
