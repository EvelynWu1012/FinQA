from data_loader import download_data
from evaluation_metrics import evaluate_answer_program
from preprocessor import preprocess_dataset
import shared_data
from prompt_answer_gen_inference import MAX_SAMPLES, load_and_preprocess_data, \
    generate_answer, generate_ground_truth
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

else:
    print("Data already loaded and preprocessed. Skipping...")

# initialize_question_clusters()

# Then run evaluation
# evaluate_answer_program(url=url, num_samples=3)

# URL to download the data

# First time: Load and preprocess the data
load_and_preprocess_data(url)

# Running inference for a single question
question_text = ("what was the percent of the growth in the revenues from "
                 "2007 to 2008")
print("\n------ GPT-3.5 Response ------\n")
generate_answer(question_text)

print("\n------ Ground Truth ------")
ground_truth = generate_ground_truth(question_text)
print("Expected Program:", ground_truth["Program"])
print("Expected Answer:", ground_truth["Answer"])

