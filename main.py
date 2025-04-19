from src.evaluation import evaluate_answer_program
from src.shared import *
from src.prompt_LLM.prompt_answer_gen_inference import *

url = "https://github.com/czyssrs/ConvFinQA/raw/main/data.zip"

if not processed_dataset:
    print("Loading and preprocessing data...")
    raw = download_data(url)
    processed_dataset = preprocess_dataset(raw, MAX_SAMPLES)  # Adjust
    # MAX_SAMPLES as necessary
    questions = list(processed_dataset.keys())  # Explicitly update questions

else:
    print("Data already loaded and preprocessed. Skipping...")

if (not shared_data.question_to_cluster_label or not shared_data.cluster_idx_to_questions):
    print("Initializing clustering...")
    initialize_question_clusters()
else:
    print("Clustering already initialized.")


# Running inference for a single question
question_text = ("what was the percent of the growth in the revenues from "
                 "2007 to 2008")
print("\n------ GPT-3.5 Response ------\n")
generate_answer(question_text)

print("\n------ Ground Truth ------")
# ground_truth = generate_ground_truth(question_text)
print("Expected Program:", ground_truth["Program"])
print("Expected Answer:", ground_truth["Answer"])

# Then run evaluation
# evaluate_answer_program(url=url, num_samples=3)