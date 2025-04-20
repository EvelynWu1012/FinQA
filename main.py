from src.prompt_LLM import generate_answer, generate_ground_truth
from src.shared import shared_data
from src.shared.shared_data import url, MAX_SAMPLES

# === Load or preprocess dataset ===

shared_data.get_dataset(url=url,
                        max_samples=MAX_SAMPLES)

# === Load or initialize FAISS index ===
shared_data.get_search_index()

# === Run Inference ===
question_text = ("what was the percent of the growth in the revenues from "
                 "2007 to 2008")
generate_answer(question_text)
print("\n------ Ground Truth ------")
ground_truth = generate_ground_truth(question_text)
print("Expected Program:", ground_truth["Program"])
print("Expected Answer:", ground_truth["Answer"])

# Then run evaluation
# evaluate_answer_program(url=url, num_samples=3)
