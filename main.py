from evaluation_metrics import evaluate_exact_match
from prompt_answer_gen_inference import load_and_preprocess_data

url = "https://github.com/czyssrs/ConvFinQA/raw/main/data.zip"

# Ensure data is loaded first
load_and_preprocess_data(url)

# Then run evaluation
evaluate_exact_match(url=url, num_samples=3)