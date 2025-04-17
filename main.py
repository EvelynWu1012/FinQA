from evaluation_metrics import evaluate_answer_match
from prompt_answer_gen_inference import load_and_preprocess_data

url = "https://github.com/czyssrs/ConvFinQA/raw/main/data.zip"

# Ensure data is loaded first
load_and_preprocess_data(url)

# Then run evaluation
evaluate_answer_match(url="your_data_url", num_samples=2)