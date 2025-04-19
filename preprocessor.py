from itertools import islice
from typing import Dict
import shared_data
from data_loader import download_data


def preprocess_example(example: Dict) -> Dict:
    """
    Preprocess a single example and return the necessary fields.
    """
    processed_data = {}

    qa_keys = [key for key in example.keys() if key.startswith("qa")]
    pre_text = example.get("pre_text", "")
    post_text = example.get("post_text", "")
    annotation = example.get("annotation", {})
    reasoning_dialogue = annotation.get("dialogue_break", [])
    turn_program = annotation.get("turn_program", [])

    for qa_key in qa_keys:
        question = example[qa_key]["question"]
        table = example["table"]
        ann_table_rows = example[qa_key].get("ann_table_rows", [])
        ann_text_rows = example[qa_key].get("ann_text_rows", [])

        processed_data[question] = {
            "question": question,
            "table": table,
            "focused_table_row": ann_table_rows,
            "focused_text_row": ann_text_rows,
            "steps": example[qa_key].get("steps", []),
            "program": example[qa_key].get("program", ""),
            "exe_ans": example[qa_key].get("exe_ans"),
            "answer": example[qa_key].get("answer"),
            "pre_text": pre_text,
            "post_text": post_text,
            "reasoning_dialogue": reasoning_dialogue,
            "turn_program": turn_program
        }

    return processed_data


def preprocess_dataset(data: Dict, max_samples: int) -> Dict:
    """
    Preprocess the dataset into a dictionary where:
    - Key: question (str)
    - Value:
    Dictionary containing all processed data(include program variants)
    """

    all_processed_data = {}

    for example in data[:max_samples]:
        # This returns {(q,p): data} dict
        processed = preprocess_example(example)
        all_processed_data.update(processed)

    shared_data.processed_dataset = all_processed_data  # Save to shared_data
    # Extract questions for later use
    shared_data.questions = list(all_processed_data.keys())
    return shared_data.processed_dataset

if __name__ == "__main__":
    url = "https://github.com/czyssrs/ConvFinQA/raw/main/data.zip"

    if (shared_data.processed_dataset is None or not
    shared_data.processed_dataset):
        print("Loading and preprocessing data...")
        raw = download_data(url)

        shared_data.processed_dataset = preprocess_dataset(raw, len(raw))
    else:
        print("Data already loaded and preprocessed. Skipping...")

    # After processing the dataset, add this code to print the first two items
    if shared_data.processed_dataset:
        print("\n=== First two items in processed_dataset ===")
        for i, (question, qa_data) in enumerate(
                islice(shared_data.processed_dataset.items(), 5)):
            print(f"\nItem {i + 1}:")
            print(f"Question: {question}")
            print(f"Answer: {qa_data.get('answer', 'N/A')}")
            print(f"Table data: {qa_data.get('table', 'N/A')}")
            print(
                f"Dialogue context: {qa_data.get("reasoning_dialogue", 'N/A')}")
            print(f"Program context: {qa_data.get("turn_program", 'N/A')}")
            print(f"Execution answer: {qa_data.get('exe_ans', 'N/A')}")
            print(f"Program: {qa_data.get('program', 'N/A')}")
