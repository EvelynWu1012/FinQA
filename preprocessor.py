from typing import List, Dict
import shared_data

def preprocess_example(example: Dict) -> Dict:
    """
    Preprocess a single example and return the necessary fields.
    """
    processed_data = {}

    qa_keys = [key for key in example.keys() if key.startswith("qa")]
    pre_text = example.get("pre_text", "")
    post_text = example.get("post_text", "")

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
        }

    return processed_data


def preprocess_dataset(data: List[Dict], max_samples: int) -> Dict:
    """
    Preprocess the dataset into a dictionary where questions are the keys.
    """

    all_processed_data = {}

    for example in data[:max_samples]:
        processed = preprocess_example(example)
        all_processed_data.update(processed)

    shared_data.processed_dataset = all_processed_data  # Save to shared_data
    return shared_data.processed_dataset
