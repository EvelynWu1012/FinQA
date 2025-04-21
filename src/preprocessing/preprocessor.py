from typing import Dict


def preprocess_example(example: Dict) -> Dict[str, Dict]:
    """
    Preprocess a single example from the dataset and return the necessary
    fields in a structured format.

    This function extracts relevant information from the given example,
    including question-answer pairs,
    table data, annotated rows, and additional context (pre_text, post_text,
    etc.). The output is a
    dictionary containing the processed data, ready for further use or
    evaluation.

    Parameters:
        example (Dict): The raw example to be processed. Expected to contain
        keys such as 'qaX',
                         'pre_text', 'post_text', and 'annotation' with
                         various nested values.

    Returns:
        Dict: The processed data for each question in the example, including
        associated table rows,
              reasoning dialogue, steps, programs, and answers.
    """

    processed_data = {}  # Initialize an empty dictionary to store processed
    # data

    # Identify all keys in the example that start with 'qa' (indicating
    # question-answer pairs)
    qa_keys = [key for key in example.keys() if key.startswith("qa")]

    # Retrieve common fields like pre_text, post_text, and annotation from
    # the example
    pre_text = example.get("pre_text", "")
    post_text = example.get("post_text", "")
    annotation = example.get("annotation", {})
    # Extract the table associated with the example
    table = example.get("table", [])

    # Extract reasoning dialogue and turn_program from the annotation if
    # available
    reasoning_dialogue = annotation.get("dialogue_break", [])
    turn_program = annotation.get("turn_program", [])

    # Iterate over each question-answer pair (identified by keys starting
    # with 'qa')
    for qa_key in qa_keys:
        qa_pair = example[qa_key]
        if "question" not in qa_pair or "answer" not in qa_pair:
            continue  # Skip malformed QA pairs or handle differently
        # Extract specific fields for each question-answer pair
        # Get the question for this QA pair
        question = example[qa_key]["question"]

        # Annotated table rows (default to empty list)
        ann_table_rows = example[qa_key].get("ann_table_rows", [])
        # Annotated text rows (default to empty list)
        ann_text_rows = example[qa_key].get("ann_text_rows", [])

        # Construct the processed data for the current question-answer pair
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
    # Return the fully processed data for all questions in the example
    return processed_data


def preprocess_dataset(data: Dict, max_samples: int) -> Dict:
    """
    Preprocess the given dataset into a dictionary where each key is a
    question,
    and the value is a dictionary containing all the processed data, including
    program variants associated with that question.

    The dataset is preprocessed by iterating over a subset of samples (up to
    `max_samples`),
    and for each example, the function preprocesses the data using the
    `preprocess_example` function.

    Parameters:
        data (Dict): The raw dataset to preprocess. Expected to be a list of
        examples.
        max_samples (int): The maximum number of examples to process from
        the dataset.

    Returns:
        Dict: A dictionary where the keys are questions (str), and the
        values are
              dictionaries containing the processed data for each question.
    """
    from src.shared import shared_data

    # Initialize an empty dictionary to store all processed data
    all_processed_data = {}

    # Iterate through each example in the dataset, limited by `max_samples`
    for example in data[:max_samples]:
        # Process each example and get the processed data as a dictionary
        # where the keys are questions and the values are associated data
        processed = preprocess_example(example)

        # Update the `all_processed_data` dictionary with the new processed
        # data
        all_processed_data.update(processed)

    # Store the processed dataset in shared_data for later use
    shared_data.processed_dataset = all_processed_data

    # Extract the list of questions (keys from the processed data) for later
    # use
    shared_data.questions = list(all_processed_data.keys())

    # Return the processed dataset
    return shared_data.processed_dataset

