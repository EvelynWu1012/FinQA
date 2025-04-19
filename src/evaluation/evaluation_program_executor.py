import csv
import os
from typing import Dict, Any
from src.evaluation import execute_program
from src.shared import shared_data
from src.utils import clean_text


def run_program_executor(program: str, question: str,
                         verbose: bool = True) -> Dict[str, Any]:
    """
    Executes a program and compares its result with the ground truth answer,
    with optional verbose printing.

    Args:
        program: The program string to execute
        question: The question used to retrieve the ground truth
        verbose: Whether to print detailed execution results

    Returns:
        Dictionary containing test results (same as before)
    """
    result = {
        "program": program,
        "question": question,
        "calculated_result": None,
        "ground_truth": None,
        "match": False,
        "error": None
    }

    try:
        # Get the preprocessed data
        processed_data = shared_data.processed_dataset.get(question)
        if not processed_data:
            result["error"] = f"Question '{question}' not found"
            if verbose:
                print(f"Question '{question}' not found in dataset")
            return result

        # Get the extract_info
        extract_info = processed_data
        # table = parse_table(extract_info["table"])
        raw_expected = extract_info.get("exe_ans")
        # If raw_expected is already a float
        if isinstance(raw_expected, (int, float)):
            expected_answer = float(raw_expected)
        elif isinstance(raw_expected, str):  # If raw_expected is a string
            cleaned_expected = clean_text(str(raw_expected))
            if cleaned_expected in {"yes", "no"}:
                expected_answer = cleaned_expected
            else:
                # Safely parse float without rounding first
                expected_answer = float(cleaned_expected)

        result["ground_truth"] = expected_answer

        # Execute and compare
        predicted_answer = execute_program(program, question)
        result["calculated_result"] = predicted_answer

        # Match logic (moved outside of verbose block)
        if isinstance(predicted_answer, float) and isinstance(expected_answer,
                                                              (int, float)):
            is_match = abs(predicted_answer - expected_answer) < 0.01
        else:
            is_match = predicted_answer == expected_answer

        result["match"] = is_match

        # Verbose printing
        if verbose:
            print(f"\n{'=' * 50}")
            print(f"Question: {question}")
            print(f"Program: {program}")
            print(f"Expected: {expected_answer}")
            print(f"Predicted: {predicted_answer}")
            print(f"Match: {'✅' if is_match else '❌'}")
            print(f"{'=' * 50}\n")

    except Exception as e:
        result["error"] = f"Error executing program: {str(e)}"
        print(f"Error details: {str(e)}")  # More specific error message

    return result


def run_evaluate_all_questions(output_dir="results",
                               verbose=False):
    """
    Evaluate all question-program pairs in shared_data.processed_dataset.
    Saves the results to a CSV file.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "executor_full_results.csv")

    with open(output_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=[
            "question", "program", "expected_answer",
            "predicted_answer", "is_correct", "error"
        ])
        writer.writeheader()

        total = len(shared_data.processed_dataset)
        passed = 0
        failed = 0

        for i, (question, data) in enumerate(
                shared_data.processed_dataset.items(), start=1):
            program = data.get("program")
            if not program:
                continue  # Skip entries without program

            result = run_program_executor(program, question, verbose=verbose)

            expected = result["ground_truth"]
            predicted = result["calculated_result"]
            is_correct = result["match"]
            error = result["error"]

            writer.writerow({
                "question": question,
                "program": program,
                "expected_answer": expected,
                "predicted_answer": predicted,
                "is_correct": is_correct,
                "error": error
            })

            if is_correct:
                passed += 1
            elif error:
                failed += 1

        print(f"\n✅ Completed evaluating {total} questions.")
        print(f"   Correct: {passed}")
        print(f"   Errors: {failed}")
        print(f"💾 Results saved to {output_file}")

"""
if __name__ == "__main__":
    url = "https://github.com/czyssrs/ConvFinQA/raw/main/data.zip"

    if (shared_data.processed_dataset is None or not
    shared_data.processed_dataset):
        print("Loading and preprocessing data...")
        raw = download_data(url)

        shared_data.processed_dataset = preprocess_dataset(raw, len(raw))
    else:
        print("Data already loaded and preprocessed. Skipping...")

    question = "what is actual operating cash flow reported for 2011?"
    program = "divide(221, const_1000000), subtract(3.35, #0)"
    run_program_executor(program, question)
    # Run batch test on all
    run_evaluate_all_questions(output_dir="../../results", verbose=False)
"""