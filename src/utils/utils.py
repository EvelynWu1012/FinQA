import csv
import os
import re
from typing import Dict, Any
from src.shared import shared_data
from src.data_loader.data_loader import download_data
from src.preprocessing.preprocessor import preprocess_dataset
from src.evaluation.program_executor import execute_program


def clean_text(text: str) -> str:
    """
    Cleans input text by removing all non-digit characters,
    while preserving decimal points and minus signs.
    If the text is 'yes' or 'no', returns the text unchanged.

    Args:
    - text (str): Input text to be cleaned.

    Returns:
    - str: Cleaned text containing only digits, decimal points, and minus
    signs,
           or the original text if it's 'yes' or 'no'.
    """
    # Check if the text is 'yes' or 'no' and return it unchanged
    if text.lower() in ['yes', 'no']:
        return text.lower()

    # Remove all non-digit characters except for decimal points and minus signs
    text = re.sub(r"[^0-9.-]", "", text)
    return text


def format_table(table):
    """
    Formats focused table rows into a readable string.

    Args:
    - focused_rows (List[List[str]]): Table rows with the header.

    Returns:
    - str: Formatted table string.
    """
    if not table:
        return ""
    header = table[0]
    rows = table[1:]
    header_line = " Rows | ".join(header)
    row_lines = [" | ".join(row) for row in rows]
    return header_line + "\n" + "\n".join(row_lines)


def construct_chain_of_thought(data):
    parts = []

    # Step 1: Understand the Problem
    question = data.get("question", "")
    parts.append("1. Understand the Problem")
    parts.append(f"Rephrase: What is being asked is - '{question}'.")

    # Step 2: Break Down the Problem
    parts.append("\n2. Break Down the Problem")
    parts.append("Sub-tasks:")

    focused_table_row = data.get("focused_table_row", [])
    focused_text_row = data.get("focused_text_row", [])
    if focused_table_row:
        for row in focused_table_row:
            parts.append(f"- Identify and analyze table row : '{row}'")
    if focused_text_row:
        for row in focused_text_row:
            parts.append(f"- Identify and analyze text row : '{row}'")

    parts.append(
        "- Locate the column in the table that contains relevant information.")
    parts.append("- Extract and interpret the corresponding value(s).")

    # Step 3: Apply Logical Reasoning
    parts.append("\n3. Apply Logical Reasoning")
    reasoning_dialogue = data.get("reasoning_dialogue", [])
    turn_program = data.get("turn_program", [])
    # steps = data.get("steps", [])
    if (not reasoning_dialogue) or (not turn_program):
        return "No reasoning steps available."
    else:
        # Pair up dialogue and programs, taking the minimum length to avoid index errors
        paired_steps = zip(reasoning_dialogue, turn_program)
        parts.extend([
            f"Step {i + 1}: {dialogue} : {answer}"
            for i, (dialogue, answer) in enumerate(paired_steps)
        ])

    return "\n".join(parts)


def extract_llm_response_components(llm_output):
    """
    Extracts key parts from a formatted LLM output.

    Parameters:
        llm_output (str): The LLM-generated response text.

    Returns:
        dict: A dictionary with keys:
            'reasoning_steps', 'program', 'answer', 'confidence'
            (only if found in the input).
    """
    result = {}

    reasoning_match = re.search(
        r'\*\*Reasoning Steps:\*\*\s*(.*?)\s*('
        r'?=\*\*Program:|\*\*Answer:|\*\*Confidence:|\Z)',
        llm_output, re.DOTALL)
    program_match = re.search(
        r'\*\*Program:\*\*\s*(.*?)\s*(?=\*\*Answer:|\*\*Confidence:|\Z)',
        llm_output, re.DOTALL)
    answer_match = re.search(
        r'\*\*Answer:\*\*\s*(.*?)\s*(?=\*\*Confidence:|\Z)', llm_output,
        re.DOTALL)
    confidence_match = re.search(r'\*\*Confidence:\*\*\s*(.+)', llm_output)

    if reasoning_match:
        result['reasoning_steps'] = reasoning_match.group(1).strip()
    if program_match:
        result['program'] = program_match.group(1).strip()
    if answer_match:
        result['answer'] = answer_match.group(1).strip()
    if confidence_match:
        result['confidence'] = confidence_match.group(1).strip()

    return result


def is_numeric(s):
    """
    Checks if a string can be converted to a number (integer, float,
    or scientific notation).

    Args:
        s (str): The string to be checked.

    Returns:
        bool: True if the string can be converted to a number, False otherwise.

    Example:
        can_be_converted_to_number("3.14") -> True
        can_be_converted_to_number("-123") -> True
        can_be_converted_to_number("abc") -> False
    """
    try:
        float(s)  # Tries to convert the string to a float
        return True
    except ValueError:
        return False


def format_executable_answer(executable_answer):
    """
    Formats a programmatically computed answer into a standardized format
    (float or 'yes'/'no' string) for comparison purposes.

    Parameters:
        executable_answer (int | float | str): The answer computed by a
        program.

    Returns:
        float | str: The formatted answer, either as a float or a normalized
        'yes'/'no' string.
    """
    if isinstance(executable_answer, (int, float)):
        return float(executable_answer)

    if isinstance(executable_answer, str):
        cleaned_answer = clean_text(executable_answer.strip().lower())
        if cleaned_answer in {"yes", "no"}:
            return cleaned_answer
        try:
            return float(cleaned_answer)
        except ValueError:
            raise ValueError(
                f"Cannot convert cleaned answer to float: '{cleaned_answer}'")

    raise TypeError(
        "Unsupported type for executable_answer. Must be int, float, or str.")


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
