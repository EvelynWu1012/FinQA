import re
import json
import psutil


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
        # Pair up dialogue and programs, taking the minimum length to avoid
        # index errors
        paired_steps = zip(reasoning_dialogue, turn_program)
        parts.extend([
            f"Step {i + 1}: {dialogue} : {answer}"
            for i, (dialogue, answer) in enumerate(paired_steps)
        ])

    return "\n".join(parts)


def extract_llm_response_components(llm_output):
    """
    Extracts key parts from a JSON-formatted LLM output.

    Parameters:
        llm_output (str): The LLM-generated response text in JSON format.

    Returns:
        dict: A dictionary with keys:
            'reasoning_steps', 'program', 'answer', 'confidence'
            (only if found in the input).
    """
    result = {}

    try:
        # Parse the JSON string into a dictionary
        llm_data = json.loads(llm_output)

        # Extract the relevant components, checking if each key exists
        if 'Logical Reasoning' in llm_data:
            result['reasoning_steps'] = llm_data['Logical Reasoning'].strip()
        if 'Program' in llm_data:
            result['program'] = llm_data['Program'].strip()
        if 'Answer' in llm_data:
            result['answer'] = llm_data['Answer'].strip()
        if 'Confidence' in llm_data:
            result['confidence'] = llm_data['Confidence'].strip()

    except json.JSONDecodeError:
        print("Invalid JSON input.")

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

def print_memory_usage():
    """Helper function to log memory usage"""
    mem = psutil.virtual_memory()
    print(
        f"Memory - Available: {mem.available / 1024 ** 3:.2f}GB | Used: {mem.used / 1024 ** 3:.2f}GB")
