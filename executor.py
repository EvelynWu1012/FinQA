import json
import os
import re
import zipfile
import requests

"""
Step 1: Understand the Data Format. 
"""


def load_data(url: str, zip_file_path: str = "data.zip",
              extract_to: str = "data", json_file: str = "train.json"):
    """
    Downloads a zip file from the given URL, extracts it, and loads the
    specified JSON file.

    Args:
    - url (str): URL to download the zip file from.
    - zip_file_path (str): Path where the zip file will be saved.
    - extract_to (str): Directory to extract the contents of the zip file.
    - json_file (str): The name of the JSON file to load from the extracted
    folder.

    Returns:
    - data (dict): The data loaded from the specified JSON file.
    """
    # 1. Download the zip file
    print(f"Downloading {zip_file_path} from {url}...")
    # Send a GET request to the provided URL
    response = requests.get(url)
    # Open the specified path to save the zip file in write-binary mode.
    with open(zip_file_path, "wb") as f:
        # Write the content of the response(the downloaded zip file) to the
        # local file
        f.write(response.content)
    print(f"Download complete: {zip_file_path}")

    # 2. Unzip the file
    print(f"Unzipping {zip_file_path} into {extract_to}...")
    # Open the zip file in read mode
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        # Extract all the contents of the zip file into the specified directory
        zip_ref.extractall(extract_to)
    print(f"Unzip complete. Files extracted to: {extract_to}")

    # 3. Load the JSON file
    # Create the full path to the JSON file inside the extracted folder
    json_path = os.path.join(extract_to, "data", json_file)
    print(f"Loading JSON data from {json_path}...")
    with open(json_path, "r") as f:  # Open the JSON file in read mode
        # Load and parse the JSON data from the file into a Python dict
        data = json.load(f)
    print(f"Loaded {len(data)} examples from {json_file}.")

    return data


"""
Step 2: Table Parser Design. 
"""


def parse_table(raw_table):
    """
    Converts raw table data into a list of dictionaries where each row is
    represented as a dictionary
    :param raw_table
    :return: dictionary e.g. [{"Year": "2021", "Revenue": "206588", "Cost":
    "181001", "Profit": "25587"}]
    """

    headers = raw_table[0]
    rows = raw_table[1:]
    parsed = []
    for row in rows:
        # Pad the row if it's shorter than headers
        if len(row) < len(headers):
            row += [''] * (len(headers) - len(row))
        # Truncate the row if it's longer than headers
        row = row[:len(headers)]
        parsed.append(dict(zip(headers, row)))

    return parsed


"""
Step 3: Build the Program Executor
"""


def eval_expr(expression, table, memory):
    """
    Evaluates a single arithmetic expression involving basic operations:
    add, subtract, multiply, divide.

    The expression can involve:
        - Constants (e.g., "const_100")
        - Table references (e.g., "table[0][\"Revenue\"]")
        - Memory references (e.g., "#0", "#1") which are previously computed
        intermediate results.

    Parameters:
        expression (str): A string expression like 'subtract(#0, 181001)'.
        table (List[Dict]): A list of dictionaries representing a table (
        e.g., parsed from HTML or CSV).
        memory (Dict[str, float]): A dictionary storing intermediate
        computation results by keys like '#0', '#1'.

    Returns:
        float: The result of evaluating the arithmetic expression.

    Raises:
        ZeroDivisionError: If a divide operation attempts to divide by zero.
        ValueError: If the expression format is unknown or operands can't be
        resolved.
    """
    # Remove any whitespace to simplify parsing (e.g., 'subtract( #0 ,
    # 100 )' becomes 'subtract(#0,100)')
    expression = expression.replace(" ", "")

    # Handle addition: 'add(a, b)'
    if expression.startswith("add("):
        operands = expression[4:-1].split(
            ",")  # Extract 'a' and 'b' from 'add(a,b)'
        val1 = resolve_value(operands[0], table, memory)
        val2 = resolve_value(operands[1], table, memory)
        result = val1 + val2
        print(f"Intermediate: {expression} → {val1} + {val2} = {result}")
        return result

    # Handle subtraction: 'subtract(a, b)'
    elif expression.startswith("subtract("):
        operands = expression[9:-1].split(
            ",")  # Extract 'a' and 'b' from 'subtract(a,b)'
        val1 = resolve_value(operands[0], table, memory)
        val2 = resolve_value(operands[1], table, memory)
        result = val1 - val2
        print(f"Intermediate: {expression} → {val1} - {val2} = {result}")
        return result

    # Handle multiplication: 'multiply(a, b)'
    elif expression.startswith("multiply("):
        operands = expression[9:-1].split(
            ",")  # Extract 'a' and 'b' from 'multiply(a,b)'
        val1 = resolve_value(operands[0], table, memory)
        val2 = resolve_value(operands[1], table, memory)
        result = val1 * val2
        print(f"Intermediate: {expression} → {val1} * {val2} = {result}")
        return result

    # Handle division: 'divide(a, b)'
    elif expression.startswith("divide("):
        operands = expression[7:-1].split(
            ",")  # Extract 'a' and 'b' from 'divide(a,b)'
        val1 = resolve_value(operands[0], table, memory)
        denominator = resolve_value(operands[1], table, memory)

        # Check for division by zero
        if denominator == 0:
            print(f"Warning: Division by zero in expression: {expression}")
            return float('nan')

        result = val1 / denominator
        print(
            f"Intermediate: {expression} → {val1} / {denominator} = {result}")
        return result

    # Handle exponentiation: exp(a, b) = a ** b
    elif expression.startswith("exp("):
        operands = expression[4:-1].split(",")
        if len(operands) != 2:
            raise ValueError(
                f"Invalid number of arguments in expression: {expression}")
        base = resolve_value(operands[0], table, memory)
        exponent = resolve_value(operands[1], table, memory)
        result = base ** exponent
        print(f"Intermediate: {expression} → {base} ** {exponent} = {result}")
        return result

    # Handle greater
    elif expression.startswith("greater("):
        operands = expression[8:-1].split(",")
        val1 = resolve_value(operands[0], table, memory)
        val2 = resolve_value(operands[1], table, memory)
        result = "yes" if val1 > val2 else "no"
        print(f"Intermediate: {expression} → {val1} > {val2} = {result}")
        return result

    else:
        # If it's not a recognized operation, attempt to directly resolve it
        # (e.g., "#0", "const_100", or "table[0][\"Revenue\"]")
        result = resolve_value(expression, table, memory)
        print(f"Intermediate: {expression} → Resolved to {result}")
        return result

def resolve_value(value, table, memory):
    """
    Resolves the input `value` to a numerical float.

    The function supports:
    - Memory references (e.g., "#0", "#1") from previous steps in `memory`
    - Named constants (e.g., "const_100" → 100.0, "const_m1" → -1.0)
    - Raw numeric strings (e.g., "75.95", "-10")
    - Percentages (e.g., "4.02%" → 0.0402)
    Parameters:
        value (str | int | float): The value or reference to resolve.
        table (list[dict]): A table of rows (dicts) to lookup values if needed.
        memory (dict): Dictionary storing intermediate computed values by
        keys like "#0", "#1".

    Returns:
        float: The resolved numeric value.

    Raises:
        ValueError: If the value can't be parsed or resolved.
    """
    # If the value is already a number, return it as float
    if isinstance(value, (int, float)):
        return float(value)

    # Clean up leading/trailing spaces
    value = value.strip()

    # Case 1: Memory reference like "#0", "#1"
    if value.startswith("#"):
        return memory.get(value, 0)  # Return stored value or 0 if not present

    # Case 2: Named constant like "const_100" → 100.0 or "const_m1" → -1.0
    elif value == "const_m1":
        return -1.0  # Special case for const_m1
    elif value.startswith("const_"):
        try:
            return float(
                value.replace("const_", ""))  # Strip "const_" and convert
        except ValueError:
            raise ValueError(f"Invalid constant: {value}")

    # Case 3: Raw numeric string (e.g., "75.95", "-100")
    elif value.replace(".", "", 1).replace("-", "", 1).isdigit():
        return float(value)

    # Case 4: Handle percentages (e.g., "4.02%" → 0.0402)
    elif value.endswith("%"):
        try:
            return float(value[:-1]) / 100  # Remove "%" and convert to float
        except ValueError:
            raise ValueError(f"Invalid percentage value: {value}")

    elif value.startswith("table"):
        try:
            parts = value[6:-1].split("][")
            row_idx = int(parts[0])
            column = parts[1].strip("\"")
            return float(table[row_idx].get(column, 0))
        except Exception as e:
            raise ValueError(f"Invalid table reference: {value} → {e}")

    else:
        raise ValueError(f"Unknown expression: {value}")


def execute_program(program, table):
    """
    Executes a sequence of arithmetic operations defined in a single string
    of expressions.

    Each expression can reference previously computed results using memory
    keys like '#0', '#1', etc.
    The results of each step are stored in a memory dictionary and can be
    used in subsequent expressions.

    Parameters:
        program (str): A string containing comma-separated expressions to be
        executed sequentially.
                       Each expression may involve constants (e.g., const_100),
                       table references (e.g., table[0]["Revenue"]),
                       or memory references (e.g., #0).
        table (List[Dict]): A list of dictionaries representing a table
        structure
                            (e.g., parsed from an HTML table or a CSV file).

    Returns:
        float: The result of the final expression in the program sequence.

    Example:
        program = "subtract(75.95, const_100), divide(#0, const_100),
        subtract(102.11, const_100), divide(#2, const_100), subtract(#1, #3)"
        result = execute_program(program, table)
        # This will evaluate the expressions step-by-step using intermediate
        memory.
    """
    memory = {}  # Dictionary to hold intermediate results, keyed as '#0',
    # '#1', etc.

    # Split the input program string into individual expressions by commas
    steps = re.findall(r'[^,()]*\([^)]*\)[^,()]*|[^,()]+', program)

    for i, step in enumerate(steps):
        result = eval_expr(step, table, memory)  # Evaluate each expression
        memory[f"#{i}"] = result  # Store result in memory with key like '#0'

    # Return the final result as float if numeric, else string (e.g., "yes"/"no")
    if isinstance(result, float):
        return round(result, 5)
    else:
        return result


def test_executor(url):
    """
    Runs a test suite over the first few examples in the dataset to validate
    the program executor.
    Now handles examples with multiple QA pairs (qa_0, qa_1, etc.)
    """
    # Load sample data from a JSON file
    data = load_data(url)
    success, total = 0, 0

    # Test on the first 5 examples from the dataset
    for example in data:
        table = parse_table(example["table"])

        # Find all QA pairs in the example
        qa_pairs = []
        i = 0
        while True:
            qa_key = f"qa_{i}"
            if qa_key in example:
                qa_pairs.append(example[qa_key])
                i += 1
            else:
                # Also check for just "qa" (without number) for backward
                # compatibility
                if i == 0 and "qa" in example:
                    qa_pairs.append(example["qa"])
                break

        if not qa_pairs:
            print(
                f"Skipping example {example.get('id', 'unknown')} - no QA "
                f"pairs found")
            continue

        print(
            f"\nProcessing example {example.get('id', 'unknown')} with "
            f"{len(qa_pairs)} QA pairs")
        print("-" * 50)

        for j, qa in enumerate(qa_pairs):
            try:
                program = qa["program"]

                # Handle percentage answers like '14.1%' -> 0.141
                answer_str = qa["exe_ans"]
                if isinstance(answer_str, str) and answer_str.strip().endswith(
                        "%"):
                    expected_answer = float(
                        answer_str.strip().replace("%", "")) / 100
                else:
                    expected_answer = float(answer_str)

                # Execute the program
                predicted_answer = execute_program(program, table)

                # Print detailed result
                print(f"QA Pair {j}:")
                print(f"Question: {qa['question']}")
                print(f"Program: {program}")
                print(f"Expected: {expected_answer}")
                print(f"Predicted: {predicted_answer}")
                # Compare and update counters
                if abs(predicted_answer - expected_answer) < 0.01:
                    print("Match: ✅")
                    success += 1
                else:
                    print("Match: ❌")
                total += 1
                print("-" * 50)

            except KeyError as e:
                print(f"Error processing QA pair {j}: Missing key {e}")
            except Exception as e:
                print(f"Error processing QA pair {j}: {str(e)}")

    # Print final accuracy
    print(
        f"\n✅ Accuracy: {success}/{total} = {success / total:.2%}" if total
                                                                      > 0
        else "\nNo QA pairs processed.")


url = "https://github.com/czyssrs/ConvFinQA/raw/main/data.zip"
# Call the test function
test_executor(url)
