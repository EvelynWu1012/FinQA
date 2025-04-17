
import re
from data_loader import download_data
from preprocessor import preprocess_dataset
import shared_data

"""
Step 2: Table Parser Design. 
"""

def parse_table(raw_table):
    """
    Converts a raw table (nested list) into a dictionary where each key is
    the first element of the row, and the value is a list of numbers (
    converted from strings). Skips the first row of the table.

    Example:
    [['beginning of year', '405'],
     ['revisions of previous estimates', '15']]
    -->
    {
        'revisions of previous estimates': [15],
        ...
    }
    """
    result = {}

    for row in raw_table[1:]:  # skip the first row
        if not row:
            continue
        key = row[0]
        values = []
        for val in row[1:]:
            # Clean up and convert to number
            try:
                num = float(val.strip().split()[0].replace(',', ''))
                values.append(num)
            except ValueError:
                pass  # skip non-numeric values
        result[key] = values

    return result



"""
Step 3: Build the Program Executor
"""


def eval_expr(expression, table, memory):
    """
    Evaluates a single arithmetic expression involving basic operations:
    add, subtract, multiply, divide.

    The expression can involve:
        - Constants (e.g., "const_100")
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

    # elif expression.startswith("table_average("):

    else:
        # If it's not a recognized operation, attempt to directly resolve it
        # (e.g., "#0", "const_100", or "table[0][\"Revenue\"]")
        result = resolve_value(expression, table, memory)
        print(f"Intermediate: {expression} → Resolved to {result}")
        return result


def resolve_value(value, memory):
    """
    Resolves the input `value` to a numerical float.

    The function supports:
    - Memory references (e.g., "#0", "#1") from previous steps in `memory`
    - Named constants (e.g., "const_100" → 100.0, "const_m1" → -1.0)
    - Raw numeric strings (e.g., "75.95", "-10")
    - Percentages (e.g., "4.02%" → 0.0402)
    Parameters:
        value (str | int | float): The value or reference to resolve.
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

    ## elif to handle the table_avarage to extract column name

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

    # Return the final result as float if numeric, else string (e.g.,
    # "yes"/"no")
    if isinstance(result, float):
        return round(result, 5)
    else:
        return result

"""
def test_executor(url):
    """
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
"""