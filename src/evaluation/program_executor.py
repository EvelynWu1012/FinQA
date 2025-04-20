from src.shared import shared_data
from src.data_loader.data_loader import download_data
from src.preprocessing.preprocessor import preprocess_dataset


def parse_table(raw_table):
    """
    Parses a raw table into a dictionary with row headers as keys and numeric values as lists.

    Skips the first row (assumed to be headers) and any invalid or incomplete rows.
    Converts values to floats, ignoring non-numeric entries or text after numbers.

    Parameters:
        raw_table (list[list[str]]): A 2D list representing a table, where each sublist is a row.

    Returns:
        dict: A dictionary where each key is a string from the first column of a row,
              and the corresponding value is a list of parsed floats from the remaining columns.
    """
    result = {}
    for i, row in enumerate(raw_table[1:], start=1):  # skip the first row
        if not row or len(row) < 2:
            print(f"Skipping invalid row {i}: {row}")
            continue
        key = row[0].strip()
        values = []
        for val in row[1:]:
            try:
                # Convert first token to float, remove commas if present
                num = float(val.strip().split()[0].replace(',', ''))
                values.append(num)
            except (ValueError, IndexError):
                pass  # Ignore non-numeric values
        result[key] = values
    return result


def eval_expr(expression, memory, question):
    """
    Evaluates a mathematical or logical expression string using values from memory or table data.

    Supported operations:
        - add(a, b)
        - subtract(a, b)
        - multiply(a, b)
        - divide(a, b)
        - exp(a, b)
        - greater(a, b)
        - table_average(column_name)

    Parameters:
        expression (str): The expression to evaluate (e.g., "add(x, y)").
        memory (dict): A dictionary mapping variable names to values.
        question (str): The current question used to retrieve context-specific data.

    Returns:
        float|str|None: The result of the evaluated expression. May return:
                        - a float for numeric results,
                        - "yes"/"no" for comparisons,
                        - None for unsupported expressions,
                        - NaN for invalid computations (e.g., division by zero).
    """
    # Strip leading/trailing whitespace
    expression = expression.strip()

    def get_operands(expr, prefix_len):
        """Extracts operands from expression after stripping function name and parentheses."""
        return expr[prefix_len:-1].split(",")

    if expression.startswith("add("):
        a, b = get_operands(expression, 4)
        val1 = resolve_value(a, memory, question)
        val2 = resolve_value(b, memory, question)
        return _log_result(expression, val1, val2, val1 + val2, "+")

    elif expression.startswith("subtract("):
        a, b = get_operands(expression, 9)
        val1 = resolve_value(a, memory, question)
        val2 = resolve_value(b, memory, question)
        return _log_result(expression, val1, val2, val1 - val2, "-")

    elif expression.startswith("multiply("):
        a, b = get_operands(expression, 9)
        val1 = resolve_value(a, memory, question)
        val2 = resolve_value(b, memory, question)
        return _log_result(expression, val1, val2, val1 * val2, "*")

    elif expression.startswith("divide("):
        a, b = get_operands(expression, 7)
        val1 = resolve_value(a, memory, question)
        val2 = resolve_value(b, memory, question)
        if val2 == 0:
            # Avoid division by zero errors
            print(f"Warning: Division by zero in expression: {expression}")
            return float('nan')
        return _log_result(expression, val1, val2, val1 / val2, "/")

    elif expression.startswith("exp("):
        a, b = get_operands(expression, 4)
        base = resolve_value(a, memory, question)
        exp = resolve_value(b, memory, question)
        return _log_result(expression, base, exp, base ** exp, "**")

    elif expression.startswith("greater("):
        a, b = get_operands(expression, 8)
        val1 = resolve_value(a, memory, question)
        val2 = resolve_value(b, memory, question)
        result = "yes" if val1 > val2 else "no"
        return result

    elif expression.startswith("table_average("):
        # Retrieve the processed dataset for the specific question
        processed_dataset = shared_data.processed_dataset
        extract_info = processed_dataset.get(question, {})

        # Parse the table if data is found, otherwise default to empty
        if extract_info:
            table = parse_table(extract_info['table'])
        else:
            print("Data for this question not found.")
            table = {}

        # Get the column name to average and resolve values from the parsed table
        col = get_operands(expression, 14)[0]
        values = resolve_value(col, memory, question, table_override=table)
        if not values:
            return float('nan')  # Avoid division by zero
        return round(sum(values) / len(values), 3)

    else:
        # Handle unsupported or malformed expressions
        print(f"Unsupported operator or expression: {expression}")
        return None



def _log_result(expr, v1, v2, result, op):
    """
    Logs the evaluation of a binary expression and returns the result.

    Parameters:
        expr (str): The original expression string (e.g., "add(a, b)").
        v1 (float): The first resolved operand.
        v2 (float): The second resolved operand.
        result (float): The computed result of applying the operation.
        op (str): The operator used in the expression (e.g., "+", "-", "*", "/", "**").

    Returns:
        float: The result of the computation.
    """

    return result  # Return the final computed result


def resolve_value(value, memory, question, table_override=None):
    """
    Resolves a given value into a numeric float or list of floats depending on its format.

    This function supports resolution of:
        - Numeric constants
        - Memory references (e.g., "#1")
        - Predefined constants (e.g., "const_5")
        - Percentage strings (e.g., "15%")
        - Table lookups by column name

    Parameters:
        value (Any): The value to resolve. Can be a float, int, string reference, etc.
        memory (dict): Dictionary storing intermediate results referenced by keys like '#1', '#2', etc.
        question (str): The associated question ID used for fetching the table if needed.
        table_override (dict, optional): If provided, use this table instead of fetching by question.

    Returns:
        float or list[float] or None: The resolved value.

    Raises:
        ValueError: If the value cannot be resolved.
    """

    # If value is already a number, return it as float
    if isinstance(value, (int, float)):
        return float(value)

    value = value.strip()  # Remove whitespace for consistent parsing

    # Memory reference (e.g., "#1" -> memory["#1"])
    if value.startswith("#"):
        idx = int(value[1:])
        if f"#{idx}" in memory:
            return memory[f"#{idx}"]
        else:
            raise ValueError(f"Memory reference #{idx} not found.")

    # Special constant -1
    if value == "const_m1":
        return -1.0

    # General constant (e.g., "const_10" -> 10.0)
    elif value.startswith("const_"):
        try:
            return float(value.replace("const_", ""))
        except ValueError:
            raise ValueError(f"Invalid constant format: {value}")

    # Numeric string (e.g., "42", "-3.5")
    if value.replace(".", "", 1).replace("-", "", 1).isdigit():
        return float(value)

    # Percentage string (e.g., "25%" -> 0.25)
    if value.endswith("%"):
        try:
            return float(value[:-1]) / 100
        except ValueError:
            raise ValueError(f"Invalid percentage format: {value}")

    # Otherwise, treat as a column name to look up in the table
    if isinstance(value, str):
        # Use override table if provided, otherwise retrieve by question
        table = table_override or _get_table_for_question(question)

        val = table.get(value, None)

        # Handle list of values (e.g., multiple entries in a table column)
        if isinstance(val, list):
            if len(val) == 1:
                return val[0]  # Return single float
            elif len(val) > 1:
                return val  # Return list of floats
            else:
                return float('nan')  # Empty list → NaN
        return val  # Could be None if column not found

    # If none of the above match, raise an error
    raise ValueError(
        f"Unknown expression or missing table column: {value}"
    )


def _get_table_for_question(question):
    """
    Retrieves and parses the table associated with a specific question from the dataset.

    This function looks up the question in the shared dataset and attempts to extract
    the corresponding table. If found, the table is parsed using the `parse_table` function.
    If no table is found for the question, an empty dictionary is returned.

    Parameters:
        question (str): The question ID to look up in the dataset.

    Returns:
        dict: A dictionary representing the parsed table if found, otherwise an empty dictionary.
    """

    # Access the shared dataset containing all processed data
    dataset = shared_data.processed_dataset

    # Get the data associated with the given question; defaults to an empty dictionary if not found
    extract_info = dataset.get(question, {})

    # If the data contains a valid table entry, parse and return it
    if extract_info:
        return parse_table(extract_info['table'])

    # Return an empty dictionary if no valid table is found for the question
    return {}


def split_program_steps(prog):
    """
    Splits a program (expression) into individual steps based on top-level commas.

    The function handles nested parentheses to ensure commas inside parentheses are not
    treated as delimiters. It returns a list of steps where each step is a string
    representing a part of the program.

    Parameters:
        prog (str): The program (expression) string to be split into steps.

    Returns:
        list: A list of program steps (strings).
    """

    # Initialize variables to store the steps and manage nested parentheses depth
    steps = []  # List to store the final steps
    depth = 0  # Variable to track the depth of nested parentheses
    current = []  # List to accumulate characters for the current step

    # Iterate through each character in the program string
    for char in prog:

        # Check for the top-level comma (ignoring nested commas within parentheses)
        if char == ',' and depth == 0:
            # If at top-level, we consider the current step complete, add it to the steps list
            steps.append(''.join(
                current).strip())  # Join current list to form the step and strip whitespace
            current = []  # Reset current to start accumulating the next step

        else:
            # Adjust depth for opening and closing parentheses
            if char == '(':
                depth += 1  # Increase depth on encountering '('
            elif char == ')':
                depth -= 1  # Decrease depth on encountering ')'

            # Append the current character to the current step list
            current.append(char)

    # After the loop, if there is any remaining content in 'current', add it as the last step
    if current:
        steps.append(''.join(
            current).strip())  # Join and strip any remaining characters

    return steps  # Return the list of program steps


def execute_program(program, question):
    """
    Executes a program by evaluating each step in sequence and storing intermediate results in memory.

    The program is split into individual steps, and each step is evaluated using the `eval_expr` function.
    The results are stored in a memory dictionary, where the key is the step index prefixed with '#'.

    Parameters:
        program (str): The program to be executed, consisting of a sequence of steps.
        question (str): The question associated with the program (used for resolving values in steps).

    Returns:
        result: The final result of executing the program. If the result is a float, it is rounded to 3 decimal places.
    """
    memory = {}  # Dictionary to store intermediate results with step indices as keys

    # Split the program into individual steps using the 'split_program_steps' function
    steps = split_program_steps(program)
    results = None # Initialize results

    # Iterate over each step in the program
    for i, step in enumerate(steps):
        # Evaluate the step using 'eval_expr' and store the result in memory
        result = eval_expr(step, memory, question)

        # Store the result in memory, using the step index (prefixed with '#') as the key
        memory[f"#{i}"] = result

    # Return the final result. If it's a float, round it to 3 decimal places, else return as-is
    return round(result, 3) if isinstance(result, float) else result

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
    program = "extract('37529')"
    predicted_answer = execute_program(program, question)

    print("debug predicted_answer", predicted_answer)
"""