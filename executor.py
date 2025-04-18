
import shared_data


# Shared data storage
class SharedData:
    processed_dataset = {}


shared_data = SharedData()


def parse_table(raw_table):
    result = {}
    for i, row in enumerate(raw_table[1:], start = 1):  # skip the first row
        if not row or len(row) < 2:
            print(f"Skipping invalid row {i}: {row}")
            continue
        key = row[0]
        values = []
        for val in row[1:]:
            try:
                num = float(val.strip().split()[0].replace(',', ''))
                values.append(num)
            except ValueError:
                pass
        result[key] = values
    print(f"Parsed table: {result}")
    return result


def eval_expr(expression, memory, question):
    expression = expression.replace(" ", "")

    def get_operands(expr, prefix_len):
        return expr[prefix_len:-1].split(",")

    if expression.startswith("add("):
        a, b = get_operands(expression, 4)
        val1, val2 = resolve_value(a, memory, question), resolve_value(b,
                                                                       memory,
                                                                       question)
        return _log_result(expression, val1, val2, val1 + val2, "+")

    elif expression.startswith("subtract("):
        a, b = get_operands(expression, 9)
        val1, val2 = resolve_value(a, memory, question), resolve_value(b,
                                                                       memory,
                                                                       question)
        return _log_result(expression, val1, val2, val1 - val2, "-")

    elif expression.startswith("multiply("):
        a, b = get_operands(expression, 9)
        val1, val2 = resolve_value(a, memory, question), resolve_value(b,
                                                                       memory,
                                                                       question)
        return _log_result(expression, val1, val2, val1 * val2, "*")

    elif expression.startswith("divide("):
        a, b = get_operands(expression, 7)
        val1, val2 = resolve_value(a, memory, question), resolve_value(b,
                                                                       memory,
                                                                       question)
        if val2 == 0:
            print(f"Warning: Division by zero in expression: {expression}")
            return float('nan')
        return _log_result(expression, val1, val2, val1 / val2, "/")

    elif expression.startswith("exp("):
        a, b = get_operands(expression, 4)
        base, exp = resolve_value(a, memory, question), resolve_value(b,
                                                                      memory,
                                                                      question)
        return _log_result(expression, base, exp, base ** exp, "**")

    elif expression.startswith("greater("):
        a, b = get_operands(expression, 8)
        val1, val2 = resolve_value(a, memory, question), resolve_value(b,
                                                                       memory,
                                                                       question)
        result = "yes" if val1 > val2 else "no"
        # print(f"Intermediate: {expression} → {val1} > {val2} = {result}")
        return result

    elif expression.startswith("table_average("):
        processed_dataset = shared_data.processed_dataset
        extract_info = processed_dataset.get(question, {})
        if extract_info:
            table = parse_table(extract_info['table'])
        else:
            print("Data for this question not found.")
            table = {}
        col = expression[14:-1]
        values = resolve_value(col, memory, question, table_override=table)
        if not values:
            return float('nan')
        return round(sum(values) / len(values), 3)

    else:
        val = resolve_value(expression, memory, question)
        if isinstance(val, (int, float)):
            return round(val, 3)
        return val


def _log_result(expr, v1, v2, result, op):
    # print(f"Intermediate: {expr} → {v1} {op} {v2} = {result}")
    return result


def resolve_value(value, memory, question, table_override=None):
    if isinstance(value, (int, float)):
        return float(value)

    value = value.strip()

    if value.startswith("#"):
        # Fetch from memory
        idx = int(value[1:])
        if f"#{idx}" in memory:
            return memory[f"#{idx}"]
        else:
            raise ValueError(f"Memory reference #{idx} not found.")

    if value == "const_m1":
        return -1.0
    elif value.startswith("const_"):
        try:
            return float(value.replace("const_", ""))
        except ValueError:
            raise ValueError(f"Invalid constant format: {value}")

    if value.replace(".", "", 1).replace("-", "", 1).isdigit():
        return float(value)

    if value.endswith("%"):
        try:
            return float(value[:-1]) / 100
        except ValueError:
            raise ValueError(f"Invalid percentage format: {value}")

    if value.startswith("table_average("):
        table = table_override or _get_table_for_question(question)
        if table and value in table:
            val = table[value]
        if isinstance(val, list):
            if len(val) == 1:
                return val[0]
            return val
        return val
        print(f"Warning: Unknown table column or value: {value}")
        raise ValueError(f"Unknown expression or missing table column: {value}")


def _get_table_for_question(question):
    dataset = shared_data.processed_dataset
    extract_info = dataset.get(question, {})
    if extract_info:
        return parse_table(extract_info['table'])
    return {}


def split_program_steps(prog):
    steps = []
    depth = 0
    current = []
    for char in prog:
        if char == ',' and depth == 0:
            steps.append(''.join(current).strip())
            current = []
        else:
            if char == '(':
                depth += 1
            elif char == ')':
                depth -= 1
            current.append(char)
    if current:
        steps.append(''.join(current).strip())
    return steps


def execute_program(program, question):
    memory = {}
    # steps = re.findall(r'[^,()]*\([^)]*\)[^,()]*|[^,()]+', program)
    steps = split_program_steps(program)
    for i, step in enumerate(steps):
        result = eval_expr(step, memory, question)
        memory[f"#{i}"] = result

    return round(result, 3) if isinstance(result, float) else result
