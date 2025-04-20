import pytest
import math
from src.evaluation.program_executor import (
    parse_table,
    eval_expr,
    resolve_value,
    split_program_steps,
    execute_program,
)


# Mock the shared_data module to avoid external dependencies
class MockSharedData:
    processed_dataset = {}


@pytest.fixture
def mock_shared_data(monkeypatch):
    mock_module = MockSharedData()
    monkeypatch.setattr('src.evaluation.program_executor.shared_data',
                        mock_module)


# Test data for tables
SAMPLE_TABLE = [
    ["Country", "Population", "GDP"],
    ["USA", "331,002,651", "20.94 trillion"],
    ["China", "1,439,323,776", "14.72 trillion"],
    ["India", "1,380,004,385", "2.87 trillion"],
]

# Updated invalid table - now properly structured
INVALID_TABLE = [
    ["Country", "Population"],
    ["USA"],  # Incomplete row
    ["China", "invalid_number"],  # Invalid number format (not a list)
]


def test_parse_table_valid():
    result = parse_table(SAMPLE_TABLE)
    assert "USA" in result
    assert result["USA"] == [331002651.0, 20.94]
    assert result["China"] == [1439323776.0, 14.72]
    assert result["India"] == [1380004385.0, 2.87]


def test_parse_table_invalid_rows():
    result = parse_table(INVALID_TABLE)
    assert "USA" not in result  # Skipped because row is incomplete
    assert "China" not in result  # "invalid_number" can't be parsed


def test_resolve_value_table_lookup(mock_shared_data):
    # Setup mock table data with proper string values (not lists)
    MockSharedData.processed_dataset = {
        "q1": {
            "table": [
                ["Country", "Population"],
                ["USA", "331,002,651"],  # String value
                ["China", "1,439,323,776"],
            ]
        }
    }
    # parse_table will convert these to lists of floats
    assert resolve_value("USA", {}, "q1") == 331002651.0
    assert resolve_value("China", {}, "q1") == 1439323776.0


def test_execute_program_with_table(mock_shared_data):
    # Setup mock table data with proper string values
    MockSharedData.processed_dataset = {
        "q2": {
            "table": [
                ["Country", "Population"],
                ["USA", "331,002,651"],  # String value
                ["China", "1,439,323,776"],
            ]
        }
    }
    program = "table_average(USA), table_average(China), add(#0, #1)"
    result = execute_program(program, "q2")
    # Since each country has one value, average will be that value
    assert math.isclose(result, 331002651.0 + 1439323776.0)


def test_execute_program_complex(mock_shared_data):
    program = """
        add(5, 3),
        subtract(10, 4),
        multiply(#0, #1),
        divide(#2, 2),
        greater(#3, 10)
    """
    result = execute_program(program, "q1")
    assert result == "yes"


def test_execute_program_with_constants():
    program = "add(const_5, const_m1), multiply(#0, 2)"
    result = execute_program(program, "q1")
    assert result == 8.0
