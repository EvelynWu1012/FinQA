import pytest
from src.utils import (
    clean_text,
    construct_chain_of_thought,
    extract_llm_response_components,
    is_numeric,
    format_executable_answer
)


# Test clean_text function
@pytest.mark.parametrize("input_text, expected", [
    # Basic cases
    ("123", "123"),
    ("abc123", "123"),
    ("12.34", "12.34"),
    ("-12.34", "-12.34"),
    # Edge cases with special characters
    ("$1,234.56", "1234.56"),
    ("12a3b4c", "1234"),
    ("1.2.3.4", "1.2.3.4"),  # multiple decimals
    ("--123", "--123"),  # multiple minus signs
    # Yes/No cases
    ("yes", "yes"),
    ("YES", "yes"),
    ("no", "no"),
    ("NO", "no"),
    # Empty/whitespace cases
    ("", ""),
    ("   ", ""),
    (" 123 ", "123")
])
def test_clean_text(input_text, expected):
    assert clean_text(input_text) == expected


# Test construct_chain_of_thought function
def test_construct_chain_of_thought():
    # Complete test case
    complete_data = {
        "question": "What is the total?",
        "focused_table_row": ["Row1", "Row2"],
        "focused_text_row": ["Text1", "Text2"],
        "reasoning_dialogue": ["Step1", "Step2"],
        "turn_program": ["Code1", "Code2"]
    }
    result = construct_chain_of_thought(complete_data)
    assert "1. Understand the Problem" in result
    assert "What is being asked is - 'What is the total?'" in result
    assert "2. Break Down the Problem" in result
    assert "- Identify and analyze table row : 'Row1'" in result
    assert "3. Apply Logical Reasoning" in result
    assert "Step 1: Step1 : Code1" in result

    # Empty test case
    empty_data = {}
    assert construct_chain_of_thought(
        empty_data) == "No reasoning steps available."

    # Partial test case
    partial_data = {
        "question": "Partial question",
        "reasoning_dialogue": ["Step1"],
        # Missing turn_program
    }
    assert construct_chain_of_thought(
        partial_data) == "No reasoning steps available."


# Test extract_llm_response_components function
@pytest.mark.parametrize("input_json, expected", [
    # Complete JSON
    (
            '{"Logical Reasoning": "steps", "Program": "code", "Answer": '
            '"42", "Confidence": "high"}',
            {"reasoning_steps": "steps", "program": "code", "answer": "42",
             "confidence": "high"}
    ),
    # Partial JSON
    (
            '{"Logical Reasoning": "steps", "Answer": "42"}',
            {"reasoning_steps": "steps", "answer": "42"}
    ),
    # Empty JSON
    ('{}', {}),
    # Invalid JSON
    ('not json', {}),
    # Malformed JSON
    ('{"key": "value"', {}),
])
def test_extract_llm_response_components(input_json, expected):
    assert extract_llm_response_components(input_json) == expected


# Test is_numeric function
@pytest.mark.parametrize("input_str, expected", [
    # Valid numbers
    ("123", True),
    ("12.34", True),
    ("-12.34", True),
    ("1.23e-4", True),
    ("0", True),
    # Invalid numbers
    ("abc", False),
    ("12a", False),
    ("1.2.3", False),
    ("--123", False),
    ("", False),
    # Edge cases
    (" 123 ", True),  # whitespace
    ("1,234", False),  # comma
    ("$100", False),  # currency
    ("NaN", False),
    ("inf", False),
])
def test_is_numeric(input_str, expected):
    assert is_numeric(input_str) == expected


# Test format_executable_answer function
@pytest.mark.parametrize("input_answer, expected", [
    # Numeric cases
    (42, 42.0),
    (3.14, 3.14),
    ("42", 42.0),
    ("3.14", 3.14),
    ("-3.14", -3.14),
    # Yes/No cases
    ("yes", "yes"),
    ("YES", "yes"),
    ("no", "no"),
    ("NO", "no"),
    # Cleaned numeric strings
    ("$1,234.56", 1234.56),
    ("42%", 42.0),
])
def test_format_executable_answer_valid(input_answer, expected):
    assert format_executable_answer(input_answer) == expected


@pytest.mark.parametrize("input_answer", [
    "not a number",
    ["list"],
    {"dict": "value"},
    None,
])
def test_format_executable_answer_invalid(input_answer):
    with pytest.raises((ValueError, TypeError)):
        format_executable_answer(input_answer)
