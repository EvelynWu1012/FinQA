import pytest
from unittest.mock import patch
from src.preprocessing.preprocessor import preprocess_example, preprocess_dataset
from src.shared import shared_data

# Sample test data
SAMPLE_EXAMPLE = {
    "pre_text": "Pre text context",
    "post_text": "Post text context",
    "table": [["Header1", "Header2"], ["Row1", "Value1"], ["Row2", "Value2"]],
    "annotation": {
        "dialogue_break": ["Dialogue part 1", "Dialogue part 2"],
        "turn_program": ["Program step 1", "Program step 2"]
    },
    "qa1": {
        "question": "What is the value for Row1?",
        "ann_table_rows": [0],
        "ann_text_rows": [],
        "steps": ["Step 1", "Step 2"],
        "program": "PROGRAM CODE",
        "exe_ans": "Value1",
        "answer": "The value is Value1"
    },
    "qa2": {
        "question": "What is the value for Row2?",
        "ann_table_rows": [1],
        "ann_text_rows": [],
        "steps": ["Step 1"],
        "program": "PROGRAM CODE 2",
        "exe_ans": "Value2",
        "answer": "The value is Value2"
    }
}

SAMPLE_DATASET = [SAMPLE_EXAMPLE, {
    "pre_text": "Another pre text",
    "post_text": "Another post text",
    "table": [["Header"]],
    "annotation": {},
    "qa1": {
        "question": "Simple question",
        "answer": "Simple answer"
    }
}]

EMPTY_EXAMPLE = {
    "pre_text": "",
    "post_text": "",
    "table": [],
    "annotation": {},
    "qa1": {
        "question": "",
        "answer": ""
    }
}


@pytest.fixture
def reset_shared_data():
    """Fixture to reset shared_data before each test"""
    shared_data.processed_dataset = None
    shared_data.questions = None


class TestPreprocessExample:
    """Test suite for preprocess_example function"""

    def test_preprocess_example_basic(self, reset_shared_data):
        """Test basic preprocessing of an example"""
        result = preprocess_example(SAMPLE_EXAMPLE)

        # Verify the output structure
        assert isinstance(result, dict)
        assert len(result) == 2  # Should have two questions

        # Check first question
        q1 = "What is the value for Row1?"
        assert q1 in result
        q1_data = result[q1]
        assert q1_data["question"] == q1
        assert q1_data["table"] == SAMPLE_EXAMPLE["table"]
        assert q1_data["focused_table_row"] == [0]
        assert q1_data["steps"] == ["Step 1", "Step 2"]
        assert q1_data["program"] == "PROGRAM CODE"
        assert q1_data["exe_ans"] == "Value1"
        assert q1_data["answer"] == "The value is Value1"
        assert q1_data["pre_text"] == "Pre text context"
        assert q1_data["post_text"] == "Post text context"
        assert q1_data["reasoning_dialogue"] == ["Dialogue part 1",
                                                 "Dialogue part 2"]
        assert q1_data["turn_program"] == ["Program step 1", "Program step 2"]

        # Check second question
        q2 = "What is the value for Row2?"
        assert q2 in result
        q2_data = result[q2]
        assert q2_data["question"] == q2
        assert q2_data["focused_table_row"] == [1]

    def test_preprocess_example_missing_fields(self, reset_shared_data):
        """Test preprocessing with missing optional fields"""
        example = {
            "table": [],
            "qa1": {
                "question": "Test question",
                "answer": "Test answer"
            }
        }
        result = preprocess_example(example)

        assert len(result) == 1
        q_data = result["Test question"]
        assert q_data["question"] == "Test question"
        assert q_data["table"] == []
        assert q_data["focused_table_row"] == []
        assert q_data["focused_text_row"] == []
        assert q_data["steps"] == []
        assert q_data["program"] == ""
        assert q_data["exe_ans"] is None
        assert q_data["answer"] == "Test answer"
        assert q_data["pre_text"] == ""
        assert q_data["post_text"] == ""
        assert q_data["reasoning_dialogue"] == []
        assert q_data["turn_program"] == []

    def test_preprocess_empty_example(self, reset_shared_data):
        """Test preprocessing with empty example"""
        result = preprocess_example(EMPTY_EXAMPLE)

        assert len(result) == 1
        q_data = result[""]
        assert q_data["question"] == ""
        assert q_data["answer"] == ""
        assert q_data["table"] == []

    def test_preprocess_example_multiple_qa(self, reset_shared_data):
        """Test preprocessing with multiple QA pairs"""
        example = {
            "table": [],
            "qa1": {"question": "Q1", "answer": "A1"},
            "qa2": {"question": "Q2", "answer": "A2"},
            "qa3": {"question": "Q3", "answer": "A3"}
        }
        result = preprocess_example(example)

        assert len(result) == 3
        assert "Q1" in result
        assert "Q2" in result
        assert "Q3" in result

    def test_preprocess_example_no_qa(self, reset_shared_data):
        """Test preprocessing with no QA pairs"""
        example = {"table": []}
        result = preprocess_example(example)

        assert isinstance(result, dict)
        assert len(result) == 0

    def test_preprocess_example_malformed_qa(self):
        """Test handling of malformed QA pairs"""
        example = {
            "table": [],
            "qa1": {"answer": "Answer without question"}  # Missing question
        }
        result = preprocess_example(example)
        assert len(result) == 0  # Should probably skip malformed QAs

    def test_preprocess_example_invalid_table_indices(self):
        """Test handling of invalid table indices"""
        example = {
            "table": [["Header"]],
            "qa1": {
                "question": "Q1",
                "answer": "A1",
                "ann_table_rows": [5]  # Index out of bounds
            }
        }
        result = preprocess_example(example)
        assert result["Q1"]["focused_table_row"] == [5]

class TestPreprocessDataset:
    """Test suite for preprocess_dataset function"""

    def test_preprocess_dataset_basic(self, reset_shared_data):
        """Test basic dataset preprocessing"""
        result = preprocess_dataset(SAMPLE_DATASET, 2)

        assert isinstance(result, dict)
        assert len(
            result) == 3  # Two questions from first example, one from second
        assert shared_data.processed_dataset == result
        assert len(shared_data.questions) == 3

        # Check a question from first example
        assert "What is the value for Row1?" in result
        # Check question from second example
        assert "Simple question" in result


    def test_preprocess_dataset_empty(self, reset_shared_data):
        """Test preprocessing empty dataset"""
        result = preprocess_dataset([], 10)

        assert isinstance(result, dict)
        assert len(result) == 0
        assert shared_data.processed_dataset == {}
        assert shared_data.questions == []

    def test_preprocess_dataset_updates_shared_data(self, reset_shared_data):
        """Test that preprocessing updates shared_data correctly"""
        assert shared_data.processed_dataset is None
        assert shared_data.questions is None

        result = preprocess_dataset(SAMPLE_DATASET[:1], 1)

        assert shared_data.processed_dataset is not None
        assert shared_data.questions is not None
        assert shared_data.processed_dataset == result
        assert shared_data.questions == list(result.keys())

    def test_preprocess_dataset_with_zero_max_samples(self, reset_shared_data):
        """Test preprocessing with max_samples=0"""
        result = preprocess_dataset(SAMPLE_DATASET, 0)

        assert isinstance(result, dict)
        assert len(result) == 0
        assert shared_data.processed_dataset == {}
        assert shared_data.questions == []



# Integration tests
class TestIntegration:
    """Integration tests for the preprocessor module"""

    def test_preprocess_example_output_structure(self, reset_shared_data):
        """Test that preprocess_example output has consistent structure"""
        result = preprocess_example(SAMPLE_EXAMPLE)

        for question_data in result.values():
            assert set(question_data.keys()) == {
                "question", "table", "focused_table_row", "focused_text_row",
                "steps", "program", "exe_ans", "answer", "pre_text",
                "post_text",
                "reasoning_dialogue", "turn_program"
            }

    def test_preprocess_dataset_uses_preprocess_example(self,
                                                        reset_shared_data):
        """Test that preprocess_dataset uses preprocess_example correctly"""
        with patch('src.preprocessing.preprocessor.preprocess_example') as mock_preprocess:
            mock_preprocess.return_value = {"mock_question": {}}
            result = preprocess_dataset(SAMPLE_DATASET[:1], 1)

            mock_preprocess.assert_called_once_with(SAMPLE_DATASET[0])
            assert result == {"mock_question": {}}
            assert shared_data.processed_dataset == {"mock_question": {}}
            assert shared_data.questions == ["mock_question"]

    def test_shared_data_persistence(self, reset_shared_data):
        """Test that shared_data persists between operations"""
        # First processing
        result1 = preprocess_dataset(SAMPLE_DATASET[:1], 1)
        assert shared_data.processed_dataset == result1

        # Second processing should overwrite
        result2 = preprocess_dataset(SAMPLE_DATASET[1:], 1)
        assert shared_data.processed_dataset == result2
        assert shared_data.processed_dataset != result1
