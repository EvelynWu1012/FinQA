import pytest
from unittest.mock import patch, MagicMock
import json
from src.prompt_LLM.prompt_answer_gen_inference import (
    query_data,
    generate_few_shot_prompt,
    query_gpt,
    generate_answer,
    generate_ground_truth
)
from src.shared import shared_data

# Sample test data
SAMPLE_PROCESSED_DATA = {
    "What was revenue in 2020?": {
        "question": "What was revenue in 2020?",
        "table": [["Year", "Revenue"], ["2020", "100M"]],
        "pre_text": "Financial report for 2020",
        "post_text": "All figures in millions",
        "program": "lookup(2020, Revenue)",
        "answer": "100M",
        "steps": ["Find 2020 row", "Extract Revenue value"]
    },
    "Calculate growth rate": {
        "question": "Calculate growth rate",
        "table": [["Year", "Revenue"], ["2019", "90M"], ["2020", "100M"]],
        "pre_text": "Growth analysis",
        "post_text": "Annual figures",
        "program": "divide(subtract(100M, 90M), 90M)",
        "answer": "11.11%",
        "steps": ["Calculate difference", "Divide by base year"]
    }
}

SAMPLE_CONTEXT = {
    "pre_text": "Quarterly earnings",
    "table": [["Qtr", "Sales"], ["Q1", "50M"], ["Q2", "60M"]],
    "post_text": "All figures unaudited"
}


@pytest.fixture(autouse=True)
def setup_teardown():
    """Setup and teardown for each test"""
    # Setup
    shared_data.processed_dataset = SAMPLE_PROCESSED_DATA.copy()

    yield  # Test runs here

    # Teardown
    shared_data.processed_dataset = None


# =============================================================================
# Test query_data()
# =============================================================================
class TestQueryData:
    def test_query_existing_question(self):
        """Test retrieving data for an existing question"""
        result = query_data("What was revenue in 2020?", SAMPLE_PROCESSED_DATA)
        assert result["answer"] == "100M"
        assert result["program"] == "lookup(2020, Revenue)"

    def test_query_nonexistent_question(self):
        """Test handling of non-existent questions"""
        result = query_data("Unknown question", SAMPLE_PROCESSED_DATA)
        assert "error" in result
        assert result["error"] == "Data for this question not found."


# =============================================================================
# Test generate_few_shot_prompt()
# =============================================================================
class TestGenerateFewShotPrompt:
    @patch(
        'src.prompt_LLM.prompt_answer_gen_inference.prompt_example_generator')
    def test_generate_prompt_with_examples(self, mock_generator):
        """Test prompt generation with few-shot examples"""
        mock_generator.return_value = ["What was revenue in 2020?",
                                       "Calculate growth rate"]
        user_question = "What is Q2 sales?"

        prompt = generate_few_shot_prompt(
            processed_data=SAMPLE_PROCESSED_DATA,
            user_question=user_question,
            context=SAMPLE_CONTEXT,
            num_example=2
        )

        assert "Example: 1" in prompt
        assert "Question: What was revenue in 2020?" in prompt
        assert "Financial report for 2020" in prompt  # pre_text
        assert "lookup(2020, Revenue)" in prompt  # program
        assert "Example: 2" in prompt
        assert "Questions\n\nYou are a helpful financial analysis assistant" in prompt
        assert user_question in prompt

    def test_generate_prompt_with_error_context(self):
        """Test prompt generation when context has an error"""
        error_context = {"error": "Invalid question format"}
        prompt = generate_few_shot_prompt(
            processed_data=SAMPLE_PROCESSED_DATA,
            user_question="Any question",
            context=error_context,
            num_example=2
        )
        assert prompt == "Invalid question format"


# =============================================================================
# Test query_gpt()
# =============================================================================
class TestQueryGPT:
    @patch('src.prompt_LLM.prompt_answer_gen_inference.openai.OpenAI')
    def test_successful_gpt_query(self, mock_openai):
        """Test successful GPT query with mock response"""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "{\"Answer\": \"50M\"}"
        mock_openai.return_value.chat.completions.create.return_value = mock_response

        response = query_gpt("Test prompt")
        assert "50M" in response

    @patch('src.prompt_LLM.prompt_answer_gen_inference.openai.OpenAI')
    def test_gpt_query_with_empty_response(self, mock_openai):
        """Test handling of empty GPT response"""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = ""
        mock_openai.return_value.chat.completions.create.return_value = mock_response

        with pytest.raises(ValueError, match="Empty response from GPT"):
            query_gpt("Test prompt")


# =============================================================================
# Test generate_answer()
# =============================================================================
class TestGenerateAnswer:
    @patch('src.prompt_LLM.prompt_answer_gen_inference.query_gpt')
    @patch(
        'src.prompt_LLM.prompt_answer_gen_inference.generate_few_shot_prompt')
    def test_successful_answer_generation(self, mock_prompt, mock_gpt):
        """Test successful answer generation"""
        # Setup mocks
        mock_prompt.return_value = "Test prompt"
        mock_gpt.return_value = json.dumps({
            "Logical Reasoning": "Test reasoning",
            "Program": "test()",
            "Answer": "50M",
            "Confidence": "90%"
        })

        response = generate_answer("What is Q2 sales?", num_example=2)
        assert "50M" in response
        mock_prompt.assert_called_once()
        mock_gpt.assert_called_once()

    def test_generate_answer_without_processed_data(self):
        """Test error when processed data is missing"""
        shared_data.processed_dataset = None
        with pytest.raises(ValueError, match="Data must be loaded"):
            generate_answer("Any question", num_example=2)


# =============================================================================
# Test generate_ground_truth()
# =============================================================================
class TestGenerateGroundTruth:
    def test_ground_truth_for_existing_question(self):
        """Test ground truth generation for existing question"""
        result = generate_ground_truth("What was revenue in 2020?")
        assert result["Program"] == "lookup(2020, Revenue)"
        assert result["Answer"] == "100M"

    def test_ground_truth_for_missing_question(self):
        """Test ground truth generation for missing question"""
        result = generate_ground_truth("Unknown question")
        assert result["Program"] == "Program not found"
        assert result["Answer"] == "Answer not found"

    def test_ground_truth_with_empty_values(self):
        """Test ground truth with empty program/answer"""
        # Add test case with empty values
        shared_data.processed_dataset["Empty case"] = {
            "question": "Empty case",
            "program": "",
            "answer": None
        }

        result = generate_ground_truth("Empty case")
        assert result["Program"] == "Program not found"
        assert result["Answer"] == "Answer not found"


# =============================================================================
# Integration Tests
# =============================================================================
class TestIntegration:
    @patch('src.prompt_LLM.prompt_answer_gen_inference.query_gpt')
    def test_full_answer_generation_flow(self, mock_gpt):
        """Test the full answer generation flow"""
        # Setup
        mock_gpt.return_value = json.dumps({
            "Answer": "60M",
            "Program": "lookup(Q2, Sales)",
            "Confidence": "95%"
        })

        # Execute
        response = generate_answer("What is Q2 sales?", num_example=2)
        data = json.loads(response)

        # Verify
        assert data["Answer"] == "60M"
        assert "Q2" in data["Program"]
        assert "%" in data["Confidence"]

    def test_ground_truth_integration(self):
        """Test ground truth matches processed data"""
        question = "Calculate growth rate"
        result = generate_ground_truth(question)

        assert result["Program"] == SAMPLE_PROCESSED_DATA[question]["program"]
        assert result["Answer"] == SAMPLE_PROCESSED_DATA[question]["answer"]