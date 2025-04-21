import pytest
import numpy as np
import faiss
from unittest.mock import patch

# Import the module to test
from src.prompt_LLM.prompt_shots_selector import (
    prepare_questions,
    generate_embeddings,
    build_faiss_index,
    get_top_similar_questions_faiss,
    initialize_faiss_index,
    prompt_example_generator,
    MODEL
)

# Mock data for testing
MOCK_QUESTIONS = [
    "What is the capital of France?",
    "How to cook pasta?",
    "What is the meaning of life?",
    "How does photosynthesis work?",
    "What are the best programming practices?"
]

MOCK_EMBEDDINGS = np.random.rand(len(MOCK_QUESTIONS), 384).astype(np.float32)


@pytest.fixture
def mock_shared_data():
    class MockSharedData:
        def __init__(self):
            self.questions = MOCK_QUESTIONS
            self.faiss_index = None
            self.question_embeddings = None

    return MockSharedData()


def test_prepare_questions(mock_shared_data):
    """Test that prepare_questions correctly retrieves questions from shared_data"""
    with patch('src.prompt_LLM.prompt_shots_selector.shared_data', mock_shared_data):
        result = prepare_questions()
        assert result == MOCK_QUESTIONS
        assert len(result) == 5


def test_generate_embeddings():
    """Test that generate_embeddings produces embeddings of correct shape and type"""
    test_questions = ["test question 1", "test question 2"]

    # Mock the MODEL.encode method to return predictable output
    with patch.object(MODEL, 'encode', return_value=np.random.rand(2, 384)):
        embeddings = generate_embeddings(test_questions)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (2, 384)
        assert embeddings.dtype == np.float32


def test_build_faiss_index():
    """Test that the FAISS index is built correctly"""
    # Create test embeddings
    test_embeddings = np.random.rand(10, 384).astype(np.float32)

    # Build index
    index = build_faiss_index(test_embeddings)

    # Verify index properties
    assert isinstance(index, faiss.IndexFlatIP)
    assert index.ntotal == 10  # Number of vectors in index
    assert index.d == 384  # Dimension of vectors


def test_get_top_similar_questions_faiss(mock_shared_data):
    """Test retrieval of similar questions using FAISS"""
    # Setup test data
    test_index = build_faiss_index(MOCK_EMBEDDINGS)
    test_question = "What is the capital of Germany?"
    top_num = 3

    # Mock the MODEL.encode method
    with patch.object(MODEL, 'encode',
                      return_value=np.random.rand(1, 384).astype(np.float32)):
        with patch('src.prompt_LLM.prompt_shots_selector.shared_data', mock_shared_data):
            result = get_top_similar_questions_faiss(test_question, test_index, top_num)
            assert len(result) == top_num


def test_initialize_faiss_index(mock_shared_data):
    """Test the initialization of the FAISS index"""
    with patch('src.prompt_LLM.prompt_shots_selector.shared_data', mock_shared_data):
        index, embeddings = initialize_faiss_index()
        assert isinstance(index, faiss.IndexFlatIP)
        assert embeddings.shape == (len(MOCK_QUESTIONS), 384)


def test_prompt_example_generator(mock_shared_data):
    """Test the end-to-end prompt example generator"""
    # Setup test data
    mock_shared_data.faiss_index = build_faiss_index(MOCK_EMBEDDINGS)
    test_question = "What is the capital of Spain?"
    top_num = 2

    with patch('src.prompt_LLM.prompt_shots_selector.shared_data', mock_shared_data):
        with patch.object(MODEL, 'encode',
                          return_value=np.random.rand(1, 384).astype(np.float32)):
            result = prompt_example_generator(test_question, top_num)
            assert len(result) == top_num


def test_edge_cases():
    """Test various edge cases"""
    # Test with empty questions list
    with patch('src.prompt_LLM.prompt_shots_selector.shared_data.questions', []):
        with pytest.raises(ValueError):
            prepare_questions()

    # Test with top_num larger than available questions
    with patch('src.prompt_LLM.prompt_shots_selector.shared_data.questions', MOCK_QUESTIONS):
        with patch('src.prompt_LLM.prompt_shots_selector.generate_embeddings',
                   return_value=MOCK_EMBEDDINGS):
            index = build_faiss_index(MOCK_EMBEDDINGS)
            result = get_top_similar_questions_faiss("test question", index, 10)
            assert len(result) == len(MOCK_QUESTIONS)


def test_performance_large_dataset():
    """Test performance with a large dataset (sanity check)"""
    large_questions = [f"question {i}" for i in range(10000)]
    large_embeddings = np.random.rand(10000, 384).astype(np.float32)

    with patch('src.prompt_LLM.prompt_shots_selector.shared_data.questions', large_questions):
        with patch('src.prompt_LLM.prompt_shots_selector.generate_embeddings',
                   return_value=large_embeddings):
            # Time the index building
            import time
            start = time.time()
            index = build_faiss_index(large_embeddings)
            end = time.time()

            # Assert it completes in reasonable time (adjust threshold as needed)
            assert end - start < 10  # seconds

            # Test query performance
            test_question = "test question"
            with patch.object(MODEL, 'encode',
                              return_value=np.random.rand(1, 384).astype(np.float32)):
                start = time.time()
                similar_questions = get_top_similar_questions_faiss(
                    test_question, index, 5
                )
                end = time.time()

                assert len(similar_questions) == 5
                assert end - start < 1  # second