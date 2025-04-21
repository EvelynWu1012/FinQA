# test_shared_utils.py
import pytest
import os
import numpy as np
import faiss
from unittest.mock import patch, MagicMock, mock_open
import psutil
import shutil

# Import the module to test
from src.shared.shared_utils import SharedData, shared_data
from src.shared.shared_data import MAX_SAMPLES, url, CACHE_DIR

# Test constants
TEST_CACHE_DIR = "test_cache"
TEST_INDEX_PATH = os.path.join(TEST_CACHE_DIR, "faiss.index")
TEST_EMBEDDINGS = np.random.rand(10, 384).astype(np.float32)


@pytest.fixture
def clean_cache_dir():
    """Fixture to create and clean up a test cache directory"""
    os.makedirs(TEST_CACHE_DIR, exist_ok=True)
    yield
    shutil.rmtree(TEST_CACHE_DIR, ignore_errors=True)


@pytest.fixture
def mock_shared_data():
    """Fixture to provide a clean SharedData instance for each test"""
    return SharedData()


def test_shared_data_initialization(mock_shared_data):
    """Test that SharedData initializes with empty attributes"""
    assert mock_shared_data.processed_dataset == {}
    assert mock_shared_data.questions == []
    assert mock_shared_data.question_embeddings is None
    assert mock_shared_data.faiss_index is None


def test_set_faiss_index(mock_shared_data):
    """Test setting FAISS index"""
    test_index = faiss.IndexFlatL2(384)
    mock_shared_data.set_faiss_index(test_index)
    assert mock_shared_data.faiss_index == test_index


def test_set_question_embeddings(mock_shared_data):
    """Test setting question embeddings"""
    mock_shared_data.set_question_embeddings(TEST_EMBEDDINGS)
    assert np.array_equal(mock_shared_data.question_embeddings,
                          TEST_EMBEDDINGS)


@patch('src.shared.shared_utils.cache_exists')
@patch('src.shared.shared_utils.load_cache')
@patch('src.shared.shared_utils.download_data')
@patch('src.shared.shared_utils.preprocess_dataset')
@patch('src.shared.shared_utils.save_cache')
def test_get_dataset_cached(mock_save_cache, mock_preprocess, mock_download,
                            mock_load_cache, mock_cache_exists,
                            mock_shared_data):
    """Test get_dataset with cached data available"""
    # Setup mocks
    mock_cache_exists.side_effect = lambda x: True
    mock_load_cache.side_effect = lambda x: {
        "key": "value"} if x == 'processed_dataset' else ["q1", "q2"]

    mock_shared_data.get_dataset(url)

    # Verify behavior
    assert mock_shared_data.processed_dataset == {"key": "value"}
    assert mock_shared_data.questions == ["q1", "q2"]
    mock_download.assert_not_called()
    mock_preprocess.assert_not_called()
    mock_save_cache.assert_not_called()


@patch('src.shared.shared_utils.cache_exists')
@patch('src.shared.shared_utils.load_cache')
@patch('src.shared.shared_utils.download_data')
@patch('src.shared.shared_utils.preprocess_dataset')
@patch('src.shared.shared_utils.save_cache')
def test_get_dataset_uncached(mock_save_cache, mock_preprocess, mock_download,
                              mock_load_cache, mock_cache_exists,
                              mock_shared_data):
    """Test get_dataset when no cached data exists"""
    # Setup mocks
    mock_cache_exists.return_value = False
    mock_download.return_value = "raw_data"
    mock_preprocess.return_value = {"processed": "data"}

    mock_shared_data.get_dataset(url)

    # Verify behavior
    mock_download.assert_called_once_with(url)
    mock_preprocess.assert_called_once_with("raw_data", MAX_SAMPLES)
    assert mock_save_cache.call_count == 2
    assert mock_shared_data.processed_dataset == {"processed": "data"}
    assert mock_shared_data.questions == ["processed"]


@patch('src.shared.shared_utils.faiss.write_index')
@patch('src.shared.shared_utils.save_cache')
def test_save_faiss_index(mock_save_cache, mock_write_index, mock_shared_data,
                          clean_cache_dir):
    """Test the save_faiss_index helper function"""
    test_index = faiss.IndexFlatL2(384)

    # We need to test the inner function, so we'll call get_search_index which uses it
    with patch('src.shared.shared_utils.cache_exists', return_value=False):
        with patch('src.shared.shared_utils.initialize_faiss_index',
                   return_value=(test_index, TEST_EMBEDDINGS)):
            mock_shared_data.get_search_index()

    # Verify the save operations
    mock_write_index.assert_called_once()
    mock_save_cache.assert_any_call('faiss_index',
                                    os.path.join(CACHE_DIR, "faiss.index"))
    mock_save_cache.assert_any_call('question_embeddings', TEST_EMBEDDINGS)


@patch('src.shared.shared_utils.cache_exists')
@patch('src.shared.shared_utils.load_cache')
@patch('src.shared.shared_utils.faiss.read_index')
@patch('src.shared.shared_utils.initialize_faiss_index')
def test_get_search_index_cached(mock_init_faiss, mock_read_index,
                                 mock_load_cache, mock_cache_exists,
                                 mock_shared_data, clean_cache_dir):
    """Test get_search_index with cached data available"""
    # Setup mocks
    mock_cache_exists.side_effect = lambda x: True
    mock_load_cache.side_effect = lambda \
        x: TEST_EMBEDDINGS if x == 'question_embeddings' else TEST_INDEX_PATH

    # Create a dummy index file
    with open(TEST_INDEX_PATH, 'wb') as f:
        f.write(b'dummy index data')

    mock_shared_data.get_search_index()

    # Verify behavior
    mock_read_index.assert_called_once_with(TEST_INDEX_PATH)
    mock_init_faiss.assert_not_called()
    assert np.array_equal(mock_shared_data.question_embeddings,
                          TEST_EMBEDDINGS)


@patch('src.shared.shared_utils.cache_exists')
@patch('src.shared.shared_utils.load_cache')
@patch('src.shared.shared_utils.faiss.read_index')
@patch('src.shared.shared_utils.initialize_faiss_index')
def test_get_search_index_missing_file(mock_init_faiss, mock_read_index,
                                       mock_load_cache, mock_cache_exists,
                                       mock_shared_data, clean_cache_dir):
    """Test get_search_index when index file is missing"""
    # Setup mocks
    mock_cache_exists.side_effect = lambda x: True
    mock_load_cache.side_effect = lambda \
        x: TEST_EMBEDDINGS if x == 'question_embeddings' else "missing_path.index"

    mock_shared_data.get_search_index()

    # Verify behavior
    mock_init_faiss.assert_called_once()
    mock_read_index.assert_not_called()


@patch('src.shared.shared_utils.cache_exists')
@patch('src.shared.shared_utils.load_cache')
@patch('src.shared.shared_utils.np.load')
@patch('src.shared.shared_utils.faiss.read_index')
def test_get_search_index_mmap_embeddings(mock_read_index, mock_np_load,
                                          mock_load_cache, mock_cache_exists,
                                          mock_shared_data, clean_cache_dir):
    """Test get_search_index with memory-mapped embeddings"""
    # Setup mocks
    mock_cache_exists.side_effect = lambda x: True
    mock_load_cache.side_effect = lambda \
        x: "embeddings.npy" if x == 'question_embeddings' else TEST_INDEX_PATH
    mock_np_load.return_value = TEST_EMBEDDINGS

    # Create a dummy index file
    with open(TEST_INDEX_PATH, 'wb') as f:
        f.write(b'dummy index data')

    mock_shared_data.get_search_index()

    # Verify behavior
    mock_np_load.assert_called_once_with("embeddings.npy", mmap_mode='r')
    assert np.array_equal(mock_shared_data.question_embeddings,
                          TEST_EMBEDDINGS)


def test_shared_data_singleton():
    """Test that shared_data is a singleton instance"""
    from src.shared.shared_utils import shared_data
    new_instance = SharedData()
    assert shared_data is not new_instance
    assert isinstance(shared_data, SharedData)


@patch('src.shared.shared_utils.psutil.virtual_memory')
def test_memory_handling(mock_virtual_memory, mock_shared_data):
    """Test behavior when system memory is low"""
    # Setup mock to simulate low memory
    mock_memory = MagicMock()
    mock_memory.available = 1 * 1024 * 1024 * 1024  # 1GB
    mock_virtual_memory.return_value = mock_memory

    with patch('src.shared.shared_utils.cache_exists', return_value=False):
        with patch('src.shared.shared_utils.download_data') as mock_download:
            # Should work normally
            mock_shared_data.get_dataset(url)
            mock_download.assert_called_once()

            # Now simulate very low memory
            mock_memory.available = 100 * 1024 * 1024  # 100MB

            with pytest.raises(MemoryError):
                mock_shared_data.get_dataset(url)


@patch('src.shared.shared_utils.os.path.exists')
@patch('src.shared.shared_utils.cache_exists')
def test_corrupted_cache_handling(mock_cache_exists, mock_path_exists,
                                  mock_shared_data):
    """Test handling of corrupted cache files"""
    # Simulate cache exists but file is corrupted/missing
    mock_cache_exists.return_value = True
    mock_path_exists.return_value = False

    with patch('src.shared.shared_utils.load_cache',
               side_effect=Exception("Corrupted file")):
        with patch('src.shared.shared_utils.download_data') as mock_download:
            with patch(
                    'src.shared.shared_utils.preprocess_dataset') as mock_preprocess:
                mock_shared_data.get_dataset(url)

                # Should fall back to downloading and processing
                mock_download.assert_called_once()
                mock_preprocess.assert_called_once()


def test_thread_safety(mock_shared_data):
    """Test that SharedData operations are thread-safe"""
    # Note: In a real test, you would use threading library to actually test concurrent access
    # This is a placeholder to remind to test thread safety

    # For actual implementation, you would:
    # 1. Create multiple threads
    # 2. Have them call SharedData methods concurrently
    # 3. Verify no race conditions or corruption occurs
    pass  # Actual implementation would go here


if __name__ == "__main__":
    pytest.main()