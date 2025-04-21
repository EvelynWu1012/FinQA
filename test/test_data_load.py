import pytest
import os
import zipfile
import shutil
from src.data_loader.data_loader import download_data

# Test configuration
TEST_ZIP_URL = "https://github.com/czyssrs/ConvFinQA/raw/main/data.zip"
TEST_ZIP_PATH = "test/data/raw/test_data.zip"
TEST_EXTRACT_PATH = "test/data/unzipped_test"


@pytest.fixture(autouse=True)
def cleanup():
    """Clean up test directories before and after each test"""
    # Setup: Ensure clean state
    if os.path.exists(TEST_ZIP_PATH):
        os.remove(TEST_ZIP_PATH)
    if os.path.exists(TEST_EXTRACT_PATH):
        shutil.rmtree(TEST_EXTRACT_PATH)

    yield  # Run the test

    # Teardown: Clean up after test
    if os.path.exists(TEST_ZIP_PATH):
        os.remove(TEST_ZIP_PATH)
    if os.path.exists(TEST_EXTRACT_PATH):
        shutil.rmtree(TEST_EXTRACT_PATH)


def test_download_and_extract_new_file():
    """Test downloading and extracting when files don't exist"""
    # Ensure test directories don't exist initially
    assert not os.path.exists(TEST_ZIP_PATH)
    assert not os.path.exists(TEST_EXTRACT_PATH)

    # Call the function with test paths
    data = download_data(TEST_ZIP_URL, TEST_ZIP_PATH, TEST_EXTRACT_PATH)

    # Verify files were created
    assert os.path.exists(TEST_ZIP_PATH)
    assert os.path.exists(TEST_EXTRACT_PATH)

    # Verify extraction worked by checking for expected files
    extracted_files = os.listdir(TEST_EXTRACT_PATH)
    assert "data" in extracted_files

    # Verify JSON data was loaded
    assert isinstance(data, list)
    assert len(data) > 0  # Ensure we got some data)


def test_skip_download_if_exists():
    """Test that download is skipped if VALID zip file exists"""
    # First do a proper download to get a valid zip file
    download_data(TEST_ZIP_URL, TEST_ZIP_PATH, TEST_EXTRACT_PATH)

    # Get the original file size and modification time
    original_size = os.path.getsize(TEST_ZIP_PATH)
    original_mtime = os.path.getmtime(TEST_ZIP_PATH)

    # Call the function again - should skip download
    data = download_data(TEST_ZIP_URL, TEST_ZIP_PATH, TEST_EXTRACT_PATH)

    # Verify the file wasn't modified
    assert os.path.getsize(TEST_ZIP_PATH) == original_size
    assert os.path.getmtime(TEST_ZIP_PATH) == original_mtime

    # Data should still be loaded correctly
    assert isinstance(data, list)
    assert len(data) > 0


def test_skip_extraction_if_exists():
    """Test that extraction is skipped if files already exist"""
    # First do a full download and extraction
    download_data(TEST_ZIP_URL, TEST_ZIP_PATH, TEST_EXTRACT_PATH)

    # Get modification time of zip file
    zip_mtime = os.path.getmtime(TEST_ZIP_PATH)

    # Call function again - should skip both download and extraction
    data = download_data(TEST_ZIP_URL, TEST_ZIP_PATH, TEST_EXTRACT_PATH)

    # Verify zip file wasn't modified
    assert os.path.getmtime(TEST_ZIP_PATH) == zip_mtime

    # Data should still be loaded correctly
    assert isinstance(data, list)
    assert len(data) > 0


def test_missing_json_raises_error(tmp_path):
    """Test that FileNotFoundError is raised if JSON is missing"""
    # Create a dummy zip file that won't contain the expected JSON
    dummy_zip = os.path.join(tmp_path, "dummy.zip")
    with zipfile.ZipFile(dummy_zip, 'w') as zipf:
        zipf.writestr("test.txt", "dummy content")

    with pytest.raises(FileNotFoundError):
        download_data(TEST_ZIP_URL, dummy_zip, TEST_EXTRACT_PATH)


def test_invalid_zip_gets_replaced():
    """Test that invalid zip files are detected and replaced"""
    # Create empty/invalid zip file
    os.makedirs(os.path.dirname(TEST_ZIP_PATH), exist_ok=True)
    with open(TEST_ZIP_PATH, 'wb') as f:
        f.write(b'')  # Empty file

    # Call the function - should detect invalid file and download fresh copy
    data = download_data(TEST_ZIP_URL, TEST_ZIP_PATH, TEST_EXTRACT_PATH)

    # Verify the file was replaced with a valid one (size changed)
    assert os.path.getsize(TEST_ZIP_PATH) > 0

    # Verify data was loaded correctly
    assert isinstance(data, list)
    assert len(data) > 0