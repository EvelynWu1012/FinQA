import pytest
from unittest import mock
import os
import requests
import json
from src.data_loader.data_loader import download_data

# Mock data for the test
mock_json_data = [{"id": 1, "text": "Example text"}]


@pytest.fixture
def mock_requests_get():
    """Fixture to mock requests.get"""
    with mock.patch("requests.get") as mock_get:
        yield mock_get


@pytest.fixture
def mock_os_makedirs():
    """Fixture to mock os.makedirs"""
    with mock.patch("os.makedirs") as mock_makedirs:
        yield mock_makedirs


@pytest.fixture
def mock_zipfile():
    """Fixture to mock zipfile.ZipFile"""
    with mock.patch("zipfile.ZipFile") as mock_zip:
        yield mock_zip


@pytest.fixture
def mock_open():
    """Fixture to mock open function"""
    with mock.patch("builtins.open", mock.mock_open(
            read_data=json.dumps(mock_json_data))) as mock_file:
        yield mock_file


@pytest.fixture
def mock_os_listdir():
    """Fixture to mock os.listdir"""
    with mock.patch("os.listdir") as mock_listdir:
        yield mock_listdir


@pytest.fixture
def mock_os_path_exists():
    """Fixture to mock os.path.exists"""
    with mock.patch("os.path.exists") as mock_exists:
        yield mock_exists


def test_download_data_new_file(mock_requests_get, mock_os_makedirs,
                                mock_zipfile, mock_open, mock_os_listdir,
                                mock_os_path_exists):
    """Test the download_data function when the zip file doesn't exist"""

    # Configure path.exists to return False for zip file but True for JSON path
    def exists_side_effect(path):
        if path == "data/raw/data.zip":
            return False
        if "train.json" in path:
            return True
        return False

    mock_os_path_exists.side_effect = exists_side_effect

    # Mock the response from requests.get
    mock_requests_get.return_value.status_code = 200
    mock_requests_get.return_value.content = b"fake zip content"

    # Mock the zipfile extraction
    mock_zipfile.return_value.__enter__.return_value.extractall = mock.Mock()

    # Mock os.listdir to simulate an empty directory for extraction
    mock_os_listdir.return_value = []

    # Call the function (this should trigger a download and extraction)
    data = download_data("http://example.com/data.zip")

    # Assert that the function attempts to download the file
    mock_requests_get.assert_called_once_with("http://example.com/data.zip")

    # Assert that os.makedirs was called for creating the necessary directories
    mock_os_makedirs.assert_any_call(os.path.dirname("data/raw/data.zip"),
                                     exist_ok=True)
    mock_os_makedirs.assert_any_call("data/unzipped", exist_ok=True)

    # Assert that zipfile extraction happens
    mock_zipfile.return_value.__enter__.return_value.extractall.assert_called_once_with(
        "data/unzipped")

    # Verify that the returned data matches the mock data
    assert data == mock_json_data


def test_download_data_existing_file(mock_requests_get, mock_os_makedirs,
                                     mock_zipfile, mock_open, mock_os_listdir,
                                     mock_os_path_exists):
    """Test the download_data function when the zip file already exists"""

    # Configure path.exists to return True for both zip and JSON paths
    def exists_side_effect(path):
        if path == "data/raw/data.zip" or "train.json" in path:
            return True
        return False

    mock_os_path_exists.side_effect = exists_side_effect

    # Mock os.listdir to simulate an empty directory for extraction
    mock_os_listdir.return_value = []

    # Call the function (no download should occur, but extraction and file loading should happen)
    data = download_data("http://example.com/data.zip")

    # Assert that requests.get was not called, since the file exists
    mock_requests_get.assert_not_called()

    # Ensure that zipfile extraction still occurs
    mock_zipfile.return_value.__enter__.return_value.extractall.assert_called_once_with(
        "data/unzipped")

    # Verify that the returned data matches the mock data
    assert data == mock_json_data


def test_download_data_no_json(mock_requests_get, mock_os_makedirs,
                               mock_zipfile, mock_os_listdir,
                               mock_os_path_exists):
    """Test the download_data function when the JSON file is missing"""

    # Configure path.exists to return True for zip but False for JSON
    def exists_side_effect(path):
        if path == "data/raw/data.zip":
            return True
        if "train.json" in path:
            return False
        return True

    mock_os_path_exists.side_effect = exists_side_effect

    # Mock the response from requests.get
    mock_requests_get.return_value.status_code = 200
    mock_requests_get.return_value.content = b"fake zip content"

    # Mock the zipfile extraction
    mock_zipfile.return_value.__enter__.return_value.extractall = mock.Mock()

    # Call the function and expect a FileNotFoundError to be raised
    with pytest.raises(FileNotFoundError):
        download_data("http://example.com/data.zip")