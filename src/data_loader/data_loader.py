import os
import zipfile
import requests
import json
from typing import Dict, List


def download_data(url: str, zip_file_path: str = "data/raw/data.zip",
                  extract_to: str = "data/unzipped") -> List[Dict]:
    """
    Downloads the dataset, extracts it, and loads the JSON file.
    The zip file is saved to FinQA/data/raw,
    and its contents are extracted to FinQA/data/unzipped.

    Args:
        url: URL to download the zip file from
        zip_file_path: Path to save the downloaded zip file
        extract_to: Directory to extract the zip contents to

    Returns:
        List of dictionaries containing the loaded JSON data

    Raises:
        FileNotFoundError: If the expected JSON file is not found after extraction
        requests.exceptions.RequestException: If there's an error downloading the file
    """
    # Ensure directories exist
    os.makedirs(os.path.dirname(zip_file_path), exist_ok=True)
    os.makedirs(extract_to, exist_ok=True)

    # Check if file exists and is a valid zip file
    download_needed = True
    if os.path.exists(zip_file_path):
        try:
            # Test if the file is a valid zip file
            with zipfile.ZipFile(zip_file_path, "r") as test_zip:
                # Check if the zip file is not empty
                if not test_zip.namelist():
                    raise zipfile.BadZipFile("Empty zip file")
            print(
                f"File {zip_file_path} already exists and is valid. Skipping download.")
            download_needed = False
        except zipfile.BadZipFile:
            print(
                f"Existing file {zip_file_path} is corrupt or invalid. Removing and downloading fresh copy...")
            os.remove(zip_file_path)

    if download_needed:
        print(f"Downloading {zip_file_path} from {url}...")
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise exception for bad status codes
            with open(zip_file_path, "wb") as f:
                f.write(response.content)
            print(f"Download complete: {zip_file_path}")
        except requests.exceptions.RequestException as e:
            raise requests.exceptions.RequestException(
                f"Failed to download file from {url}: {str(e)}")

    # Extract only if folder is empty
    if not os.listdir(extract_to):
        print(f"Unzipping {zip_file_path} into {extract_to}...")
        try:
            with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
                zip_ref.extractall(extract_to)
            print(f"Unzip complete. Files extracted to: {extract_to}")
        except zipfile.BadZipFile as e:
            # Clean up invalid zip file
            os.remove(zip_file_path)
            raise zipfile.BadZipFile(
                f"Invalid zip file {zip_file_path}: {str(e)}")
    else:
        print(f"Data already extracted. Skipping extraction.")

    # Load the JSON data
    json_path = os.path.join(extract_to, "data", "train.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Could not find JSON file at {json_path}")

    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        print(f"Loaded {len(data)} examples from {json_path}.")
        return data
    except json.JSONDecodeError as e:
        raise ValueError(f"Error parsing JSON file at {json_path}: {str(e)}")

if __name__ == "__main__":
    # URL to download the data
    url = "https://github.com/czyssrs/ConvFinQA/raw/main/data.zip"
    print("Downloading and extracting data...")
    data = download_data(url)
    for i, example in enumerate(data[:7]):
        print(f"Example {i}: {example}\n")