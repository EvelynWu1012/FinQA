import os
import zipfile
import requests
import json


def download_data(url: str, zip_file_path: str = "data/raw/data.zip",
                  extract_to: str = "data/unzipped") -> dict:
    """
    Downloads the dataset, extracts it, and loads the JSON file.
    The zip file is saved to FinQA/data/raw,
    and its contents are extracted to FinQA/data/unzipped.
    """
    # Ensure directories exist
    os.makedirs(os.path.dirname(zip_file_path), exist_ok=True)
    os.makedirs(extract_to, exist_ok=True)

    if not os.path.exists(zip_file_path):
        print(f"Downloading {zip_file_path} from {url}...")
        response = requests.get(url)
        with open(zip_file_path, "wb") as f:
            f.write(response.content)
        print(f"Download complete: {zip_file_path}")
    else:
        print(f"File {zip_file_path} already exists. Skipping download.")

    # Extract only if folder is empty
    if not os.listdir(extract_to):
        print(f"Unzipping {zip_file_path} into {extract_to}...")
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Unzip complete. Files extracted to: {extract_to}")
    else:
        print(f"Data already extracted. Skipping extraction.")

    # Load the JSON data
    json_path = os.path.join(extract_to, "data", "train.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Could not find JSON file at {json_path}")
    with open(json_path, "r") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} examples from {json_path}.")
    return data
