import os
import zipfile
import requests
import json

def download_data(url: str, zip_file_path: str = "data.zip", extract_to: str = "data") -> dict:
    """
    Downloads the dataset, extracts it, and loads the JSON file.
    """
    if not os.path.exists(zip_file_path):
        print(f"Downloading {zip_file_path} from {url}...")
        response = requests.get(url)
        with open(zip_file_path, "wb") as f:
            f.write(response.content)
        print(f"Download complete: {zip_file_path}")
    else:
        print(f"File {zip_file_path} already exists. Skipping download.")

    # Extract the data if not already extracted
    if not os.path.exists(extract_to):
        print(f"Unzipping {zip_file_path} into {extract_to}...")
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Unzip complete. Files extracted to: {extract_to}")
    else:
        print(f"Data already extracted. Skipping extraction.")

    # Load the JSON data
    json_path = os.path.join(extract_to, "data", "train.json")
    with open(json_path, "r") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} examples from {json_path}.")
    return data