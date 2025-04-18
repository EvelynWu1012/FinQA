from typing import Dict, Any
import pytest
import shared_data
from data_loader import download_data
from program_executor import execute_program
from preprocessor import preprocess_dataset
from prompt_answer_gen_inference import MAX_SAMPLES
from utils import clean_text


# Define a fixture for loading and preprocessing data


@pytest.fixture(scope="module")
def prepare_data():
    # Define the URL and data preprocessing logic
    url = "https://github.com/czyssrs/ConvFinQA/raw/main/data.zip"

    # Only load and preprocess the data if it's not already done
    if not shared_data.processed_dataset:
        print("Loading and preprocessing data...")
        raw = download_data(url)
        if raw is None:
            print("Failed to download data.")
            return None  # Return None or raise an exception depending on your preference

        shared_data.processed_dataset = preprocess_dataset(raw, MAX_SAMPLES)

        # Check if the processed dataset is populated
        if not shared_data.processed_dataset:
            print("Processed dataset is still empty after preprocessing.")
        else:
            print(
                f"Processed dataset has {len(shared_data.processed_dataset)} entries.")
    else:
        print("Data already loaded and preprocessed. Skipping...")

    # Yield the processed dataset so it can be used in tests
    return shared_data.processed_dataset


# Modify your test to use the fixture
@pytest.fixture
def user_question():
    return ("by how much did total proved undeveloped reserves decrease "
            "during 2011?")
def program():
    return "subtract(395, 405), divide(#0, 405)"



# This part is the same as the example you provided, for running batch tests
if __name__ == "__main__":
    # In your example, you had something like this for batch processing
    print("Running tests...")

    # Run the single test
    test_executor(program(), user_question())

    print("Tests completed.")
