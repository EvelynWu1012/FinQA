"""

"""

import pytest
import shared_data
from data_loader import download_data
from preprocessor import preprocess_dataset
from prompt_example_selector import (
    prompt_example_generator,
    initialize_question_clusters,
)
from prompt_answer_gen_inference import MAX_SAMPLES


@pytest.fixture
def user_question():
    return ("by how much did total proved undeveloped reserves decrease "
            "during 2011?")


@pytest.fixture
def url():
    return "https://github.com/czyssrs/ConvFinQA/raw/main/data.zip"


# Function to run the test for the prompt example generator
def test_prompt_example_generator(user_question, url):
    # Ensure data is loaded (similar to the example pattern you shared)
    if not shared_data.questions:  # Check if questions are empty or not loaded
        print("Loading data...")
        raw = download_data(url)
        shared_data.processed_dataset = preprocess_dataset(raw, MAX_SAMPLES)
    else:
        print("Data already loaded.")

    # Step 2: Initialize clustering variables if not done
    if (not shared_data.question_to_cluster_label or not
    shared_data.cluster_idx_to_questions):
        print("Initializing clustering...")
        initialize_question_clusters()
    else:
        print("Clustering already initialized.")

    # Step 3: Run the prompt example generator
    top_num = 3
    top_examples = prompt_example_generator(user_question, top_num)

    # Step 4: Validate results
    assert isinstance(top_examples, list)
    assert len(top_examples) <= 3
    for q in top_examples:
        assert isinstance(q, str)
        assert q in shared_data.questions  # It should return questions that
        # exist in shared_data.questions

    print(f"\nTop 3 similar questions to '{user_question}':\n", top_examples)


# This part is the same as the example you provided, for running batch tests
if __name__ == "__main__":
    # In your example, you had something like this for batch processing
    print("Running tests...")

    # Run the single test
    test_prompt_example_generator(user_question(), url())

    # Run batch test on all questions (you can add more tests as needed)
    # You can iterate over questions or add more complex batch evaluations here
    # For simplicity, let's assume evaluating all questions is unnecessary
    # for now.
    print("Tests completed.")
