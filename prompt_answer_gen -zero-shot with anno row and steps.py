import json
from dotenv import load_dotenv
import os
import zipfile
from typing import Dict, List
import requests
import openai
import json



# Global variable to hold the max_samples value
MAX_SAMPLES = 3037


# =============================================================================
# Step 1: Preprocess Examples
# =============================================================================
def load_data(url: str, zip_file_path: str = "data.zip",
              extract_to: str = "data", json_file: str = "train.json"):
    """
    Downloads a zip file from the given URL, extracts it, and loads the
    specified JSON file.

    Args:
    - url (str): URL to download the zip file from.
    - zip_file_path (str): Path where the zip file will be saved.
    - extract_to (str): Directory to extract the contents of the zip file.
    - json_file (str): The name of the JSON file to load from the extracted
    folder.

    Returns:
    - data (dict): The data loaded from the specified JSON file.
    """
    # 1. Download the zip file
    print(f"Downloading {zip_file_path} from {url}...")
    # Send a GET request to the provided URL
    response = requests.get(url)
    # Open the specified path to save the zip file in write-binary mode.
    with open(zip_file_path, "wb") as f:
        # Write the content of the response(the downloaded zip file) to the
        # local file
        f.write(response.content)
    print(f"Download complete: {zip_file_path}")

    # 2. Unzip the file
    print(f"Unzipping {zip_file_path} into {extract_to}...")
    # Open the zip file in read mode
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        # Extract all the contents of the zip file into the specified directory
        zip_ref.extractall(extract_to)
    print(f"Unzip complete. Files extracted to: {extract_to}")

    # 3. Load the JSON file
    # Create the full path to the JSON file inside the extracted folder
    json_path = os.path.join(extract_to, "data", json_file)
    print(f"Loading JSON data from {json_path}...")
    with open(json_path, "r") as f:  # Open the JSON file in read mode
        # Load and parse the JSON data from the file into a Python dict
        data = json.load(f)
    print(f"Loaded {len(data)} examples from {json_file}.")

    return data


def preprocess_example(example: Dict) -> Dict:
    """
    For a single example, return a dictionary with the question as the key and
    associated metadata as the value.
    """
    processed_data = {}

    # Find ALL 'qa' keys (e.g., 'qa_0', 'qa_1', etc.)
    qa_keys = [key for key in example.keys() if key.startswith("qa")]

    if not qa_keys:
        raise KeyError("No 'qa' key found in the example.")

    # Process each QA pair
    for qa_key in qa_keys:
        question = example[qa_key]["question"]
        table = example["table"]
        table_header = table[0]
        ann_table_rows = example[qa_key].get("ann_table_rows", [])

        # Extract just the annotated rows for simplicity
        focused_rows_header = [table_header] + [table[i] for i in
                                                ann_table_rows]

        processed_data[question] = {
            "table": table,
            "focused_rows": focused_rows_header,
            "steps": example[qa_key].get("steps", []),
            "program": example[qa_key].get("program", ""),
            "exe_ans": example[qa_key].get("exe_ans"),
            "answer": example[qa_key].get("answer"),
        }

    return processed_data


def preprocess_dataset(data: List[Dict], max_samples) -> Dict:
    """
    Preprocess the entire dataset and return a dictionary where each key is
    a question
    and the corresponding value is the associated metadata.

    Args:
    - data (List[Dict]): The list of data examples.
    - max_samples (int): The maximum number of samples to process.

    Returns:
    - Dict: A dictionary with questions as keys and their corresponding
    metadata as values.
    """
    # Initialize an empty dictionary to hold the results
    processed_data = {}

    # Process each example
    for example in data[:max_samples]:
        # Get the preprocessed data for the example
        example_data = preprocess_example(example)

        # Since the key in the result is the question, we'll update the main
        # dictionary with the question as key
        processed_data.update(example_data)

    return processed_data


# =============================================================================
# Step 2: Set up LangChain Prompt Template
# =============================================================================
# Initialize OpenAI API Key
load_dotenv()  # This loads variables from .env into the environment
openai.api_key = os.getenv("OPENAI_API_KEY")


# print(f"Loaded API key: {openai.api_key[:5]}...")  # Don’t print the full key!


# Define a function for querying the processed data
def query_data(question: str, processed_data: Dict) -> str:
    """
    Given a question, return the associated context from the preprocessed data.
    """
    return processed_data.get(question,
                              {"error": "Data for this question not found."})


# Step 3: Create a LangChain Prompt Template
# Define a prompt template to generate an answer or code based on the
# question and context.
def format_prompt(question: str, context: Dict) -> str:
    """
    Formats a prompt by using context retrieved from the processed data.
    The prompt incorporates table information, focused rows, steps, and exe_ans.
    """
    if "error" in context:
        return context["error"]

    table_str = "\n".join(
        [", ".join(map(str, row)) for row in context["focused_rows"]])
    # Convert each step (a dict) to a formatted string using json.dumps
    steps_str = "\n".join(
        [json.dumps(step, indent=2) for step in context["steps"]]) if context[
        "steps"] else "N/A"

    prompt = f""" **Financial QA System Instructions**
    
    You are a helpful financial analysis assistant. Using the 
    table and reasoning details below, please write a Python-style program 
    that calculates the answer to the question and then provides the final 
    answer.

1. Analyze this question:
{question}
2. Use this table data: (focused rows):
{table_str}
3. Reasoning Steps:
{steps_str}
4. Please output:
- Program: function-style operations or function call expressions
- Answer: Just the final value as string with max.2 digits decimal
- Confidence: 0-100% certainty 

**Example Output Format:**
Program: such as "multiply(2.12, const_1000), add(#0, 112)"
Answer: such as "5.2", "-4.9%", "8.92%", "$ 378.7 million", "2232"
Confidence: 92%
"""
    return prompt.strip()


# =============================================================================
# Step 3: Initialize the LLM
# =============================================================================
def query_gpt(prompt: str) -> str:
    """
        Sends a prompt to the GPT-3.5-turbo model and returns the generated response.
        """
    client = openai.OpenAI()  # Create a client instance

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system",
             "content": "You are a helpful financial analysis assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content.strip()



# =============================================================================
# Step 4: Generate Output
# ============================================================================


if __name__ == "__main__":
    # Define the question for which we want to retrieve context and generate
    # an answer.
    question_text = ("for commercial mortgage recourse obligations , what "
                     "was average reserve adjustments net for 2010 and 2011 "
                     ", in millions?")

    # Download and load data
    url = "https://github.com/czyssrs/ConvFinQA/raw/main/data.zip"
    data = load_data(url)

    # Preprocess the dataset (limiting to a number of samples)
    processed = preprocess_dataset(data, max_samples=3037)

    # Retrieve context for the given question
    context = query_data(question_text, processed)
    print("context", context)

    # Format the prompt using the retrieved context
    prompt = format_prompt(question_text, context)
    print("prompt", prompt)

    # Display the prompt for debugging/inspection
    print("------ Prompt Sent to GPT ------\n")
    print(prompt)
    print("\n------ GPT-3.5 Response ------\n")

    # Send the prompt to GPT-3.5 and print the response
    response = query_gpt(prompt)
    print(response)

    print("\n------ Ground Truth ------")
    print("Expected Program:", context.get("program"))
    print("Expected Answer:", context.get("answer"))

